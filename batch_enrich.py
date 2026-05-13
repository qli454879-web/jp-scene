"""批量 AI 加工词库：对未 enrichment 的词按优先级生成 social_context + heatmap + insight"""
import asyncio
import os
import sys
import logging
import json as _json

# 直接在 .env 文件读环境变量
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_service import AIService
import psycopg
import psycopg.rows
import pykakasi as _pykakasi

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

DRY_RUN = os.getenv("BATCH_ENRICH_DRY", "") == "1"
BATCH_SIZE = 200          # 每批 AI 并发数
CONCURRENCY = 60         # AI API 并发上限（降一半给前端让路）
DB_SUB_BATCH = 50         # 每 N 个词提交一次 DB（断点安全）
MAX_TOTAL = int(os.getenv("BATCH_ENRICH_MAX", "200000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("batch_enrich")

KATA_CHARS = set(range(0x30A0, 0x30FF + 1)) | {0x30FC}

def is_katakana(s):
    return bool(s) and all(ord(c) in KATA_CHARS for c in s)

def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, options="-c plan_cache_mode=force_custom_plan", prepare_threshold=None)


async def enrich_one(ai, sem, w):
    """单个词的 AI 加工，只做 AI 调用 + 数据整理，不碰 DB"""
    async with sem:
        try:
            result = await ai.enrich_library_entry(
                word=w["word"],
                reading=w.get("reading") or "",
                meaning=w.get("meaning") or "",
                level=w.get("level") or "N2",
            )
            if not result:
                log.warning(f"    ✗ {w['word']}: AI 返回空")
                return None

            sc = _json.dumps(result.get("social_context") or {}, ensure_ascii=False)
            hm = _json.dumps(result.get("heatmap_data") or {}, ensure_ascii=False)
            it = (result.get("insight_text") or "").replace("\x00", "")
            fs = result.get("frequency_stars")
            ex = result.get("examples")
            ex_json = _json.dumps(ex, ensure_ascii=False) if ex else None
            meaning_cn = result.get("meaning_cn")
            pos_cn = result.get("pos_cn")
            pitch = result.get("pitch")

            # 片假名外来语：保留原文 + 中文翻译
            final_meaning = meaning_cn
            if is_katakana(w["word"]) and meaning_cn:
                orig = (w.get("meaning") or "").strip()
                orig_first = orig.split(" / ")[0].split("；")[0].split(";")[0].strip()
                if orig_first and not any('一' <= c <= '鿿' for c in orig_first):
                    final_meaning = f"{orig_first}，{meaning_cn}"

            # 频率 → 等级映射
            freq_to_level = {5: "常用", 4: "高频", 3: "一般", 2: "生僻", 1: "罕用"}
            new_level = freq_to_level.get(fs) if isinstance(fs, int) else None

            return {
                "id": w["id"],
                "word": w["word"],
                "level": w.get("level"),
                "sc": sc,
                "hm": hm,
                "it": it,
                "fs": fs,
                "ex_json": ex_json,
                "meaning": final_meaning,
                "pos": pos_cn,
                "pitch": pitch,
                "new_level": new_level,
            }

        except Exception as e:
            log.error(f"    ✗ {w['word']}: {e}")
            return None


UPDATE_SQL = """
    UPDATE vocab_library
    SET social_context = %(sc)s::jsonb,
        heatmap_data = %(hm)s::jsonb,
        insight_text = %(it)s,
        frequency = %(fs)s,
        examples = %(ex_json)s::jsonb,
        meaning = COALESCE(%(meaning)s, meaning),
        pos = COALESCE(%(pos)s, pos),
        pitch = COALESCE(%(pitch)s, pitch),
        level = CASE WHEN level = '未分级' THEN COALESCE(%(new_level)s, level) ELSE level END,
        is_ai_enriched = TRUE
    WHERE id = %(id)s
"""


def flush_db(updates):
    """批量写入 DB，每批一个事务"""
    if not updates:
        return 0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for u in updates:
                cur.execute(UPDATE_SQL, u)
        conn.commit()
        return len(updates)
    except Exception as e:
        log.error(f"DB 批量写入失败: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        # 逐条重试
        saved = 0
        for u in updates:
            try:
                rconn = get_conn()
                with rconn.cursor() as cur:
                    cur.execute(UPDATE_SQL, u)
                rconn.commit()
                rconn.close()
                saved += 1
            except Exception as e2:
                log.error(f"    逐条重试失败 {u['word']}: {e2}")
        return saved
    finally:
        conn.close()


async def main():
    if not SUPABASE_DB_URL:
        log.error("SUPABASE_DB_URL not configured")
        return

    ai = AIService()
    sem = asyncio.Semaphore(CONCURRENCY)

    conn = get_conn()
    total_processed = 0
    total_success = 0

    try:
        for priority_level in ("N5", "N4", "N3", "未分级"):
            if total_processed >= MAX_TOTAL:
                break

            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, word, reading, meaning, level
                    FROM vocab_library
                    WHERE is_ai_enriched = FALSE AND level = %s
                      AND (level != '未分级' OR (
                        LENGTH(meaning) < 100
                        AND meaning !~* 'archaic|obsolete|ancient|era|period|system|surname|buddhist|shinto|ritual|ceremony'
                      ))
                    ORDER BY CASE WHEN level = 'N2' THEN LENGTH(meaning) ELSE 0 END ASC,
                             LENGTH(meaning) DESC
                    LIMIT %s
                    """,
                    (priority_level, min(MAX_TOTAL - total_processed, MAX_TOTAL)),
                )
                words = cur.fetchall()

            if not words:
                log.info(f"[{priority_level}] 全部已加工，跳过")
                continue

            limit = min(len(words), MAX_TOTAL - total_processed)
            log.info(f"[{priority_level}] 待加工 {len(words)} 词，本次处理 {limit}")

            words = words[:limit]

            for batch_idx in range(0, len(words), BATCH_SIZE):
                batch = words[batch_idx:batch_idx + BATCH_SIZE]

                batch_no = batch_idx // BATCH_SIZE + 1
                log.info(f"  批次 {batch_no}: {len(batch)} 词 — {', '.join(w['word'] for w in batch[:5])}...")

                if DRY_RUN:
                    for w in batch:
                        log.info(f"    [DRY] {w['word']} [{w['level']}]")
                    total_processed += len(batch)
                    continue

                # Step 1: 并发 AI 加工
                tasks = [enrich_one(ai, sem, w) for w in batch]
                results = await asyncio.gather(*tasks)

                # Step 2: 收集有效结果
                updates = [r for r in results if r is not None]

                # Step 3: 分批写 DB（每 DB_SUB_BATCH 提交一次，断点安全）
                saved = 0
                for sub_start in range(0, len(updates), DB_SUB_BATCH):
                    sub = updates[sub_start:sub_start + DB_SUB_BATCH]
                    saved += flush_db(sub)

                total_processed += len(batch)
                total_success += saved
                log.info(f"   批次完成: {saved}/{len(batch)} 成功 (总成功 {total_success}/{total_processed})")

        # 补漏：重试剩余未加工的词，直到全部完成（最多10轮）
        retry_round = 0
        while total_processed < MAX_TOTAL:
            retry_round += 1
            if retry_round > 10:
                log.info("补漏已达10轮，停止")
                break

            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, word, reading, meaning, level
                    FROM vocab_library
                    WHERE is_ai_enriched = FALSE
                    ORDER BY LENGTH(meaning) ASC
                    LIMIT %s
                    """,
                    (MAX_TOTAL - total_processed,),
                )
                words = cur.fetchall()

            if not words:
                log.info("全部词已加工完毕！")
                break

            log.info(f"[补漏第{retry_round}轮] 剩余 {len(words)} 词")
            prev_success = total_success

            for batch_idx in range(0, len(words), BATCH_SIZE):
                batch = words[batch_idx:batch_idx + BATCH_SIZE]
                batch_no = batch_idx // BATCH_SIZE + 1

                if DRY_RUN:
                    for w in batch:
                        log.info(f"    [DRY] {w['word']} [{w['level']}]")
                    total_processed += len(batch)
                    continue

                tasks = [enrich_one(ai, sem, w) for w in batch]
                results = await asyncio.gather(*tasks)
                updates = [r for r in results if r is not None]

                saved = 0
                for sub_start in range(0, len(updates), DB_SUB_BATCH):
                    sub = updates[sub_start:sub_start + DB_SUB_BATCH]
                    saved += flush_db(sub)

                total_success += saved
                log.info(f"  补漏批次{batch_no}: {saved}/{len(batch)} (总 {total_success})")

            if total_success == prev_success:
                log.info(f"补漏第{retry_round}轮无新增成功，停止重试")
                break

    finally:
        conn.close()

    log.info(f"完成。共处理 {total_processed} 词，成功 {total_success}。DRY_RUN={DRY_RUN}")


if __name__ == "__main__":
    asyncio.run(main())
