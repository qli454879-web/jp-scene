#!/usr/bin/env python3
"""修复所有缺失「日常用语解析 / 游戏术语解析」双标题的游戏相关词条（含纯游戏词条）"""
import os, sys, json, asyncio, argparse, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

ENV_FILE = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import psycopg
from ai_service import AIService

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "5"))

ai = AIService()

NON_GAMING_TAGS = [
    'loanword', 'kaoyan', 'onomatopoeia', 'slang',
    'jlpt_n1', 'jlpt_n2', 'jlpt_n3', 'jlpt_n4', 'jlpt_n5',
    'n1', 'n2', 'n3', 'yoji_jukugo',
]

GAMING_TAGS = [
    'gaming_valorant', 'gaming_lol',
    'gaming_val_agents', 'gaming_val_weapons', 'gaming_val_tactics',
    'gaming_val_settings', 'gaming_val_ranks', 'gaming_val_maps',
    'gaming_lol_champions', 'gaming_lol_items', 'gaming_lol_tactics',
    'gaming_lol_roles', 'gaming_lol_objectives',
]

GAMING_MARKERS = ['【游戏中含义】', '【原意】', '【游戏中使用方式】', '【对应中文术语】', '【游戏用语】']


def get_db_conn():
    return psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c plan_cache_mode=force_custom_plan -c statement_timeout=30000",
    )


def save_enrichment(conn, word_id, insight_text: str, meaning: str, enriched: dict) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE vocab_library
                   SET insight_text = %s,
                       meaning = %s,
                       social_context = %s,
                       heatmap_data = %s,
                       frequency = %s,
                       examples = %s,
                       is_ai_enriched = TRUE
                   WHERE id = %s""",
                (
                    insight_text,
                    meaning,
                    json.dumps(enriched.get("social_context"), ensure_ascii=False),
                    json.dumps(enriched.get("heatmap_data"), ensure_ascii=False),
                    enriched.get("frequency_stars"),
                    json.dumps(enriched.get("examples"), ensure_ascii=False),
                    word_id,
                ),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"  保存失败 (id={word_id}): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def merge_examples(daily_examples, existing_examples):
    daily = daily_examples or []
    existing = existing_examples or []
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except Exception:
            existing = []
    merged = []
    seen_jp = set()
    for ex in daily:
        jp = (ex.get("jp") or "").strip()
        if jp and jp not in seen_jp:
            seen_jp.add(jp)
            merged.append({"jp": jp, "cn": (ex.get("cn") or "").strip()})
    for ex in existing:
        jp = (ex.get("jp") or "").strip()
        if jp and jp not in seen_jp:
            seen_jp.add(jp)
            merged.append({"jp": jp, "cn": (ex.get("cn") or "").strip()})
    return merged[:4]


def has_gaming_markers(text: str) -> bool:
    return any(m in text for m in GAMING_MARKERS)


def build_gaming_fallback(word: str, existing_insight: str, existing_meaning: str) -> str:
    """当现有内容不是游戏格式时，构造简化的游戏术语解析"""
    game_name = ""
    for t in GAMING_TAGS:
        if t.startswith('gaming_val'):
            game_name = "VALORANT"
            break
        if t.startswith('gaming_lol'):
            game_name = "英雄联盟"
            break
    return f"（游戏术语: {existing_meaning}）\n该词在{game_name}等游戏中作为专业术语使用。"


async def enrich_one(sem, conn, row):
    word_id, word, reading, meaning, tags, existing_insight, existing_examples = row

    async with sem:
        try:
            print(f"  🤖 AI 生成日常解析: {word} ({(meaning or '')[:40]})")

            existing = (existing_insight or "").strip()
            existing_has_gaming = has_gaming_markers(existing)

            # AI 生成日常分析（category=None = everyday context）
            enriched = await ai.enrich_library_entry(
                word=word, reading=reading, meaning=meaning, category=None
            )
            if not enriched:
                print(f"  ✗ AI 返回空: {word}")
                return False

            daily_insight = (enriched.get("insight_text") or "").strip()
            daily_meaning = (enriched.get("meaning_cn") or "").strip()

            # 构造游戏术语解析部分
            tags_set = set(tags or [])
            is_gaming_only = not tags_set.intersection(NON_GAMING_TAGS)

            if existing_has_gaming:
                # 保留现有内容作为游戏部分（已有【游戏中含义】等标记）
                game_insight = existing
            elif is_gaming_only and existing:
                # 纯游戏词条，现有内容就是游戏术语分析
                game_insight = existing
            elif len(existing) > 50:
                # 现有内容较长，保留作为游戏术语参考
                game_insight = existing
            else:
                game_insight = ""

            # 合并为双标题格式
            sections = []
            if daily_insight:
                sections.append(f"日常用语解析\n{daily_insight}")
            if game_insight:
                sections.append(f"游戏术语解析\n{game_insight}")
            merged_insight = "\n\n".join(sections) if sections else daily_insight

            # 合并 meaning
            gaming_meaning = (meaning or "").strip()
            if daily_meaning and gaming_meaning and daily_meaning != gaming_meaning:
                merged_meaning = f"{daily_meaning}（游戏术语: {gaming_meaning}）"
            elif daily_meaning:
                merged_meaning = daily_meaning
            else:
                merged_meaning = gaming_meaning

            # 合并例句
            daily_examples = enriched.get("examples") or []
            merged_examples = merge_examples(daily_examples, existing_examples)

            if save_enrichment(conn, word_id, merged_insight, merged_meaning, {
                "social_context": enriched.get("social_context"),
                "heatmap_data": enriched.get("heatmap_data"),
                "frequency_stars": enriched.get("frequency_stars"),
                "examples": merged_examples,
            }):
                print(f"  ✓ 完成: {word}")
                return True
            print(f"  ✗ 保存失败: {word}")
            return False

        except Exception as e:
            print(f"  ✗ 异常 {word}: {e}")
            traceback.print_exc()
            return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conn = get_db_conn()
    cur = conn.cursor()

    gaming_arr = "ARRAY[" + ",".join(f"'{t}'" for t in GAMING_TAGS) + "]"

    cur.execute(f"""
        SELECT id, word, reading, meaning, tags, insight_text, examples
        FROM vocab_library
        WHERE is_ai_enriched = TRUE
          AND tags && {gaming_arr}
          AND (insight_text IS NULL
               OR insight_text = ''
               OR (insight_text NOT LIKE '%日常用语解析%'
                   AND insight_text NOT LIKE '%游戏术语解析%'))
        ORDER BY word
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"缺失双标题的游戏词条: {len(rows)}")

    if args.dry_run:
        for r in rows:
            has_gm = has_gaming_markers(r[5] or "")
            print(f"  {r[1]} | has_gaming_markers={has_gm} | {(r[5] or '')[:80]}")
        return

    if ai.provider == "mock":
        print("无 AI Key，退出。")
        return

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"AI Provider: {ai.provider}")
    print(f"并发: {CONCURRENCY}")

    conn = get_db_conn()
    sem = asyncio.Semaphore(CONCURRENCY)

    success = 0
    failed = 0
    total = len(rows)

    for i in range(0, total, CONCURRENCY):
        batch = rows[i:i + CONCURRENCY]
        tasks = []
        for j, row in enumerate(batch):
            async def run_one(idx, r):
                nonlocal success, failed
                ok = await enrich_one(sem, conn, r)
                if ok:
                    success += 1
                else:
                    failed += 1
                print(f"  进度: {success + failed}/{total}")
            tasks.append(run_one(i + j, row))
        await asyncio.gather(*tasks, return_exceptions=True)

    conn.close()
    print(f"\n完成: 成功 {success}, 失败 {failed}")


if __name__ == "__main__":
    asyncio.run(main())
