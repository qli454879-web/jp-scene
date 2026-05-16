#!/usr/bin/env python3
"""对游戏+日常重叠词条补全日常日语解析，形成双语境（日常用语解析 + 游戏术语解析）"""
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

# 非游戏标签
NON_GAMING_TAGS = [
    'loanword', 'kaoyan', 'onomatopoeia', 'slang',
    'jlpt_n1', 'jlpt_n2', 'jlpt_n3', 'jlpt_n4', 'jlpt_n5',
    'n1', 'n2', 'n3', 'yoji_jukugo',
]

# 游戏标签
GAMING_TAGS = [
    'gaming_valorant', 'gaming_lol',
    'gaming_val_agents', 'gaming_val_weapons', 'gaming_val_tactics',
    'gaming_val_settings', 'gaming_val_ranks', 'gaming_val_maps',
    'gaming_lol_champions', 'gaming_lol_items', 'gaming_lol_tactics',
    'gaming_lol_roles', 'gaming_lol_objectives',
]


def get_db_conn():
    return psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c plan_cache_mode=force_custom_plan -c statement_timeout=30000",
    )


def save_enrichment(conn, word_id, enriched: dict) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE vocab_library
                   SET social_context = %s,
                       heatmap_data = %s,
                       insight_text = %s,
                       frequency = %s,
                       examples = %s,
                       pos = %s,
                       meaning = %s,
                       is_ai_enriched = TRUE
                   WHERE id = %s""",
                (
                    json.dumps(enriched.get("social_context"), ensure_ascii=False),
                    json.dumps(enriched.get("heatmap_data"), ensure_ascii=False),
                    enriched.get("insight_text"),
                    enriched.get("frequency_stars"),
                    json.dumps(enriched.get("examples"), ensure_ascii=False),
                    enriched.get("pos_cn"),
                    enriched.get("meaning"),
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


def merge_examples(daily_examples, gaming_examples):
    """合并日常例句和游戏例句，去重，最多 4 条"""
    daily = daily_examples or []
    gaming = gaming_examples or []
    if isinstance(gaming, str):
        try:
            gaming = json.loads(gaming)
        except Exception:
            gaming = []

    merged = []
    seen_jp = set()

    for ex in daily:
        jp = (ex.get("jp") or "").strip()
        if jp and jp not in seen_jp:
            seen_jp.add(jp)
            merged.append({"jp": jp, "cn": (ex.get("cn") or "").strip()})

    for ex in gaming:
        jp = (ex.get("jp") or "").strip()
        if jp and jp not in seen_jp:
            seen_jp.add(jp)
            merged.append({"jp": jp, "cn": (ex.get("cn") or "").strip()})

    return merged[:4]


def get_game_label(tags):
    """从标签推断游戏名称"""
    tags_set = set(tags or [])
    if tags_set & {'gaming_valorant', 'gaming_val_agents', 'gaming_val_weapons',
                   'gaming_val_tactics', 'gaming_val_settings', 'gaming_val_ranks',
                   'gaming_val_maps'}:
        return '瓦罗兰特'
    if tags_set & {'gaming_lol', 'gaming_lol_champions', 'gaming_lol_items',
                   'gaming_lol_tactics', 'gaming_lol_roles', 'gaming_lol_objectives'}:
        return '英雄联盟'
    return ''


async def enrich_one(sem, conn, row):
    word_id, word, reading, meaning, tags, existing_insight, existing_examples = row

    # 跳过已包含双解析的词条
    if existing_insight and '日常用语解析' in existing_insight:
        print(f"  ⏭ 跳过（已有双解析）: {word}")
        return True

    async with sem:
        try:
            print(f"  🤖 分析: {word} ({(meaning or '')[:40]})")
            enriched = await ai.enrich_library_entry(
                word=word, reading=reading, meaning=meaning, category=None
            )
            if not enriched:
                print(f"  ✗ AI 返回空: {word}")
                return False

            daily_insight = (enriched.get("insight_text") or "").strip()
            daily_meaning = (enriched.get("meaning_cn") or "").strip()
            gaming_insight = (existing_insight or "").strip()
            gaming_meaning = (meaning or "").strip()

            # 构建双语境 insight_text
            sections = []
            if daily_insight:
                sections.append(f"日常用语解析\n{daily_insight}")
            if gaming_insight:
                sections.append(f"游戏术语解析\n{gaming_insight}")
            merged_insight = "\n\n".join(sections)

            # 合并 meaning：日常释义为主，游戏释义追加
            if daily_meaning and gaming_meaning and daily_meaning != gaming_meaning:
                merged_meaning = f"{daily_meaning}（游戏术语: {gaming_meaning}）"
            elif daily_meaning:
                merged_meaning = daily_meaning
            else:
                merged_meaning = gaming_meaning

            # 合并例句
            daily_examples = enriched.get("examples") or []
            merged_examples = merge_examples(daily_examples, existing_examples)

            if save_enrichment(conn, word_id, {
                "insight_text": merged_insight,
                "meaning": merged_meaning,
                "examples": merged_examples,
                "social_context": enriched.get("social_context"),
                "heatmap_data": enriched.get("heatmap_data"),
                "frequency_stars": enriched.get("frequency_stars"),
                "pos_cn": enriched.get("pos_cn"),
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

    # 查询同时有游戏标签和非游戏标签的词条，且 insight 只有游戏内容
    gaming_arr = "ARRAY[" + ",".join(f"'{t}'" for t in GAMING_TAGS) + "]"
    nongaming_arr = "ARRAY[" + ",".join(f"'{t}'" for t in NON_GAMING_TAGS) + "]"
    cur.execute(f"""
        SELECT id, word, reading, meaning, tags, insight_text, examples
        FROM vocab_library
        WHERE is_ai_enriched = TRUE
          AND tags && {gaming_arr}
          AND tags && {nongaming_arr}
        ORDER BY word
    """)
    rows = cur.fetchall()
    conn.close()

    # 分类过滤
    # gaming-only: 只有【原意】/【游戏中含义】格式，没有【游戏用语】附录（需要AI生成日常分析）
    # already_merged: 已有【游戏用语】附录（源自 merge_duplicates 策略A），日常分析已存在
    # everyday_only: 纯日常解析（极少，来自策略B反例）
    # 已存在目标格式: 已有 日常用语解析 或 游戏术语解析 标题
    original_total = len(rows)
    target_rows = []
    skip_already_merged = 0
    skip_has_dual = 0
    skip_other = 0

    for r in rows:
        insight = r[5] or ''
        # 已有目标格式
        if '日常用语解析' in insight or '游戏术语解析' in insight:
            skip_has_dual += 1
            continue
        # 已有【游戏用语】附录 = merge脚本策略A处理过的，日常分析已存在
        if '【游戏用语】' in insight:
            skip_already_merged += 1
            continue
        # 只处理有游戏格式但无日常分析的
        if '【游戏中含义】' in insight:
            target_rows.append(r)
        else:
            skip_other += 1

    print(f"重叠词条总数: {original_total}")
    print(f"已有双解析标题: {skip_has_dual}")
    print(f"已有合并格式（日常+游戏附录）: {skip_already_merged}")
    print(f"其他格式: {skip_other}")
    print(f"待AI生成日常分析: {len(target_rows)}")

    if args.limit > 0:
        target_rows = target_rows[:args.limit]

    if args.dry_run:
        for r in target_rows:
            game = get_game_label(r[4])
            print(f"  {r[1]} | {(r[3] or '')[:60]} | 游戏: {game or '无'}")
        print(f"\nDRY RUN. 使用不带 --dry-run 运行以执行。")
        return

    if ai.provider == "mock":
        print("无 AI Key，退出。")
        return

    print(f"AI Provider: {ai.provider}")
    print(f"并发: {CONCURRENCY}")

    conn = get_db_conn()
    sem = asyncio.Semaphore(CONCURRENCY)

    success = 0
    failed = 0
    total = len(target_rows)

    async def run_one(idx, row):
        nonlocal success, failed
        ok = await enrich_one(sem, conn, row)
        if ok:
            success += 1
        else:
            failed += 1
        print(f"  进度: {success + failed}/{total}")

    # 并发批处理，每批最多 CONCURRENCY 个
    for i in range(0, total, CONCURRENCY):
        batch = target_rows[i:i + CONCURRENCY]
        tasks = [run_one(i + j, row) for j, row in enumerate(batch)]
        await asyncio.gather(*tasks, return_exceptions=True)

    conn.close()
    print(f"\n完成: 成功 {success}, 失败 {failed}")


if __name__ == "__main__":
    asyncio.run(main())
