#!/usr/bin/env python3
"""重试之前跳过的 22 条词"""
import os, sys, json, asyncio, re, traceback

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

ai = AIService()

SKIPPED_WORDS = [
    # AI empty
    "アーバンスプロール", "ガスパネル", "クリエイティブコモンズ",
    "ノングレア", "ベニオチョウチョウウオ", "ボソン",
    "偽面梟", "共同絶交", "寄金", "平幕", "木地", "洒落本", "薄暑",
    "順建て", "頓服",
    # Still JP
    "サプレス", "ナッシャートゥース", "ネクサス", "ラーク",
    "薩埵", "陽線", "顆粒性",
]

JP_SENTENCE_PATTERNS = [
    r'です', r'ます', r'しました', r'している', r'されている',
    r'である', r'という', r'しない', r'できない',
]

def has_jp_sentences(text: str) -> bool:
    if not text:
        return False
    count = sum(1 for p in JP_SENTENCE_PATTERNS if re.search(p, text))
    return count >= 2

def kana_ratio(text: str) -> float:
    if not text:
        return 0.0
    kana = len(re.findall(r'[ぁ-んァ-ン]', text))
    return kana / len(text) if len(text) > 0 else 0.0

async def retry_one(sem, conn, word_id, word, reading, meaning, stats):
    async with sem:
        try:
            enriched = await ai.enrich_library_entry(
                word=word, reading=reading, meaning=meaning, category=None
            )
            if not enriched:
                print(f"  ✗ AI empty: {word}")
                stats["empty"] += 1
                return

            new_insight = (enriched.get("insight_text") or "").strip()
            if not new_insight:
                print(f"  ✗ AI empty: {word}")
                stats["empty"] += 1
                return

            kr = kana_ratio(new_insight)
            is_jp = has_jp_sentences(new_insight)

            if is_jp:
                print(f"  ⚠ Still JP: {word} ({kr:.0%} kana)")
                stats["still_jp"] += 1
                return
            if kr > 0.5:
                print(f"  ⚠ High kana: {word} ({kr:.0%})")
                stats["still_jp"] += 1
                return

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE vocab_library SET insight_text = %s WHERE id = %s",
                    (new_insight, word_id),
                )
            conn.commit()
            print(f"  ✓ {word}")
            stats["fixed"] += 1

        except Exception as e:
            print(f"  ✗ {word}: {e}")
            stats["error"] += 1


async def main():
    conn = psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c statement_timeout=30000",
    )
    cur = conn.cursor()

    placeholders = ",".join(["%s"] * len(SKIPPED_WORDS))
    cur.execute(
        f"SELECT id, word, COALESCE(reading,''), COALESCE(meaning,'') FROM vocab_library WHERE word IN ({placeholders})",
        SKIPPED_WORDS,
    )
    rows = cur.fetchall()
    print(f"找到 {len(rows)}/{len(SKIPPED_WORDS)} 条\n")

    stats = {"fixed": 0, "empty": 0, "still_jp": 0, "error": 0}
    sem = asyncio.Semaphore(10)

    tasks = [
        retry_one(sem, conn, row[0], row[1], row[2], row[3], stats)
        for row in rows
    ]
    await asyncio.gather(*tasks)

    print(f"\n结果: 修复 {stats['fixed']} | AI空 {stats['empty']} | 仍日语 {stats['still_jp']} | 异常 {stats['error']}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
