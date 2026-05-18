#!/usr/bin/env python3
"""Normalize JLPT tags and expand onomatopoeia with precise filtering."""
import os, sys, re

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

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

def is_onomatopoeia(word: str, reading: str, meaning: str) -> bool:
    """Strict onomatopoeia detection to avoid false positives on loanwords."""
    w = word.strip()
    # Must be pure katakana
    if not re.match(r'^[ァ-ヴー]+$', w):
        return False
    # Exclude single char + ー
    if re.match(r'^[ア-ヴ]ー$', w):
        return False
    # Exclude loanword markers: words that start with アー, エー, オー, イー (long vowels at start = loanword)
    if re.match(r'^[アエオイ]ー', w):
        # Check if it looks like ABAB onomatopoeia despite long vowel start
        if not re.match(r'^(..)\1$', w):
            return False
    # Exclude words that are clearly English loanwords (contain certain patterns)
    if re.search(r'(ティ|ディ|ファ|フィ|フェ|フォ|ヴァ|ヴィ|ヴェ|ヴォ|ツー|トゥ|ドゥ|グァ|グィ|グェ|グォ|スィ|ズィ|シェ|ジェ|チェ|ティー|ディー)', w):
        return False
    # Exclude long words (5+ kana) that aren't ABAB
    if len(w) >= 5 and not re.match(r'^(..)\1$', w) and not re.match(r'^(..)\1..$', w):
        return False

    # ── Core onomatopoeia patterns ──
    # ABAB exact: ドキドキ, ワクワク, モリモリ
    if re.match(r'^(..)\1$', w):
        return True

    # AッBリ: ハッキリ, ガッカリ, シッカリ, スッキリ, バッチリ, コッソリ
    if re.match(r'^.[ッ].リ$', w):
        return True

    # AンBリ: ボンヤリ, ノンビリ, ゲンナリ, シンミリ, ウットリ, ハッキリ
    if re.match(r'^.[ン].リ$', w):
        return True

    # Short, ends with っ: グッ, バッ, ピッ, ムッ, ホッ
    if re.match(r'^.[ァ-ヴ]ッ$', w):
        return True

    # Short, ends with ん: ガツン, ゴロン, コロン, ドキン, ポロン (but exclude long loanwords)
    if re.match(r'^.{2,3}ン$', w) and len(w) <= 4:
        return True

    # Short, ends with り: コロリ, ペタリ, テカリ, ユラリ (but exclude long words)
    if re.match(r'^.{2,3}リ$', w) and len(w) <= 4:
        return True

    return False


def main():
    conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c statement_timeout=120000")
    conn.autocommit = False
    cur = conn.cursor()
    stats = {}

    # ━━━ 1. Normalize JLPT tags ━━━
    print("=" * 50)
    print("1. Normalizing JLPT tags")
    print("=" * 50)

    jlpt_merge = [
        ('n1', 'jlpt_n1'),
        ('n2', 'jlpt_n2'),
        ('n3', 'jlpt_n3'),
    ]
    for old_tag, new_tag in jlpt_merge:
        cur.execute(
            "SELECT count(*) FROM vocab_library WHERE %s = ANY(tags) AND NOT (%s = ANY(tags))",
            (old_tag, new_tag)
        )
        cnt = cur.fetchone()[0]
        if cnt > 0:
            cur.execute(
                "UPDATE vocab_library SET tags = array_append(tags, %s) "
                "WHERE %s = ANY(tags) AND NOT (%s = ANY(tags))",
                (new_tag, old_tag, new_tag)
            )
            conn.commit()
            print(f"  {old_tag} → {new_tag}: {cnt} updated")
        else:
            print(f"  {old_tag} → {new_tag}: already normalized")
        stats[f'jlpt_merge_{old_tag}'] = cnt

    # Show final JLPT counts
    print("\nFinal JLPT counts:")
    total_jlpt = 0
    for tag in ['jlpt_n1', 'jlpt_n2', 'jlpt_n3', 'jlpt_n4', 'jlpt_n5']:
        cur.execute(f"SELECT count(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
        cnt = cur.fetchone()[0]
        total_jlpt += cnt
        print(f"  {tag}: {cnt}")
    print(f"  TOTAL (unique may overlap): {total_jlpt}")

    # ━━━ 2. Expand onomatopoeia ━━━
    print("\n" + "=" * 50)
    print("2. Expanding onomatopoeia (strict filtering)")
    print("=" * 50)

    cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags)")
    current = cur.fetchone()[0]
    print(f"Current onomatopoeia: {current}")

    # Fetch kana-only words matching broad patterns
    cur.execute("""
        SELECT id, word, COALESCE(reading,''), COALESCE(meaning,'') FROM vocab_library
        WHERE word ~ '^[ァ-ヴー]{2,8}$'
          AND word !~ '^[ア-ヴ]ー$'
          AND NOT ('onomatopoeia' = ANY(tags))
    """)
    candidates = cur.fetchall()
    print(f"Kana-only candidates to check: {len(candidates)}")

    # Filter with strict onomatopoeia detection
    new_onomatopoeia_ids = []
    for row_id, word, reading, meaning in candidates:
        if is_onomatopoeia(word, reading, meaning):
            new_onomatopoeia_ids.append(row_id)

    print(f"Passed strict filter: {len(new_onomatopoeia_ids)}")

    # Batch update
    if new_onomatopoeia_ids:
        for i in range(0, len(new_onomatopoeia_ids), 500):
            chunk = new_onomatopoeia_ids[i:i+500]
            cur.execute(
                "UPDATE vocab_library SET tags = array_append(tags, 'onomatopoeia') "
                "WHERE id = ANY(%s) AND NOT ('onomatopoeia' = ANY(tags))",
                (chunk,)
            )
            conn.commit()

    cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags)")
    final = cur.fetchone()[0]
    print(f"Final onomatopoeia: {final}")
    stats['onomatopoeia_added'] = len(new_onomatopoeia_ids)
    stats['onomatopoeia_final'] = final

    # Show samples of newly tagged
    if new_onomatopoeia_ids:
        cur.execute("""
            SELECT word, reading, COALESCE(meaning,'') FROM vocab_library
            WHERE id = ANY(%s) LIMIT 50
        """, (new_onomatopoeia_ids[:50],))
        print("\nNewly tagged onomatopoeia samples:")
        for w, r, m in cur.fetchall():
            print(f"  {w:20s} {r or '':15s} {(m or '')[:60]}")

    # ━━━ 3. Create 专八 tags ━━━
    print("\n" + "=" * 50)
    print("3. Creating 专八 tags from high-level vocabulary")
    print("=" * 50)

    # 专八 vocabulary: kanji words that are NOT already gaming/loanword/slang
    # Target: formal/literary/academic words
    # Strategy: tag words that appear in yoji_jukugo OR have high complexity kanji
    # AND aren't already in other major categories

    # 专八词汇 (tem8_vocab): academic/formal Japanese vocabulary
    # Select kanji compound words that are at 考研 level (kaoyan) and beyond
    cur.execute("""
        SELECT count(*) FROM vocab_library
        WHERE 'kaoyan' = ANY(tags) AND NOT ('tem8_vocab' = ANY(tags))
    """)
    tem8_from_kaoyan = cur.fetchone()[0]
    print(f"考研 words to tag as 专八: {tem8_from_kaoyan}")

    if tem8_from_kaoyan > 0:
        cur.execute("""
            UPDATE vocab_library SET tags = array_append(tags, 'tem8_vocab')
            WHERE 'kaoyan' = ANY(tags) AND NOT ('tem8_vocab' = ANY(tags))
        """)
        conn.commit()
        print(f"  Tagged {tem8_from_kaoyan} words as tem8_vocab")

    # 专八惯用语 (tem8_idiom): idiomatic expressions
    cur.execute("""
        SELECT count(*) FROM vocab_library
        WHERE (
            word ~ '[気心手足目耳口鼻腹頭顔身].*[がをにでと]'
            OR word ~ '.*の.*[気心手足目]'
            OR word ~ '[上下内外前後]'
        )
        AND word ~ '[一-鿿].*[぀-ゟ゠-ヿ]'  -- mixed kanji+kana
        AND NOT ('tem8_idiom' = ANY(tags))
        AND NOT ('loanword' = ANY(tags))
        AND array_length(tags, 1) IS NOT NULL
    """)
    idiom_candidates = cur.fetchone()[0]
    print(f"\nPotential 惯用语 candidates: {idiom_candidates}")

    # Be selective: tag only a reasonable subset
    cur.execute("""
        UPDATE vocab_library SET tags = array_append(tags, 'tem8_idiom')
        WHERE (
            word ~ '気[がをにでとの]'
            OR word ~ '手[がをにでと]'
            OR word ~ '身[がをにでと]'
            OR word ~ '頭[がをにでと]'
            OR word ~ '腹[がをにでと]'
            OR word ~ '虫[がをにでと]'
        )
        AND NOT ('tem8_idiom' = ANY(tags))
        AND NOT ('loanword' = ANY(tags))
        AND NOT ('slang' = ANY(tags))
    """)
    idiom_tagged = cur.rowcount
    conn.commit()
    print(f"Tagged {idiom_tagged} words as tem8_idiom")

    # ━━━ Final stats ━━━
    print("\n" + "=" * 50)
    print("Final Tag Summary")
    print("=" * 50)
    for tag in ['loanword', 'yoji_jukugo', 'kaoyan', 'onomatopoeia', 'slang',
                'jlpt_n1', 'jlpt_n2', 'jlpt_n3', 'jlpt_n4', 'jlpt_n5',
                'tem8_vocab', 'tem8_idiom']:
        cur.execute(f"SELECT count(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
        cnt = cur.fetchone()[0]
        print(f"  {tag:20s}: {cnt:>6}")

    conn.close()
    print(f"\nDone. Stats: {stats}")


if __name__ == "__main__":
    main()
