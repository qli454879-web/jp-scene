#!/usr/bin/env python3
"""Analyze the DB to find onomatopoeia, JLPT tags, and potential 专八 vocabulary."""
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
conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None, options="-c statement_timeout=30000")

cur = conn.cursor()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Find potential onomatopoeia (kana-only repeating patterns)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("1. Potential Onomatopoeia (katakana words with repeating patterns)")
print("=" * 60)

# Onomatopoeia patterns: ABAB, AッBリ, ABッ, etc.
cur.execute("""
    SELECT word, reading, meaning, tags FROM vocab_library
    WHERE word ~ '^[ァ-ヴー]{2,8}$'
      AND (
        word ~ '(..)\\1'                    -- ABAB pattern
        OR word ~ 'ッ'                      -- contains small tsu
        OR word ~ 'ン$'                     -- ends with ン
        OR word ~ 'リ$'                     -- ends with リ
      )
      AND NOT ('onomatopoeia' = ANY(tags))  -- not already tagged
      AND word !~ '^[ア-ヴ]ー$'             -- not just a single char + ー
    ORDER BY word
    LIMIT 100
""")
rows = cur.fetchall()
print(f"Found {len(rows)} potential onomatopoeia (first 100 shown)")

# Count matching patterns
cur.execute("""
    SELECT count(*) FROM vocab_library
    WHERE word ~ '^[ァ-ヴー]{2,8}$'
      AND (
        word ~ '(..)\\1'
        OR word ~ 'ッ'
        OR word ~ 'ン$'
        OR word ~ 'リ$'
      )
      AND NOT ('onomatopoeia' = ANY(tags))
      AND word !~ '^[ア-ヴ]ー$'
""")
total_potential = cur.fetchone()[0]
print(f"Total potential onomatopoeia to tag: {total_potential}")

# Show some samples
print("\nSample potential onomatopoeia:")
for i, (w, r, m, t) in enumerate(rows[:20]):
    meaning_short = (m or '')[:50] if m else ''
    print(f"  {w:20s} {r or '':15s} {meaning_short}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. JLPT tag analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("2. JLPT Tag Analysis")
print("=" * 60)

jlpt_tags = ['jlpt_n1', 'jlpt_n2', 'jlpt_n3', 'jlpt_n4', 'jlpt_n5',
             'n1', 'n2', 'n3', 'n4', 'n5']
for tag in jlpt_tags:
    cur.execute(f"SELECT count(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    cnt = cur.fetchone()[0]
    print(f"  {tag:15s}: {cnt:>6}")

# Words with multiple JLPT tags
cur.execute("""
    SELECT word, tags FROM vocab_library
    WHERE tags && ARRAY['n1','n2','n3','n4','n5','jlpt_n1','jlpt_n2','jlpt_n3','jlpt_n4','jlpt_n5']
    LIMIT 10
""")
print("\nSample JLPT-tagged words:")
for w, t in cur.fetchall():
    print(f"  {w}: {t}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Find 专八-relevant vocabulary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("3. 专八 Vocabulary (high-level/complex words)")
print("=" * 60)

# 专八 words tend to be: 漢語, literary, advanced
# Check for words with certain characteristics
# First, look for words that have been tagged with anything related
cur.execute("SELECT count(*) FROM vocab_library WHERE tags && ARRAY['kaoyan']")
kaoyan_total = cur.fetchone()[0]
print(f"Current 考研 words: {kaoyan_total}")

# Find potential 专八 words: formal/literary vocabulary
# These are typically 漢語 (kango) - 2-4 kanji compounds
cur.execute("""
    SELECT count(*) FROM vocab_library
    WHERE word ~ '^[一-鿿]{2,6}$'
      AND 'loanword' != ALL(tags)
      AND 'onomatopoeia' != ALL(tags)
      AND 'slang' != ALL(tags)
      AND 'gaming_valorant' != ALL(tags)
      AND 'gaming_lol' != ALL(tags)
""")
kanji_words = cur.fetchone()[0]
print(f"Pure kanji words (2-6 chars): {kanji_words}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Check for vocabulary that could be 惯用语/谚语
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("4. Potential 惯用语/谚语/四字熟语")
print("=" * 60)

cur.execute("SELECT count(*) FROM vocab_library WHERE 'yoji_jukugo' = ANY(tags)")
print(f"Current 四字熟语: {cur.fetchone()[0]}")

# Find 4-kanji compounds not yet tagged as yoji_jukugo
cur.execute("""
    SELECT count(*) FROM vocab_library
    WHERE word ~ '^[一-鿿]{4}$'
      AND NOT ('yoji_jukugo' = ANY(tags))
""")
print(f"4-kanji words not tagged yoji_jukugo: {cur.fetchone()[0]}")

# Find potential 惯用语 (idiomatic expressions) - often contain 気、心、手、目 etc.
cur.execute("""
    SELECT word, meaning FROM vocab_library
    WHERE word ~ '[気心手足目耳口腹頭].*[がをにでと]'
       OR word ~ '.*の.*[気心手足目]'
    LIMIT 10
""")
print("\nSample potential 惯用语:")
for w, m in cur.fetchall():
    print(f"  {w}: {(m or '')[:60]}")

conn.close()
print("\nDone.")
