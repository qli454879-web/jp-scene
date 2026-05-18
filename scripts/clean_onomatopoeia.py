#!/usr/bin/env python3
"""清理 onomatopoeia 错标 — 绝大多数字汉字词不可能是拟声词

策略：
1. 含汉字的词 → 直接移除 onomatopoeia（汉字词绝不可能是拟声词）
2. 纯假名词 → 匹配拟声词特征（重复音节、っ/ん/り结尾等），匹配的保留，否则移除
3. 移除后 0 标签 → 根据特征补充 loanword / yoji_jukugo / slang

分批执行（2000 条/批），速度快，不影响线上。
"""
import os, sys, re, time

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

# 汉字正则
KANJI_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
# 纯假名
KANA_ONLY_RE = re.compile(r'^[ぁ-んァ-ンー]+$')
# 拟声拟态词特征模式
ONOMATOPOEIA_PATTERNS = [
    r'(.)\1',           # 重复音节: わくわく
    r'っ',              # 促音
    r'ん$',             # ん结尾
    r'り$',             # り结尾（拟态词特征）
    r'(.{2})\1',        # 双音节重复: ふわふわ
]

def has_kanji(word: str) -> bool:
    return bool(KANJI_RE.search(word))

def is_kana_only(word: str) -> bool:
    return bool(KANA_ONLY_RE.match(word))

def matches_onomatopoeia_pattern(word: str) -> bool:
    """检测假名词是否匹配拟声拟态词特征"""
    for pat in ONOMATOPOEIA_PATTERNS:
        if re.search(pat, word):
            return True
    return False

def main():
    conn = psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c statement_timeout=30000",
    )

    # 1. 统计当前 onomatopoeia 标签分布
    print("=== 当前 onomatopoeia 标签统计 ===")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags)")
        total = cur.fetchone()[0]
        print(f"总 onomatopoeia 词数: {total}")

        cur.execute("""
            SELECT count(*) FROM vocab_library
            WHERE 'onomatopoeia' = ANY(tags) AND word ~ '[一-鿿㐀-䶿]'
        """)
        kanji_cnt = cur.fetchone()[0]
        print(f"  含汉字（应移除）: {kanji_cnt} ({100*kanji_cnt/max(total,1):.1f}%)")

        cur.execute("""
            SELECT count(*) FROM vocab_library
            WHERE 'onomatopoeia' = ANY(tags) AND word !~ '[一-鿿㐀-䶿]'
        """)
        non_kanji = cur.fetchone()[0]
        print(f"  纯假名（需进一步判断）: {non_kanji}")

    # 2. 展示含汉字但标了 onomatopoeia 的样例
    print("\n=== 含汉字的 onomatopoeia 样例 ===")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT word FROM vocab_library
            WHERE 'onomatopoeia' = ANY(tags) AND word ~ '[一-鿿㐀-䶿]'
            LIMIT 10
        """)
        for (w,) in cur.fetchall():
            print(f"  {w}")

    # 3. 展示纯假名 onomatopoeia 样例
    print("\n=== 纯假名 onomatopoeia 样例 ===")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT word FROM vocab_library
            WHERE 'onomatopoeia' = ANY(tags) AND word !~ '[一-鿿㐀-䶿]'
            LIMIT 15
        """)
        for (w,) in cur.fetchall():
            match = "✓" if matches_onomatopoeia_pattern(w) else "✗"
            print(f"  {match} {w}")

    conn.close()
    print("\nDone. 确认无误后运行 --execute 执行清理。")


if __name__ == "__main__":
    main()
