"""移除同时有 loanword 标签的词的 onomatopoeia 标签"""
import os, psycopg

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

db_url = os.environ.get('SUPABASE_DB_URL', '').strip()
conn = psycopg.connect(db_url, prepare_threshold=None, connect_timeout=10)
cur = conn.cursor()

# 1. 现状
cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags)")
total_before = cur.fetchone()[0]

cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags) AND 'loanword' = ANY(tags)")
overlap = cur.fetchone()[0]

cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags) AND NOT ('loanword' = ANY(tags))")
pure_ono = cur.fetchone()[0]

print(f"清理前: onomatopoeia 总数={total_before}, 与loanword重叠={overlap}, 纯拟声={pure_ono}")

# 2. 查看纯拟声词（保留）
if pure_ono > 0:
    cur.execute("SELECT word, reading FROM vocab_library WHERE 'onomatopoeia' = ANY(tags) AND NOT ('loanword' = ANY(tags)) ORDER BY word")
    print("保留的纯拟声词:")
    for row in cur.fetchall():
        print(f"  {row[0]} ({row[1]})")

# 3. 移除 onomatopoeia（从同时有 loanword 的词）
if overlap > 0:
    cur.execute("""
        UPDATE vocab_library
        SET tags = array_remove(tags, 'onomatopoeia')
        WHERE 'onomatopoeia' = ANY(tags)
          AND 'loanword' = ANY(tags)
    """)
    conn.commit()
    print(f"  REMOVED onomatopoeia from {cur.rowcount} words")

# 4. 验证
cur.execute("SELECT count(*) FROM vocab_library WHERE 'onomatopoeia' = ANY(tags)")
total_after = cur.fetchone()[0]
print(f"清理后: onomatopoeia 总数={total_after}")

# 展示剩余
cur.execute("SELECT word, reading, tags FROM vocab_library WHERE 'onomatopoeia' = ANY(tags) ORDER BY word")
remaining = cur.fetchall()
print("剩余拟声词:")
for row in remaining:
    print(f"  {row[0]} ({row[1]}) tags={row[2]}")

conn.close()
print("\nDone.")
