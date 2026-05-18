"""对照 CSV N1 单词 vs DB jlpt_n1 标签"""
import csv, os, psycopg

# .env 在项目根目录
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

csv_path = "/Users/chenchanghou/Documents/trae2/词汇/n1_work/n1_standardized_final_副本.csv"

# 1. CSV words
csv_words = set()
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        w = (row.get('word') or '').strip()
        if w:
            csv_words.add(w)
print(f"CSV 单词数: {len(csv_words)}")

# 2. DB words
db_url = os.environ.get('SUPABASE_DB_URL', '').strip()
conn = psycopg.connect(db_url, prepare_threshold=None, connect_timeout=10)
cur = conn.cursor()

cur.execute("SELECT word FROM vocab_library WHERE 'jlpt_n1' = ANY(tags)")
db_words = set(row[0] for row in cur.fetchall())
print(f"DB jlpt_n1: {len(db_words)}")

cur.execute("SELECT word FROM vocab_library WHERE level = 'N1'")
level_words = set(row[0] for row in cur.fetchall())
print(f"DB level=N1: {len(level_words)}")

# 3. 对照
in_csv_not_db = csv_words - db_words
in_db_not_csv = db_words - csv_words
# 转换大小写后再比对（CSV可能有个别写法差异）
csv_lower = {w.lower() for w in csv_words}
db_lower = {w.lower() for w in db_words}
in_csv_not_db_lower = csv_lower - db_lower
in_db_not_csv_lower = db_lower - csv_lower

print(f"\nCSV有但DB(jlpt_n1)无: {len(in_csv_not_db)}")
print(f"  (忽略大小写后): {len(in_csv_not_db_lower)}")
print(f"DB(jlpt_n1)有但CSV无: {len(in_db_not_csv)}")
print(f"  (忽略大小写后): {len(in_db_not_csv_lower)}")

# 4. 分析CSV中缺失的词
if in_csv_not_db:
    print(f"\n=== CSV有但DB无 jlpt_n1 标签的采样(60) ===")
    missing_list = sorted(in_csv_not_db)
    count_found_other_tag = 0
    count_not_found = 0
    count_has_level_n1 = 0
    for w in missing_list[:60]:
        cur.execute("SELECT word, level, tags FROM vocab_library WHERE word = %s", (w,))
        row = cur.fetchone()
        if row:
            if row[1] == 'N1':
                count_has_level_n1 += 1
                print(f"  {w}: level={row[1]!r} tags={row[2]} ★ level=N1 但缺 jlpt_n1 标签!")
            else:
                count_found_other_tag += 1
                print(f"  {w}: level={row[1]!r} tags={row[2]}")
        else:
            count_not_found += 1
            print(f"  {w}: NOT IN DB")
    if len(missing_list) > 60:
        print(f"  ... 还有 {len(missing_list)-60} 个")

    # 统计全文
    print(f"\n完整统计({len(missing_list)}个缺失词):")
    found_level_n1 = 0
    found_other = 0
    not_in_db = 0
    for w in missing_list:
        cur.execute("SELECT level FROM vocab_library WHERE word = %s", (w,))
        row = cur.fetchone()
        if row:
            if row[0] == 'N1':
                found_level_n1 += 1
            else:
                found_other += 1
        else:
            not_in_db += 1
    print(f"  level=N1 但缺少 jlpt_n1 标签: {found_level_n1}")
    print(f"  在DB但level不是N1: {found_other}")
    print(f"  完全不在DB中: {not_in_db}")

# 5. 反向：DB有但CSV无
if in_db_not_csv:
    print(f"\n=== DB有 jlpt_n1 但CSV无的采样(30) ===")
    for w in sorted(in_db_not_csv)[:30]:
        cur.execute("SELECT word, level FROM vocab_library WHERE word = %s", (w,))
        row = cur.fetchone()
        lv = row[1] if row else '?'
        print(f"  {w}: level={lv!r}")

conn.close()
