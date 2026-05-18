"""给 CSV 中缺失 jlpt_n1 标签的词补上标签"""
import csv, os, psycopg

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

# 2. DB words with jlpt_n1
db_url = os.environ.get('SUPABASE_DB_URL', '').strip()
conn = psycopg.connect(db_url, prepare_threshold=None, connect_timeout=10)
cur = conn.cursor()

cur.execute("SELECT word FROM vocab_library WHERE 'jlpt_n1' = ANY(tags)")
db_jlpt_n1 = set(row[0] for row in cur.fetchall())

# 3. Missing
missing = csv_words - db_jlpt_n1
print(f"CSV: {len(csv_words)}, DB jlpt_n1: {len(db_jlpt_n1)}, 需补充: {len(missing)}")

# 4. Update in batches
if missing:
    missing_list = list(missing)
    batch_size = 200
    total_updated = 0
    for i in range(0, len(missing_list), batch_size):
        batch = missing_list[i:i+batch_size]
        placeholders = ','.join(['%s'] * len(batch))
        cur.execute(f"""
            UPDATE vocab_library
            SET tags = array_append(tags, 'jlpt_n1')
            WHERE word IN ({placeholders})
            AND NOT ('jlpt_n1' = ANY(tags))
        """, batch)
        updated = cur.rowcount
        total_updated += updated
        if updated != len(batch):
            # Some words not found — these are in CSV but not in DB
            print(f"  batch {i//batch_size}: expected {len(batch)}, updated {updated}")
    conn.commit()
    print(f"Total updated: {total_updated}")

# 5. Verify
cur.execute("SELECT COUNT(*) FROM vocab_library WHERE 'jlpt_n1' = ANY(tags)")
final_count = cur.fetchone()[0]
print(f"Final jlpt_n1 count: {final_count}")

# Also count how many CSV words now match
cur.execute("SELECT word FROM vocab_library WHERE 'jlpt_n1' = ANY(tags)")
new_db = set(row[0] for row in cur.fetchall())
still_missing = csv_words - new_db
print(f"Still missing from jlpt_n1: {len(still_missing)}")
if still_missing:
    print("Samples:", list(still_missing)[:10])

conn.close()
