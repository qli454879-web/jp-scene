"""给所有 CSV 中缺失 JLPT 标签的词补标签"""
import csv, os, psycopg

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

csv_files = {
    'N3': "/Users/chenchanghou/Documents/trae2/词汇/n3_work/vocab_library_import_n3_pro_dedup.csv",
    'N4': "/Users/chenchanghou/Documents/trae2/词汇/n4_work/vocab_library_import_n4_pro_dedup.csv",
    'N5': "/Users/chenchanghou/Documents/trae2/词汇/n5_work/vocab_library_import_n5_pro_dedup.csv",
}

db_url = os.environ.get('SUPABASE_DB_URL', '').strip()
conn = psycopg.connect(db_url, prepare_threshold=None, connect_timeout=10)
cur = conn.cursor()

for level, csv_path in csv_files.items():
    tag = 'jlpt_' + level.lower()

    # Read CSV words
    csv_words = set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get('word') or '').strip()
            if w:
                csv_words.add(w)

    # DB words with this tag
    cur.execute(f"SELECT word FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    db_words = set(row[0] for row in cur.fetchall())

    missing = csv_words - db_words
    if not missing:
        print(f"{level}: 已完整，无需补充")
        continue

    print(f"{level}: CSV={len(csv_words)}, DB={len(db_words)}, 需补充={len(missing)}")

    # Check how many are actually in DB
    missing_list = list(missing)
    in_db = []
    not_in_db = []
    for w in missing_list:
        cur.execute("SELECT 1 FROM vocab_library WHERE word = %s", (w,))
        if cur.fetchone():
            in_db.append(w)
        else:
            not_in_db.append(w)

    print(f"  在DB中可补充: {len(in_db)}, 不在DB中: {len(not_in_db)}")
    if not_in_db:
        print(f"  不在DB采样: {not_in_db[:10]}")

    # Update in batches
    if in_db:
        batch_size = 200
        total = 0
        for i in range(0, len(in_db), batch_size):
            batch = in_db[i:i+batch_size]
            placeholders = ','.join(['%s'] * len(batch))
            cur.execute(f"""
                UPDATE vocab_library
                SET tags = array_append(tags, %s)
                WHERE word IN ({placeholders})
                AND NOT (%s = ANY(tags))
            """, [tag] + batch + [tag])
            total += cur.rowcount
        conn.commit()
        print(f"  Updated: {total}")

    # Verify
    cur.execute(f"SELECT COUNT(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    new_count = cur.fetchone()[0]
    print(f"  最终 {tag}: {new_count}")

conn.close()
print("\nDone.")
