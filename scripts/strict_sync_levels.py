"""严格按照 CSV 单词表同步 JLPT 标签：CSV有的补上，CSV没有的移除"""
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
    'N1': "/Users/chenchanghou/Documents/trae2/词汇/n1_work/n1_standardized_final_副本.csv",
    'N2': "/Users/chenchanghou/Documents/trae2/词汇/n2_work/n2_standardized_final.csv",
    'N3': "/Users/chenchanghou/Documents/trae2/词汇/n3_work/vocab_library_import_n3_pro_dedup.csv",
    'N4': "/Users/chenchanghou/Documents/trae2/词汇/n4_work/vocab_library_import_n4_pro_dedup.csv",
    'N5': "/Users/chenchanghou/Documents/trae2/词汇/n5_work/vocab_library_import_n5_pro_dedup.csv",
}

db_url = os.environ.get('SUPABASE_DB_URL', '').strip()
conn = psycopg.connect(db_url, prepare_threshold=None, connect_timeout=10)
cur = conn.cursor()

for level, csv_path in csv_files.items():
    tag = 'jlpt_' + level.lower()
    print(f"\n{'='*60}")
    print(f"{level} ({tag})")

    # Read CSV words
    csv_words = set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get('word') or '').strip()
            if w:
                csv_words.add(w)
    print(f"  CSV: {len(csv_words)}")

    # DB words with this tag
    cur.execute(f"SELECT word FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    db_tagged = set(row[0] for row in cur.fetchall())
    print(f"  DB 当前标签: {len(db_tagged)}")

    # 1. ADD: CSV有但DB无
    to_add = csv_words - db_tagged
    if to_add:
        add_list = list(to_add)
        # 只更新在DB中存在的词
        in_db = []
        not_found = []
        for w in add_list:
            cur.execute("SELECT 1 FROM vocab_library WHERE word = %s", (w,))
            if cur.fetchone():
                in_db.append(w)
            else:
                not_found.append(w)

        if in_db:
            batch_size = 200
            added = 0
            for i in range(0, len(in_db), batch_size):
                batch = in_db[i:i+batch_size]
                ph = ','.join(['%s'] * len(batch))
                cur.execute(f"""
                    UPDATE vocab_library SET tags = array_append(tags, %s)
                    WHERE word IN ({ph}) AND NOT (%s = ANY(tags))
                """, [tag] + batch + [tag])
                added += cur.rowcount
            conn.commit()
            print(f"  ADD: {added}")
        if not_found:
            print(f"  不在DB中: {len(not_found)}")

    # 2. REMOVE: DB有但CSV无
    to_remove = db_tagged - csv_words
    if to_remove:
        remove_list = list(to_remove)
        batch_size = 200
        removed = 0
        for i in range(0, len(remove_list), batch_size):
            batch = remove_list[i:i+batch_size]
            ph = ','.join(['%s'] * len(batch))
            cur.execute(f"""
                UPDATE vocab_library SET tags = array_remove(tags, %s)
                WHERE word IN ({ph}) AND (%s = ANY(tags))
            """, [tag] + batch + [tag])
            removed += cur.rowcount
        conn.commit()
        print(f"  REMOVE: {removed}")
        # 采样显示几个被移除的
        if removed > 0:
            print(f"    采样: {sorted(remove_list)[:8]}")

    # Verify
    cur.execute(f"SELECT COUNT(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    final = cur.fetchone()[0]

    # Check alignment
    cur.execute(f"SELECT word FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    final_set = set(row[0] for row in cur.fetchall())
    still_missing = csv_words - final_set
    still_extra = final_set - csv_words

    print(f"  FINAL: {final}  (CSV={len(csv_words)}  缺={len(still_missing)}  多={len(still_extra)})")
    if still_extra:
        print(f"    多出采样: {sorted(still_extra)[:8]}")

conn.close()
print("\nDone.")
