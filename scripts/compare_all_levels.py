"""对照所有 CSV 单词 vs DB JLPT 标签"""
import csv, os, psycopg

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# CSV文件映射 (level -> path)
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

    # Read CSV
    csv_words = set()
    if not os.path.exists(csv_path):
        print(f"{level}: CSV NOT FOUND: {csv_path}")
        continue
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get('word') or '').strip()
            if w:
                csv_words.add(w)

    # DB count
    cur.execute(f"SELECT COUNT(*) FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    db_count = cur.fetchone()[0]

    cur.execute(f"SELECT word FROM vocab_library WHERE %s = ANY(tags)", (tag,))
    db_words = set(row[0] for row in cur.fetchall())

    # Compare
    missing = csv_words - db_words
    extra = db_words - csv_words

    print(f"{level}: CSV={len(csv_words)}  DB(tag)={db_count}  CSV缺={len(missing)}  DB多={len(extra)}")

    if len(missing) > 0 and len(missing) <= 10:
        print(f"  缺: {sorted(missing)}")

conn.close()
