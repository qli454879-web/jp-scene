"""从 JMdict 导入常用词 + 外来语 + 学术术语（跳过库中已存在）"""
import os
import sys
import logging
import xml.etree.ElementTree as ET

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import psycopg

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("jmdict_import")

KATA_CHARS = set(range(0x30A0, 0x30FF + 1)) | {0x30FC}
def is_katakana(s):
    return bool(s) and all(ord(c) in KATA_CHARS for c in s)

PRIORITY_TAGS = {"ichi1", "ichi2", "spec1", "spec2", "news1", "news2", "gai1", "gai2"}
NF_TAGS = {f"nf{n:02d}" for n in range(1, 49)}

# JMdict 学术/专业领域标签 —— 只要带这些标签就收录
ACADEMIC_FIELDS = {
    "comp",     # computer science
    "med",      # medicine
    "law",      # law
    "ling",     # linguistics
    "biol",     # biology
    "chem",     # chemistry
    "math",     # mathematics
    "phys",     # physics
    "geol",     # geology
    "astron",   # astronomy
    "engr",     # engineering
    "econ",     # economics
    "finc",     # finance
    "bus",      # business
    "biochem",  # biochemistry
    "genet",    # genetics
    "psych",    # psychology
    "phil",     # philosophy
    "shinto",   # shinto
    "Buddh",    # buddhism
    "MA",       # martial arts
    "music",    # music
    "food",     # food / cooking
    "sports",   # sports (catch-all)
    "baseb",    # baseball
    "sumo",     # sumo
    "judo",     # judo
    "elec",     # electricity / electronics
    "telec",    # telecommunications
    "print",    # printing
    "photo",    # photography
    "aviat",    # aviation
    "naut",     # nautical
    "mil",      # military
    "archit",   # architecture
    "art",      # art / aesthetics
    "cloth",    # clothing
    "zool",     # zoology
    "bot",      # botany
    "met",      # meteorology
    "ocean",    # oceanography
    "geogr",    # geography
}

def has_priority_tag(k_eles, r_eles):
    tags = set()
    for ele in k_eles:
        for pri in ele.findall("ke_pri"):
            tags.add(pri.text)
    for ele in r_eles:
        for pri in ele.findall("re_pri"):
            tags.add(pri.text)
    return bool(tags & (PRIORITY_TAGS | NF_TAGS))

def has_academic_field(senses):
    for s in senses:
        for f in s.findall("field"):
            if f.text and f.text.strip() in ACADEMIC_FIELDS:
                return True
    return False

def level_from_tags(k_eles, r_eles):
    tags = set()
    for ele in k_eles:
        for pri in ele.findall("ke_pri"):
            tags.add(pri.text)
    for ele in r_eles:
        for pri in ele.findall("re_pri"):
            tags.add(pri.text)
    if "ichi1" in tags:
        return "N5"
    if "ichi2" in tags:
        return "N4"
    if "news1" in tags or "spec1" in tags:
        return "N3"
    if "gai1" in tags:
        return "N2"
    return "N2"

def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, autocommit=True, prepare_threshold=None)

def parse_jmdict(filepath):
    """解析 JMdict XML"""
    context = ET.iterparse(filepath, events=("end",))
    for event, elem in context:
        if elem.tag == "entry":
            k_eles = elem.findall("k_ele")
            r_eles = elem.findall("r_ele")
            senses = elem.findall("sense")

            # 先取 word
            word = ""
            for k in k_eles:
                keb = k.findtext("keb", "")
                if keb:
                    word = keb
                    break
            if not word:
                for r in r_eles:
                    reb = r.findtext("reb", "")
                    if reb:
                        word = reb
                        break

            # 收录条件：有常用标签 / 片假名外来语 / 学术专业词
            include = (
                has_priority_tag(k_eles, r_eles)
                or is_katakana(word)
                or has_academic_field(senses)
            )
            if not include:
                elem.clear()
                continue

            reading = ""
            for r in r_eles:
                reb = r.findtext("reb", "")
                if reb:
                    reading = reb
                    break

            meaning_parts = []
            pos_parts = []
            for s in senses[:3]:
                for gloss in s.findall("gloss"):
                    g = (gloss.text or "").strip()
                    if g:
                        meaning_parts.append(g)
                for p in s.findall("pos"):
                    pt = (p.text or "").strip()
                    if pt and pt not in pos_parts:
                        pos_parts.append(pt)

            meaning = " / ".join(meaning_parts[:5]) if meaning_parts else ""
            pos = pos_parts[0] if pos_parts else None
            level = level_from_tags(k_eles, r_eles)

            yield {
                "word": word,
                "reading": reading,
                "meaning": meaning,
                "pos": pos,
                "level": level,
                "source": "jmdict",
            }

            elem.clear()


def main():
    if not SUPABASE_DB_URL:
        log.error("SUPABASE_DB_URL not configured")
        return

    jmdict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JMdict_e")
    if not os.path.exists(jmdict_path):
        log.error(f"JMdict file not found: {jmdict_path}")
        return

    # 获取库中已有词
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT word FROM vocab_library")
            existing = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    log.info(f"库中已有 {len(existing)} 词条")

    # 解析 JMdict
    new_words = []
    total_parsed = 0
    kata_count = 0
    acad_count = 0
    for entry in parse_jmdict(jmdict_path):
        total_parsed += 1
        if entry["word"] in existing or not entry["meaning"]:
            continue
        new_words.append(entry)
        if is_katakana(entry["word"]):
            kata_count += 1
        if total_parsed % 50000 == 0:
            log.info(f"  已解析 {total_parsed} 条，新词 {len(new_words)}")

    log.info(f"解析完成。总常用词 {total_parsed}，待导入 {len(new_words)}")

    # 估算片假名数量
    kata_in_new = sum(1 for w in new_words if is_katakana(w["word"]))
    log.info(f"  其中片假名: {kata_in_new}, 非片假名(学术等): {len(new_words)-kata_in_new}")

    # 批量插入 —— 每批独立连接
    batch_size = 500
    inserted = 0
    skipped = 0
    for i in range(0, len(new_words), batch_size):
        batch = new_words[i : i + batch_size]
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                for w in batch:
                    try:
                        cur.execute(
                            """
                            INSERT INTO vocab_library (word, reading, meaning, level, pos, source, is_ai_enriched)
                            VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                            ON CONFLICT DO NOTHING
                            """,
                            (w["word"], w["reading"], w["meaning"], w["level"], w["pos"], w["source"]),
                        )
                        inserted += 1
                    except Exception as e:
                        skipped += 1
        finally:
            conn.close()
        if (i // batch_size + 1) % 10 == 0:
            log.info(f"  已插入 {inserted}/{len(new_words)}")

    log.info(f"完成。导入 {inserted} 条，跳过 {skipped} 条。")


if __name__ == "__main__":
    main()
