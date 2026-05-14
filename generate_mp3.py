"""为所有有读音但无 MP3 的词生成音频并上传 Supabase Storage"""
import os
import re
import logging
import tempfile
import asyncio

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import edge_tts
import psycopg
import pykakasi as _pykakasi
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip("`")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
VOCAB_AUDIO_BUCKET = os.getenv("VOCAB_AUDIO_BUCKET", "vocab-audio").strip()
CONCURRENCY = int(os.getenv("MP3_CONCURRENCY", "100"))
BATCH_SIZE = 200
DB_BATCH = 50  # 每 N 条写一次 DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mp3_gen")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
kakasi = _pykakasi.kakasi()

LEVEL_MAP = {"常用": "CY", "高频": "GP", "一般": "YB", "生僻": "SP", "罕用": "HY",
              "未分级": "WJ", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N4", "N5": "N5",
              "专八": "ZB"}


def make_mp3_name(level, reading, word_id):
    """纯 ASCII 文件名"""
    romaji = "".join([item["hepburn"] for item in kakasi.convert(reading or "")])
    romaji = re.sub(r"[^a-zA-Z0-9]", "", romaji)
    if not romaji:
        return None
    level_safe = LEVEL_MAP.get(level, re.sub(r"[^a-zA-Z0-9]", "", str(level or "XX")))
    return f"{level_safe}__{romaji}__{str(word_id)[:8]}.mp3"


def db_write(results):
    """批量写 DB（无 prepare，单连接）"""
    if not results:
        return 0
    conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")
    try:
        with conn.cursor() as cur:
            for wid, mp3_url in results:
                cur.execute("UPDATE vocab_library SET mp3 = %s WHERE id = %s", (mp3_url, wid))
        conn.commit()
        return len(results)
    except Exception as e:
        log.error(f"DB 写入失败: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        conn.close()


async def gen_one(sem, wid, word, reading, mp3_name):
    """TTS + 上传（不碰 DB），返回 (id, url) 或 None"""
    async with sem:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            communicate = edge_tts.Communicate(reading or word, "ja-JP-NanamiNeural")
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                supabase.storage.from_(VOCAB_AUDIO_BUCKET).upload(
                    path=mp3_name,
                    file=f.read(),
                    file_options={"content-type": "audio/mpeg", "upsert": "true"},
                )
            mp3_url = supabase.storage.from_(VOCAB_AUDIO_BUCKET).get_public_url(mp3_name)
            os.unlink(tmp_path)
            return (wid, mp3_url)
        except Exception as e:
            log.warning(f"  ✗ {word}: {e}")
            return None


async def main():
    conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, word, reading, level
            FROM vocab_library
            WHERE reading IS NOT NULL AND reading != ''
              AND (mp3 IS NULL OR mp3 NOT LIKE 'http%')
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
    conn.close()

    total = len(rows)
    log.info(f"待生成 MP3: {total} 词 (并发={CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]

        # Step 1: 并发生成 + 上传
        tasks = []
        for wid, word, reading, level in batch:
            mp3_name = make_mp3_name(level, reading, wid)
            if not mp3_name:
                continue
            tasks.append(gen_one(sem, wid, word, reading, mp3_name))
        results = await asyncio.gather(*tasks)

        # Step 2: 收集有效结果
        ok = [r for r in results if r is not None]

        # Step 3: 分批写 DB
        saved = 0
        for j in range(0, len(ok), DB_BATCH):
            sub = ok[j:j + DB_BATCH]
            saved += db_write(sub)

        done += saved
        log.info(f"  进度: {done}/{total}")

    log.info(f"完成: {done}/{total}")

    # 补漏：重试失败/遗漏的词，直到全部完成（最多10轮）
    retry_round = 0
    while True:
        retry_round += 1
        if retry_round > 10:
            log.info("补漏已达10轮，停止")
            break

        conn = get_conn()
        try:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute("""
                    SELECT id, word, reading, level
                    FROM vocab_library
                    WHERE reading IS NOT NULL AND reading != ''
                      AND (mp3 IS NULL OR mp3 NOT LIKE 'http%')
                    ORDER BY created_at DESC
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            log.info("全部 MP3 生成完毕！")
            break

        log.info(f"[补漏第{retry_round}轮] 剩余 {len(rows)} 词")
        prev_done = done

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            tasks = []
            for wid, word, reading, level in batch:
                mp3_name = make_mp3_name(level, reading, wid)
                if not mp3_name:
                    continue
                tasks.append(gen_one(sem, wid, word, reading, mp3_name))
            results = await asyncio.gather(*tasks)
            ok = [r for r in results if r is not None]
            saved = 0
            for j in range(0, len(ok), DB_BATCH):
                sub = ok[j:j + DB_BATCH]
                saved += db_write(sub)
            done += saved
            log.info(f"  补漏进度: {done}")

        if done == prev_done:
            log.info(f"补漏第{retry_round}轮无新增成功，停止")
            break

    log.info(f"最终: {done}")


if __name__ == "__main__":
    asyncio.run(main())
