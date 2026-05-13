"""批量生成单词朗读 MP3 并上传到 Supabase Storage"""
import asyncio
import os
import sys
import logging
import tempfile

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
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip("`")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
VOCAB_AUDIO_BUCKET = os.getenv("VOCAB_AUDIO_BUCKET", "vocab-audio").strip()

BATCH_SIZE = 50  # 每批处理词数
VOICE = "ja-JP-NanamiNeural"  # 最自然的日语女声

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mp3_upload")


async def gen_save_mp3(word: str, reading: str, voice: str, target_path: str) -> bool:
    """生成单词朗读 MP3"""
    # 只用单词本身，带上读音保证准确
    text = f"{reading}。{word}。"
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(target_path)
        return True
    except Exception as e:
        log.error(f"  TTS 生成失败 {word}: {e}")
        return False


async def main():
    if not all([SUPABASE_URL, SUPABASE_DB_URL]):
        log.error("SUPABASE_URL / SUPABASE_DB_URL 未配置")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    conn = psycopg.connect(SUPABASE_DB_URL)

    # 取有 mp3 文件名但需要上传的词（mp3 列有值）
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, word, reading, mp3
            FROM vocab_library
            WHERE mp3 IS NOT NULL AND mp3 != ''
              AND is_ai_enriched = TRUE
            ORDER BY created_at DESC
            LIMIT %s
        """, (BATCH_SIZE,))
        rows = cur.fetchall()

    if not rows:
        log.info("没有需要上传 MP3 的词")
        conn.close()
        return

    log.info(f"待上传 {len(rows)} 个 MP3 → bucket: {VOCAB_AUDIO_BUCKET}")

    success_count = 0
    for i, (wid, word, reading, mp3_name) in enumerate(rows):
        # 先检查 bucket 里是否已存在
        try:
            existing = supabase.storage.from_(VOCAB_AUDIO_BUCKET).list(mp3_name)
            if existing and len(existing) > 0:
                log.info(f"  [{i+1}/{len(rows)}] 跳过（已存在） {word} → {mp3_name}")
                success_count += 1
                continue
        except Exception:
            pass  # 文件不存在，继续生成

        # 生成 MP3 到临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        ok = await gen_save_mp3(word, reading or word, VOICE, tmp_path)
        if not ok:
            os.unlink(tmp_path)
            continue

        # 上传到 Supabase Storage
        try:
            with open(tmp_path, "rb") as f:
                supabase.storage.from_(VOCAB_AUDIO_BUCKET).upload(
                    path=mp3_name,
                    file=f,
                    file_options={"content-type": "audio/mpeg", "upsert": "true"},
                )
            os.unlink(tmp_path)
            success_count += 1
            log.info(f"  [{i+1}/{len(rows)}] ✓ {word} → {mp3_name}")
        except Exception as e:
            os.unlink(tmp_path)
            log.error(f"  [{i+1}/{len(rows)}] ✗ 上传失败 {word}: {e}")

        if i < len(rows) - 1:
            await asyncio.sleep(0.5)  # 避免请求太密集

    conn.close()
    log.info(f"完成。成功上传 {success_count}/{len(rows)} 个 MP3")


if __name__ == "__main__":
    asyncio.run(main())
