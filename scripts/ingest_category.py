#!/usr/bin/env python3
"""批量录入分类词表：基础信息 + AI 富化 + MP3 生成

Usage:
  python3 scripts/ingest_category.py --category slang --limit 3 --skip-mp3
  python3 scripts/ingest_category.py --category gaming_lol --limit 5
  python3 scripts/ingest_category.py --category gaming_valorant --skip-enrich
  python3 scripts/ingest_category.py --category slang            # 全量
"""

import os
import sys
import re
import json
import asyncio
import argparse
import logging
import tempfile
import traceback

# 确保项目根目录在 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# 加载 .env
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import psycopg
import edge_tts
import pykakasi as _pykakasi
from supabase import create_client

from ai_service import AIService

# ---- Config ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip("`")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
VOCAB_AUDIO_BUCKET = os.getenv("VOCAB_AUDIO_BUCKET", "vocab-audio").strip()
CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "20"))  # 同时处理 N 个词，可通过环境变量覆盖

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_category")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
kakasi = _pykakasi.kakasi()
ai = AIService()

LEVEL_MAP = {"常用": "CY", "高频": "GP", "一般": "YB", "生僻": "SP", "罕用": "HY",
             "未分级": "WJ", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N4", "N5": "N5",
             "专八": "ZB"}


def load_words(category: str) -> list:
    """根据 category 加载对应的词表"""
    if category == "slang":
        from data.slang_words import WORDS
    elif category == "gaming_lol":
        from data.gaming_lol_terms import WORDS
    elif category == "gaming_valorant":
        from data.gaming_valorant_terms import WORDS
    else:
        raise ValueError(f"Unknown category: {category}")
    return WORDS


def make_mp3_name(word_id, reading: str) -> str:
    """纯 ASCII 文件名，用于 Supabase Storage"""
    romaji = "".join([item["hepburn"] for item in kakasi.convert(reading or "")])
    romaji = re.sub(r"[^a-zA-Z0-9]", "", romaji)
    if not romaji:
        romaji = re.sub(r"[^a-zA-Z0-9]", "", str(word_id))
    if not romaji:
        return None
    return f"cat__{romaji}__{str(word_id)[:8]}.mp3"


def get_db_connection():
    return psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c plan_cache_mode=force_custom_plan",
    )


# ---- DB Operations ----

def ensure_word_exists(conn, word: str, reading: str, meaning: str, tag: str):
    """检查词是否存在；不存在则 INSERT 基础信息；返回 (id, is_new)"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, tags FROM vocab_library WHERE word = %s AND reading = %s LIMIT 1",
            (word, reading),
        )
        row = cur.fetchone()

    if row:
        word_id = row[0]
        existing_tags = list(row[1]) if row[1] else []

        # 检查是否需要添加 tag
        if tag not in existing_tags:
            existing_tags.append(tag)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE vocab_library SET tags = %s WHERE id = %s",
                    (existing_tags, word_id),
                )
            conn.commit()
            return {"id": word_id, "is_new": False, "tag_added": True}

        return {"id": word_id, "is_new": False, "tag_added": False}

    # 新词：INSERT
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO vocab_library (word, reading, meaning, level, tags)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING id""",
            (word, reading, meaning, "未分级", [tag]),
        )
        word_id = cur.fetchone()[0]
    conn.commit()
    return {"id": word_id, "is_new": True, "tag_added": True}


def save_enrichment(conn, word_id, enriched: dict) -> bool:
    """将 AI 富化结果写入 DB"""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE vocab_library
                   SET social_context = %s,
                       heatmap_data = %s,
                       insight_text = %s,
                       frequency = %s,
                       examples = %s,
                       pos = %s,
                       pitch = %s,
                       is_ai_enriched = TRUE
                   WHERE id = %s""",
                (
                    json.dumps(enriched.get("social_context"), ensure_ascii=False),
                    json.dumps(enriched.get("heatmap_data"), ensure_ascii=False),
                    enriched.get("insight_text"),
                    enriched.get("frequency_stars"),
                    json.dumps(enriched.get("examples"), ensure_ascii=False),
                    enriched.get("pos_cn"),
                    enriched.get("pitch"),
                    word_id,
                ),
            )
        conn.commit()
        return True
    except Exception as e:
        log.error(f"  保存富化结果失败 (id={word_id}): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def set_mp3(conn, word_id, mp3_url: str) -> bool:
    """更新 DB 中的 mp3 字段"""
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE vocab_library SET mp3 = %s WHERE id = %s", (mp3_url, word_id))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"  保存 MP3 URL 失败 (id={word_id}): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


# ---- MP3 Generation ----

async def gen_mp3(sem, word_id, word: str, reading: str):
    """TTS + 上传 Supabase Storage，返回 public URL"""
    async with sem:
        try:
            mp3_name = make_mp3_name(word_id, reading)
            if not mp3_name:
                return None

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
            return mp3_url
        except Exception as e:
            log.warning(f"  MP3 生成失败 {word}: {e}")
            return None


# ---- Main Processing ----

async def process_one(sem, conn, word_data: dict, category: str, skip_enrich: bool, skip_mp3: bool, stats: dict):
    """处理单个词：录入 + AI 富化 + MP3"""
    word = word_data["word"]
    reading = word_data["reading"]
    meaning = word_data["meaning"]

    # 确定 tag 名
    tag_map = {
        "slang": "slang",
        "gaming_lol": "gaming_lol",
        "gaming_valorant": "gaming_valorant",
    }
    tag = tag_map.get(category, category)

    try:
        # Step 1: 确保词存在于 DB
        result = ensure_word_exists(conn, word, reading, meaning, tag)
        word_id = result["id"]
        is_new = result["is_new"]
        tag_added = result["tag_added"]

        if is_new:
            stats["new_words"] += 1
            log.info(f"  + 新词: {word} (id={word_id})")
        elif tag_added:
            stats["tagged"] += 1
            log.info(f"  ~ 标签: {word} +{tag}")
        else:
            stats["skipped"] += 1
            log.info(f"  = 已存在: {word}")

        # Step 2: AI 富化
        if not skip_enrich:
            log.info(f"  🤖 AI 分析: {word} (category={category})")
            enriched = await ai.enrich_library_entry(
                word=word, reading=reading, meaning=meaning, category=category
            )
            if enriched:
                if save_enrichment(conn, word_id, enriched):
                    stats["enriched"] += 1
                    log.info(f"  ✓ 富化完成: {word}")
                else:
                    stats["enrich_failed"] += 1
            else:
                stats["enrich_failed"] += 1
                log.warning(f"  ✗ AI 返回空: {word}")
        else:
            stats["enrich_skipped"] += 1

        # Step 3: MP3 生成
        if not skip_mp3:
            mp3_url = await gen_mp3(sem, word_id, word, reading)
            if mp3_url:
                if set_mp3(conn, word_id, mp3_url):
                    stats["mp3"] += 1
                    log.info(f"  🎵 MP3 上传: {word}")
                else:
                    stats["mp3_failed"] += 1
            else:
                stats["mp3_failed"] += 1
        else:
            stats["mp3_skipped"] += 1

    except Exception as e:
        log.error(f"  ✗ 处理失败 {word}: {e}")
        traceback.print_exc()
        stats["errors"] += 1


async def main():
    parser = argparse.ArgumentParser(description="批量录入分类词表")
    parser.add_argument("--category", required=True,
                        choices=["slang", "gaming_lol", "gaming_valorant"],
                        help="词表分类")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="跳过 AI 富化，只录入基础信息")
    parser.add_argument("--skip-mp3", action="store_true",
                        help="跳过 MP3 生成")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个词（0=全部）")
    args = parser.parse_args()

    # 加载词表
    words = load_words(args.category)
    if args.limit > 0:
        words = words[: args.limit]

    total = len(words)
    log.info(f"=== 开始批量录入 ===")
    log.info(f"分类: {args.category}")
    log.info(f"词数: {total}")
    log.info(f"AI 富化: {'跳过' if args.skip_enrich else '启用'}")
    log.info(f"MP3 生成: {'跳过' if args.skip_mp3 else '启用'}")
    log.info(f"并发: {CONCURRENCY}")
    log.info("")

    # 统计
    stats = {
        "new_words": 0,
        "tagged": 0,
        "skipped": 0,
        "enriched": 0,
        "enrich_failed": 0,
        "enrich_skipped": 0,
        "mp3": 0,
        "mp3_failed": 0,
        "mp3_skipped": 0,
        "errors": 0,
    }

    conn = get_db_connection()
    sem = asyncio.Semaphore(CONCURRENCY)

    try:
        tasks = [
            process_one(sem, conn, wd, args.category, args.skip_enrich, args.skip_mp3, stats)
            for wd in words
        ]
        await asyncio.gather(*tasks)
    finally:
        conn.close()

    # 输出统计
    log.info("")
    log.info("=== 处理完成 ===")
    log.info(f"总词数: {total}")
    log.info(f"  新录入: {stats['new_words']}")
    log.info(f"  新标签: {stats['tagged']}")
    log.info(f"  已存在(跳过): {stats['skipped']}")
    log.info(f"  AI 富化成功: {stats['enriched']}")
    log.info(f"  AI 富化失败: {stats['enrich_failed']}")
    log.info(f"  AI 富化跳过: {stats['enrich_skipped']}")
    log.info(f"  MP3 成功: {stats['mp3']}")
    log.info(f"  MP3 失败: {stats['mp3_failed']}")
    log.info(f"  MP3 跳过: {stats['mp3_skipped']}")
    log.info(f"  处理异常: {stats['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
