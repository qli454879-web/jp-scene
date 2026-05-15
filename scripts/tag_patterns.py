#!/usr/bin/env python3
"""对现有词库按字符模式自动打标签：四字熟语 / 片假名外来语 / 拟声拟态词
Python 分类 + ID 批量更新（绕过 SQL regex 性能问题）

Usage:
  python3 scripts/tag_patterns.py          # 全量扫描并打标签
  python3 scripts/tag_patterns.py --dry-run  # 只看统计不打标签
"""

import os
import sys
import re
import argparse
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

ENV_FILE = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import psycopg

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tag_patterns")

# Python 正则
YOJIJUKUGO_RE = re.compile(r"^[一-鿿㐀-䶿]{4}$")
KATAKANA_RE = re.compile(r"^[゠-ヿー・\s]+$")
ONOMATO_RE = re.compile(
    r"([぀-ゟ゠-ヿ]{2})\1"  # ABAB
    r"|[぀-ゟ゠-ヿ]{2,4}[っん]$"  # ぎゅっ
    r"|[぀-ゟ゠-ヿ]{2}ー[぀-ゟ゠-ヿ]{2}$"  # にぱー
)


def classify(word, reading):
    tags = []
    w = (word or "").strip()
    r = (reading or "").strip()

    if YOJIJUKUGO_RE.match(w) and len(w) == 4:
        tags.append("yoji_jukugo")
    if KATAKANA_RE.match(w) and len(w) >= 2:
        tags.append("loanword")
    if ONOMATO_RE.search(w) or ONOMATO_RE.search(r):
        tags.append("onomatopoeia")
    return tags


def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计不打标签")
    args = parser.parse_args()

    # Step 1: 读取所有词并分类
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, word, reading, tags FROM vocab_library WHERE word IS NOT NULL")
            rows = cur.fetchall()
    finally:
        conn.close()

    total = len(rows)
    log.info(f"总词数: {total}")

    # 分类收集
    tag_ids = {"yoji_jukugo": [], "loanword": [], "onomatopoeia": []}
    existing_counts = {"yoji_jukugo": 0, "loanword": 0, "onomatopoeia": 0}

    for row_id, word, reading, existing_tags in rows:
        existing = list(existing_tags) if existing_tags else []
        new_tags = classify(word, reading)
        for tag in new_tags:
            if tag in existing:
                existing_counts[tag] += 1
            else:
                tag_ids[tag].append(row_id)

    name_map = {"yoji_jukugo": "四字熟語", "loanword": "片假名外来語", "onomatopoeia": "擬声擬態語"}

    for tag, ids in tag_ids.items():
        log.info(f"  {name_map[tag]}: {len(ids)} 个（已有: {existing_counts[tag]}）")

    total_new = sum(len(ids) for ids in tag_ids.values())

    if args.dry_run:
        log.info(f"\n总计 {total_new} 新标签（dry-run，未更新）")
        return

    if total_new == 0:
        log.info("无需更新。")
        return

    # Step 2: 分批更新（autocommit，每批独立提交，避免长事务锁表）
    log.info(f"\n开始更新 {total_new} 个词...")
    BATCH = 100

    conn = get_conn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '30s'")
        for tag, ids in tag_ids.items():
            if not ids:
                continue
            name = name_map[tag]
            updated = 0
            log.info(f"\n[{name}] {len(ids)} 个")
            for i in range(0, len(ids), BATCH):
                batch = ids[i:i + BATCH]
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE vocab_library
                           SET tags = array_append(COALESCE(tags, '{}'), %s)
                           WHERE id = ANY(%s)""",
                        (tag, batch),
                    )
                    n = cur.rowcount
                updated += n
                if i % 5000 == 0 and i > 0:
                    log.info(f"  进度: {updated}/{len(ids)}")
            log.info(f"  完成: {updated}/{len(ids)}")
        log.info(f"\n全部完成: {total_new} 个标签")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
