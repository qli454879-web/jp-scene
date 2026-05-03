from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from supabase import create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.environ.get("SUPABASE_KEY", "").strip()
BUCKET_NAME = "vocab-audio"

BASE_DIR = Path("/sessions/69ea219cffe1d8535cb99cd3/workspace/词汇/考研日语核心3500词")
CSV_PATH = BASE_DIR / "考研核心3500_严格入库_批量版.csv"
MP3_DIR = BASE_DIR / "mp3"
REPORT_PATH = BASE_DIR / "考研核心3500_上传结果.json"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("缺少 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY 环境变量")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_report(**kwargs):
    REPORT_PATH.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")


def list_remote_files() -> set[str]:
    remote = set()
    offset = 0
    while True:
        res = supabase.storage.from_(BUCKET_NAME).list("", {"limit": 1000, "offset": offset})
        if not res:
            break
        names = [f["name"] for f in res if f.get("name") and f["name"] != ".emptyFolderPlaceholder"]
        remote.update(names)
        if len(names) < 1000:
            break
        offset += 1000
    return remote


def upload_mp3() -> tuple[int, int]:
    remote = list_remote_files()
    local_files = sorted([p for p in MP3_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"])
    uploaded = 0
    skipped = 0
    total = len(local_files)
    print(f"MP3 总数: {total}")
    for i, path in enumerate(local_files, start=1):
        name = path.name
        if name in remote:
            skipped += 1
            if i % 50 == 0 or i == total:
                print(f"[mp3 {i}/{total}] 跳过已存在: {name}")
            continue
        try:
            with open(path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=name,
                    file=f,
                    file_options={"content-type": "audio/mpeg", "upsert": "true"},
                )
            uploaded += 1
            print(f"[mp3 {i}/{total}] 上传完成: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                skipped += 1
                print(f"[mp3 {i}/{total}] 已存在: {name}")
            else:
                print(f"[mp3 {i}/{total}] 上传失败: {name} | {e}")
        time.sleep(0.05)
    return uploaded, skipped


def load_records() -> list[dict]:
    rows = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = (row.get("word") or "").strip()
            meaning = (row.get("meaning") or "").strip()
            if not word or not meaning:
                continue
            try:
                examples = json.loads(row.get("examples") or "[]")
            except Exception:
                examples = []
            try:
                social_context = json.loads(row.get("social_context") or "{}")
            except Exception:
                social_context = {}
            try:
                heatmap_data = json.loads(row.get("heatmap_data") or "{}")
            except Exception:
                heatmap_data = {}
            rec = {
                "level": (row.get("level") or "").strip(),
                "word": word,
                "reading": (row.get("reading") or "").strip() or None,
                "meaning": meaning,
                "mp3": (row.get("mp3") or "").strip() or None,
                "pos": (row.get("pos") or "").strip() or None,
                "frequency": int(str(row.get("frequency") or "0") or "0") or None,
                "examples": examples,
                "social_context": social_context,
                "heatmap_data": heatmap_data,
                "insight_text": (row.get("insight_text") or "").strip() or None,
                "image_url": (row.get("image_url") or "").strip() or None,
                "is_ai_enriched": str(row.get("is_ai_enriched") or "").strip().lower() in ("true", "1", "yes"),
                "order_no": int(str(row.get("order_no") or "0") or "0") or None,
                "tags": [str((row.get("level") or "").strip()).lower(), "kaoyan"],
            }
            rows.append(rec)
    return rows


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def upload_vocab() -> tuple[int, int]:
    rows = load_records()
    total = len(rows)
    uploaded = 0
    failed = 0
    print(f"词条总数: {total}")
    for batch_no, batch in enumerate(chunked(rows, 100), start=1):
        try:
            supabase.table("vocab_library").upsert(batch, on_conflict="level,word").execute()
            uploaded += len(batch)
            print(f"[csv batch {batch_no}] 成功: {len(batch)} 条")
        except Exception as e:
            failed += len(batch)
            print(f"[csv batch {batch_no}] 失败: {len(batch)} 条 | {e}")
        time.sleep(0.2)
    return uploaded, failed


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"找不到 CSV: {CSV_PATH}")
    if not MP3_DIR.exists():
        raise SystemExit(f"找不到 mp3 文件夹: {MP3_DIR}")

    print("开始上传考研核心词到 Supabase...")
    mp3_uploaded, mp3_skipped = upload_mp3()
    vocab_uploaded, vocab_failed = upload_vocab()

    save_report(
        status="done",
        mp3_uploaded=mp3_uploaded,
        mp3_skipped=mp3_skipped,
        vocab_uploaded=vocab_uploaded,
        vocab_failed=vocab_failed,
        csv_path=str(CSV_PATH),
        mp3_dir=str(MP3_DIR),
    )
    print("上传完成")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
