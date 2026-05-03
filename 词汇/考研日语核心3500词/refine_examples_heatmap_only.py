from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
import os
from pathlib import Path

import requests


BASE_DIR = Path("/sessions/69ea219cffe1d8535cb99cd3/workspace/词汇/考研日语核心3500词")
CSV_PATH = BASE_DIR / "考研核心3500_严格入库_批量版.csv"
TMP_PATH = BASE_DIR / "考研核心3500_严格入库_批量版.examples_heatmap.tmp.csv"
BACKUP_PATH = BASE_DIR / "考研核心3500_严格入库_批量版.backup.csv"
PROGRESS_PATH = BASE_DIR / "考研核心3500_例句场所精修进度.json"

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview").strip()
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
BASE_URL_CANDIDATES = [
    x.strip() for x in os.environ.get(
        "OPENAI_BASE_URLS",
        os.environ.get("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1")
    ).split(",") if x.strip()
]

HEAT_TAGS = [
    "交通", "便利店", "餐厅", "购物", "医院", "家里", "学校", "面试", "会议", "邮件",
    "商务礼仪", "电话", "公司日常", "SNS", "游戏", "弹幕", "寒暄", "请求", "委婉拒绝",
    "赞美", "吐槽/调侃", "情绪发泄", "职场潜规则",
    "日常会话", "朋友闲聊", "家庭场景", "职场沟通", "会议汇报", "邮件书面",
    "阅读理解", "评论/议论文", "新闻报道", "课堂/讲座", "学习复述", "作文表达",
    "人物描写", "情绪表达", "抽象讨论", "社会话题", "说明文", "叙事描写"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全量")
    parser.add_argument("--start", type=int, default=1, help="从第几行开始处理，行号从 1 开始（不含表头）")
    parser.add_argument("--batch-size", type=int, default=5, help="每次请求处理几个词，建议 3-5")
    parser.add_argument("--sleep", type=float, default=0.5, help="每条之间额外等待秒数")
    parser.add_argument("--no-backup", action="store_true", help="不生成 CSV 备份")
    return parser.parse_args()


def save_progress(**kwargs):
    PROGRESS_PATH.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_rows() -> list[dict]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def backup_csv():
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_bytes(CSV_PATH.read_bytes())


def write_rows(rows: list[dict]):
    with TMP_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    TMP_PATH.replace(CSV_PATH)


def resolved_urls() -> list[str]:
    urls = []
    for url in BASE_URL_CANDIDATES:
        u = url.strip()
        if not u:
            continue
        if u.endswith("/v1/chat/completions"):
            urls.append(u)
        elif u.endswith("/v1"):
            urls.append(u.rstrip("/") + "/chat/completions")
        else:
            urls.append(u.rstrip("/") + "/v1/chat/completions")
    return urls


def parse_existing_examples(text: str) -> list[dict]:
    try:
        arr = json.loads(text or "[]")
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    cleaned = []
    seen = set()
    for item in arr:
        if not isinstance(item, dict):
            continue
        jp = str(item.get("jp", "")).strip()
        cn = str(item.get("cn", "")).strip()
        if not jp:
            continue
        key = (jp, cn)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"jp": jp, "cn": cn})
    return cleaned


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_prompt(batch: list[dict]) -> str:
    payload_items = []
    for i, row in enumerate(batch, start=1):
        payload_items.append({
            "idx": i,
            "word": row.get("word", ""),
            "reading": row.get("reading", ""),
            "meaning": row.get("meaning", ""),
            "pos": row.get("pos", ""),
        })
    return (
        "只返回 JSON。对每个词输出 3 条自然日语例句+中文翻译，以及从给定标签里选出的 Top3 高频场所/语用标签。"
        "不要解释，不要输出多余字段。例句必须自然，不要元话语。heatmap_data 只能保留 3 个标签。"
        f"标签池：{HEAT_TAGS}。"
        'JSON: {"items":[{"idx":1,"examples":[{"jp":"...","cn":"..."},{"jp":"...","cn":"..."},{"jp":"...","cn":"..."}],"heatmap_data":{"标签1":95,"标签2":85,"标签3":75}}]}'
        f"\n词条列表：{json.dumps(payload_items, ensure_ascii=False)}"
    )


def call_ai_for_batch(batch: list[dict]) -> dict[int, tuple[list[dict], dict]]:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_prompt(batch)}],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    last_err = None
    for url in resolved_urls():
        for attempt in range(1, 5):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=180)
                if resp.status_code == 429:
                    wait_s = min(20 * attempt, 90)
                    print(f"    限流：{wait_s}s 后重试 | {url}")
                    time.sleep(wait_s)
                    last_err = RuntimeError(f"429 on {url} attempt {attempt}")
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)
                items = data.get("items")
                if not isinstance(items, list):
                    raise RuntimeError("items 不是数组")
                result = {}
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("idx")
                    if not isinstance(idx, int):
                        try:
                            idx = int(idx)
                        except Exception:
                            continue
                    result[idx] = (
                        normalize_examples(item.get("examples")),
                        normalize_heatmap(item.get("heatmap_data")),
                    )
                return result
            except Exception as e:
                last_err = e
                if attempt >= 4:
                    break
                wait_s = min(8 * attempt, 30)
                print(f"    请求失败：{wait_s}s 后重试 | {url} | {e}")
                time.sleep(wait_s)
    raise last_err if last_err else RuntimeError("AI 未返回结果")


def normalize_examples(v) -> list[dict]:
    if not isinstance(v, list):
        raise RuntimeError("examples 不是数组")
    cleaned = []
    seen = set()
    for item in v:
        if not isinstance(item, dict):
            continue
        jp = str(item.get("jp", "")).strip()
        cn = str(item.get("cn", "")).strip()
        if not jp:
            continue
        key = (jp, cn)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"jp": jp, "cn": cn})
    if len(cleaned) != 3:
        raise RuntimeError(f"examples 数量不等于 3，实际 {len(cleaned)}")
    return cleaned


def normalize_heatmap(v) -> dict:
    if not isinstance(v, dict):
        raise RuntimeError("heatmap_data 不是对象")
    cleaned = {}
    for k, val in v.items():
        kk = str(k).strip()
        if kk not in HEAT_TAGS:
            continue
        try:
            cleaned[kk] = int(val)
        except Exception:
            continue
    if len(cleaned) != 3:
        raise RuntimeError(f"合法 heatmap 标签数量不等于 3，实际 {len(cleaned)}")
    return dict(sorted(cleaned.items(), key=lambda x: x[1], reverse=True)[:3])


def main():
    args = parse_args()
    rows = load_rows()
    if not args.no_backup:
        backup_csv()

    total = len(rows)
    done = 0
    failed = 0
    start_index = max(args.start - 1, 0)
    stop_index = total if args.limit == 0 else min(total, start_index + args.limit)

    save_progress(
        status="running",
        updated_at=now_iso(),
        total=total,
        start=args.start,
        limit=args.limit,
        done=0,
        failed=0,
        current_word="",
        current_index=0,
        csv_path=str(CSV_PATH)
    )

    targets = list(range(start_index, stop_index))
    for batch in chunked(targets, max(args.batch_size, 1)):
        batch_rows = [rows[idx] for idx in batch]
        batch_words = "、".join(r.get("word", "") for r in batch_rows)
        print(f"[{batch[0]+1}-{batch[-1]+1}/{total}] 处理中：{batch_words}")
        save_progress(
            status="running",
            updated_at=now_iso(),
            total=total,
            start=args.start,
            limit=args.limit,
            done=done,
            failed=failed,
            current_word=batch_words,
            current_index=batch[-1] + 1,
            csv_path=str(CSV_PATH)
        )
        try:
            result_map = call_ai_for_batch(batch_rows)
            for inner_i, idx in enumerate(batch, start=1):
                row = rows[idx]
                word = row.get("word", "")
                if inner_i not in result_map:
                    failed += 1
                    print(f"  -> 失败：{word} | AI 未返回该词")
                    continue
                examples, heatmap = result_map[inner_i]
                row["examples"] = json.dumps(examples, ensure_ascii=False)
                row["heatmap_data"] = json.dumps(heatmap, ensure_ascii=False)
                done += 1
                print(f"  -> 完成：{word}")
            write_rows(rows)
        except Exception as e:
            failed += len(batch)
            print(f"  -> 批次失败：{batch_words} | {e}")
        save_progress(
            status="running",
            updated_at=now_iso(),
            total=total,
            start=args.start,
            limit=args.limit,
            done=done,
            failed=failed,
            current_word=batch_words,
            current_index=batch[-1] + 1,
            csv_path=str(CSV_PATH)
        )
        time.sleep(max(args.sleep, 0))

    save_progress(
        status="done",
        updated_at=now_iso(),
        total=total,
        start=args.start,
        limit=args.limit,
        done=done,
        failed=failed,
        current_word="",
        current_index=stop_index,
        csv_path=str(CSV_PATH)
    )
    print(f"done={done} failed={failed}")
    print(CSV_PATH)


if __name__ == "__main__":
    main()
