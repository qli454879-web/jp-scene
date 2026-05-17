#!/usr/bin/env python3
"""修复双语境词的日常解析质量问题：
1. 整段日语 → 用中文重写（可夹杂日语词汇）
2. 发音/读音内容 → 删除，改为语言来源/日常用法
3. 日常解析太偏游戏 → 调整为日常用法为主
只处理 insight_text 包含「日常用语解析」+「游戏术语解析」的词条。
"""
import os, sys, json, asyncio, argparse, re, traceback

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
from ai_service import AIService

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
CONCURRENCY = int(os.getenv("FIX_INSIGHT_CONCURRENCY", "10"))

ai = AIService()

# 发音教学内容关键词（教你怎么读，不是简单提到"发音"二字）
PRON_PATTERNS = [
    r'发音注意', r'读音注意', r'读错', r'读法',
    r'声调', r'重音在', r'平板型', r'头高型', r'中高型',
    r'発音.*注意', r'アクセント', r'イントネーション',
    r'念成', r'读起来', r'要读成', r'别读成', r'不要读成',
    r'音调', r'拍型',
    r'重音.*拍', r'第.*拍.*降',
    r'注意别把', r'发音.*简单', r'发音.*难', r'发音.*容易',
    r'读的时候', r'注意是.*音', r'注意发音', r'发成',
]

# 日语句式特征（判断是否整段日语）
JP_SENTENCE_PATTERNS = [
    r'です', r'ます', r'しました', r'している', r'されている',
    r'である', r'という', r'しない', r'できない',
]

# 游戏偏向关键词（日常解析中过多出现说明太偏游戏）
GAMING_HEAVY_PATTERNS = [
    r'游戏圈', r'只在.*玩家', r'不玩游戏', r'玩家圈子', r'游戏内',
    r'游戏里', r'开黑', r'对局', r'队友',
]


def get_db_conn():
    return psycopg.connect(
        SUPABASE_DB_URL,
        prepare_threshold=None,
        options="-c statement_timeout=30000",
    )


def split_insight(insight_text: str) -> tuple:
    """将 insight_text 拆分为 (daily_part, gaming_part)"""
    if not insight_text:
        return "", ""
    parts = insight_text.split("游戏术语解析", 1)
    daily = parts[0].replace("日常用语解析", "").strip()
    if daily.startswith("\n\n"):
        daily = daily[2:].strip()
    elif daily.startswith("\n"):
        daily = daily[1:].strip()
    daily = daily.strip()
    gaming = parts[1].strip() if len(parts) > 1 else ""
    return daily, gaming


def kana_ratio(text: str) -> float:
    """日文假名占比"""
    if not text:
        return 0.0
    kana_count = len(re.findall(r'[ぁ-んァ-ン]', text))
    return kana_count / len(text) if len(text) > 0 else 0.0


def has_pronunciation(text: str) -> bool:
    """是否包含发音教学内容"""
    for pat in PRON_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def has_jp_sentences(text: str) -> bool:
    """是否包含整段日语句式（不仅是词汇）"""
    if not text:
        return False
    match_count = 0
    for pat in JP_SENTENCE_PATTERNS:
        if re.search(pat, text):
            match_count += 1
    return match_count >= 2  # 至少2个日语句式特征


def is_gaming_heavy(text: str) -> bool:
    """日常解析是否太偏游戏"""
    if not text:
        return False
    score = 0
    for pat in GAMING_HEAVY_PATTERNS:
        if re.search(pat, text):
            score += 1
    return score >= 2


def needs_fix(daily: str) -> tuple:
    """检查日常解析是否需要修复，返回 (need_fix, reasons)"""
    reasons = []
    kr = kana_ratio(daily)
    is_jp = has_jp_sentences(daily)
    if is_jp:
        reasons.append(f"整段日语句式")
    elif kr > 0.3:
        reasons.append(f"日语假名占比 {kr:.0%}")
    if has_pronunciation(daily):
        reasons.append("包含发音内容")
    if is_gaming_heavy(daily):
        reasons.append("日常解析太偏游戏")
    return len(reasons) > 0, reasons


def save_daily_insight(conn, word_id: str, daily: str, gaming: str) -> bool:
    """仅更新 insight_text，保留其他字段不动"""
    merged = f"日常用语解析\n{daily}\n\n游戏术语解析\n{gaming}"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE vocab_library SET insight_text = %s WHERE id = %s",
                (merged, word_id),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"  保存失败 (id={word_id}): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


async def fix_one(sem, conn, row, dry_run: bool, stats: dict):
    word_id, word, reading, meaning, insight_text = row
    daily, gaming = split_insight(insight_text)
    fix_needed, reasons = needs_fix(daily)

    if not fix_needed:
        return True  # 无需修复

    tag = ", ".join(reasons)
    print(f"  🔧 {word}: {tag}")
    stats["to_fix"] += 1

    if dry_run:
        stats["dry_run"] += 1
        return True

    async with sem:
        try:
            # 使用改进后的 _default_prompt 重新生成日常解析
            enriched = await ai.enrich_library_entry(
                word=word, reading=reading, meaning=meaning, category=None
            )
            if not enriched:
                print(f"    ✗ AI 返回空")
                stats["ai_empty"] += 1
                return False

            new_daily = (enriched.get("insight_text") or "").strip()
            if not new_daily:
                print(f"    ✗ 新日常解析为空")
                stats["ai_empty"] += 1
                return False

            # 二次检查：新生成的日常解析是否还有问题
            still_kana = kana_ratio(new_daily)
            still_jp = has_jp_sentences(new_daily)
            still_pron = has_pronunciation(new_daily)

            if still_jp:
                print(f"    ⚠ 新解析仍是日语句式，跳过保存")
                print(f"    新内容: {new_daily[:120]}...")
                stats["still_bad"] += 1
                return False
            if still_kana > 0.5:
                print(f"    ⚠ 新解析仍有 {still_kana:.0%} 假名，跳过保存")
                print(f"    新内容: {new_daily[:120]}...")
                stats["still_bad"] += 1
                return False
            if still_pron:
                print(f"    ⚠ 新解析仍包含发音内容，跳过保存")
                print(f"    新内容: {new_daily[:120]}...")
                stats["still_bad"] += 1
                return False

            if save_daily_insight(conn, word_id, new_daily, gaming):
                print(f"    ✓ 已更新 (旧假名{kana_ratio(daily):.0%} → 新假名{still_kana:.0%})")
                stats["fixed"] += 1
                return True
            else:
                stats["save_fail"] += 1
                return False

        except Exception as e:
            print(f"    ✗ 异常: {e}")
            traceback.print_exc()
            stats["error"] += 1
            return False


async def main():
    parser = argparse.ArgumentParser(description="修复双语境词日常解析质量")
    parser.add_argument("--dry-run", action="store_true", help="仅检查不修复")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数")
    parser.add_argument("--all", action="store_true", help="修复所有双语境词（默认只修有问题的）")
    args = parser.parse_args()

    conn = get_db_conn()
    cur = conn.cursor()

    # 仅查双语境词
    cur.execute("""
        SELECT id, word, reading,
               COALESCE(meaning, '') as meaning,
               insight_text
        FROM vocab_library
        WHERE is_ai_enriched = TRUE
          AND insight_text LIKE '%日常用语解析%'
          AND insight_text LIKE '%游戏术语解析%'
        ORDER BY word
    """)
    rows = cur.fetchall()
    print(f"双语境词总数: {len(rows)}")

    # 统计
    stats = {
        "total": len(rows),
        "to_fix": 0,
        "dry_run": 0,
        "fixed": 0,
        "ai_empty": 0,
        "still_bad": 0,
        "save_fail": 0,
        "error": 0,
        "ok": 0,
    }

    # 筛选需修复的
    to_process = []
    for row in rows:
        daily, gaming = split_insight(row[4])
        fix_needed, reasons = needs_fix(daily)
        if fix_needed or args.all:
            to_process.append(row)
        else:
            stats["ok"] += 1

    print(f"需修复: {len(to_process)} / 无需修复: {stats['ok']}")
    if args.limit > 0:
        to_process = to_process[:args.limit]
        print(f"限制处理: {args.limit} 条")

    if args.dry_run:
        print("\n⚠ DRY RUN 模式，不实际修改\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [fix_one(sem, conn, row, args.dry_run, stats) for row in to_process]
    await asyncio.gather(*tasks)

    print(f"\n{'='*50}")
    print(f"完成: 总计 {stats['total']} | 需修 {stats['to_fix']} | 已修 {stats['fixed']}")
    if stats['dry_run']:
        print(f"  dry-run 跳过: {stats['dry_run']}")
    print(f"  AI空响应: {stats['ai_empty']}")
    print(f"  二次检查不通过: {stats['still_bad']}")
    print(f"  保存失败: {stats['save_fail']}")
    print(f"  异常: {stats['error']}")
    print(f"  无需修复: {stats['ok']}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
