#!/usr/bin/env python3
"""
Fix gaming word classifications and enrich missing AI content.

Usage:
  python3 scripts/fix_gaming_classification.py          # 执行修正
  python3 scripts/fix_gaming_classification.py --dry-run  # 只预览
"""

import os
import sys
import re
import argparse
import logging
import uuid

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
log = logging.getLogger("fix_classify")

# ── Valorant: words that should move FROM tactics TO agents (abilities/roles) ──
VAL_TACTICS_TO_AGENTS = {
    # 能力/技能相关 → agents
    "アビリティ", "アビリティ使って", "ウルト", "アルティメット",
    "グレネード", "モロトフ", "スモーク", "フラッシュ",
    "ダッシュ", "テレポート", "トラップ", "ドローン",
    "ヒール", "リコン", "リバイブ", "シグネチャー",
    "ウォール", "カメラ", "スパイク", "ブラインド",
    "デコイ", "スロー", "スタン",
    # 角色定位 → agents
    "イニシエーター", "デュエリスト", "コントローラー", "センチネル",
}

# ── Valorant: words that should move FROM tactics TO weapons ──
VAL_TACTICS_TO_WEAPONS = {
    "HS", "ワンショット", "フリック", "ヘッドショット",
    "タップ撃ち", "バースト", "ワンタップ", "スプレー",
}

# ── Valorant: words that should move FROM weapons/agents TO tactics ──
VAL_TO_TACTICS = {
    "プリエイム", "置きエイム",  # 战术概念，不是武器
}

# ── Valorant: words that should move FROM tactics TO settings ──
VAL_TACTICS_TO_SETTINGS = {
    "クロスヘア", "DPI", "感度", "解像度",
    "グラフィック", "フレームレート", "FPS",
    "キー設定", "キーバインド", "マウス",
}

# ── Valorant: words that should move FROM tactics TO maps ──
VAL_TACTICS_TO_MAPS = {
    "サイト", "Aサイト", "Bサイト", "Cサイト",
    "ロング", "ショート", "ミッド", "ガレージ", "エルボー",
    "コネクター", "ランプ", "バックサイト", "ベント",
    "スポーン", "スポーン地点", "ヘブン",
}

# ── Valorant: words that should move FROM tactics TO ranks ──
VAL_TACTICS_TO_RANKS = {
    "ランク", "段位", "アイアン", "ブロンズ", "シルバー",
    "ゴールド", "プラチナ", "ダイヤモンド", "アセンダント",
    "イモータル", "レディアント",
    "ランクマ", "ランクマッチ", "MMR", "コンペティティブ",
    "アップランク", "ダウンランク", "スマーフ", "ブースト",
}

# ── LoL: words that should move FROM tactics TO champions ──
LOL_TACTICS_TO_CHAMPIONS = {
    "スタン", "スネア", "スロー", "ノックアップ", "ノックバック",
    "サプレス", "タウント", "チャーム", "フィアー",
    "シールド", "ブリンク", "クールダウン", "CD",
    "マナ", "パッシブ", "スキン", "スタック",
    "CC", "AP", "AD", "DPS",
    "スキルショット", "ポイントクリック",
    "Q", "W", "E", "R",
    "イグナイト", "スマイト", "バリア", "クレンズ", "テレポート",
    "サモナースペル", "インビジブル",
    "レベルアップ", "強化", "弱体化",
    "魔法ダメージ", "物理ダメージ", "確定ダメージ",
    "無敵", "レジスト",
    "回復スキル",
}

# ── LoL: words that should move FROM tactics TO items ──
LOL_TACTICS_TO_ITEMS = {
    "ボイドスタッフ", "トリニティフォース", "インフィニティエッジ",
    "ラバドンデスキャップ", "ゾーニャの砂時計", "ゾーニャ",
    "ブラッドサースター", "ステラックの篭手", "モレロノミコン",
    "サンファイアイギス", "リッチベイン", "ルーデンテンペスト",
    "ソーンメイル", "ワーモグアーマー", "ブラッククリーバー",
    "ガーディアンエンジェル", "ナッシャートゥース",
    "ドランブレード", "ドランリング", "ドランシールド",
    "ブーツ", "ポーション", "トリンケット",
    "ワード", "ピンクワード", "デワード",
    "コアアイテム", "スタートアイテム", "ミシックアイテム",
    "レジェンダリー",
}

# ── LoL: words that should move FROM tactics TO objectives ──
LOL_TACTICS_TO_OBJECTIVES = {
    "ピン", "マップ", "マップ確認", "視界", "視界取り",
    "ドラゴン", "バロン", "ヘラルド", "ネクサス",
    "インヒビター", "タワー", "タワー下", "ミニオン",
    "ブルー", "レッド", "ブッシュ",
    "リスポーン", "リコール",
    "CS", "ゴールド", "経験値",
    "ファーストブラッド", "キル",
    "アシスト", "デス", "KDA",
    "ダブルキル", "トリプルキル", "クアッドラキル", "ペンタキル", "エース",
}

# ── LoL: words that should move FROM tactics TO roles ──
LOL_TACTICS_TO_ROLES = {
    "トップ", "ミッド", "ジャングル", "ボット", "ADC",
    "サポート", "キャリー", "レーン", "サイドレーン",
    "レーナー", "ファイター", "タンク", "アサシン",
    "メイジ", "マークスマン", "ローム",
    "レーン戦", "レーニングフェーズ",
    "オートフィル",
}

# ── Fix inaccurate meanings ──
MEANING_FIXES = {
    # Valorant
    "スパイク": "Spike；炸弹（spike），Valorant中进攻方需要安放的目标物",
    "カメラ": "摄像头技能（camera），如Cypher的监控摄像头能力",
    "ウォール": "墙体技能；屏障（wall），如Sage的冰墙或Viper的毒墙",
    "アウトロー": "Outlaw；双管狙击枪，打身体高伤",
    "ラーク": "潜伏（lurk），在敌人意想不到的位置蹲守，切断回防路线",
    "ラスト": "最后一个（last），指残局中最后存活的玩家",
    "ハーフ": "半场（half），12回合后攻守互换",
    "ヘル": "地狱（hell），吐槽局面极其糟糕",
    "グッドラック": "祝好运（good luck），对局开始时的礼貌用语",
    "ヒール": "治疗技能（heal），回复生命值的能力",
    "カニ歩き": "螃蟹步；左右横移对枪（crab walk/strafe shooting）",
    "投げ物": "投掷物；技能道具（utility/nades），包括闪光、烟雾等",
    "詰める": "前压；推进（push forward），向前施压压缩敌人空间",
    "リバイブ": "复活技能（revive），如Sage的终极技能",
    "ドローン": "无人机技能（drone），如Sova的侦查无人机",
    "トラップ": "陷阱技能（trap），用于侦查或限制敌人移动的装备",
    "テレポート": "传送技能（teleport），如Yoru/Chamber的瞬移能力",
    "グレネード": "手雷技能（grenade），范围伤害投掷物，如Raze的手雷",
    "モロトフ": "燃烧弹技能（molotov/incendiary），火焰范围伤害",
    "アルティメット": "终极技能（ultimate），每个特工最强大的技能",
    "シグネチャー": "标志技能（signature ability），每回合免费刷新",
    "リコン": "侦查技能（recon），信息收集能力，如Sova的侦查箭",
    "デコイ": "诱饵技能（decoy），制造假象迷惑敌人，如Yoru的脚步声",
    "ブラインド": "致盲效果（blind），使敌人视野受限的状态",
    "スロー": "减速效果（slow），降低敌人移动速度的状态",

    # LoL
    "マナ": "法力值（mana），释放技能消耗的资源",
    "スタック": "层数；叠加（stack），累积的技能或物品效果",
    "スキン": "皮肤（skin），英雄的外观装扮",
    "クールダウン": "冷却时间（cooldown），技能再次可用前的等待时间",
    "パッシブ": "被动技能（passive），自动生效无需主动释放的能力",
    "スキルショット": "指向性技能（skillshot），需手动瞄准方向的技能",
    "ポイントクリック": "点选技能（point-and-click），点击目标直接释放",
    "イグナイト": "点燃（Ignite），召唤师技能，对敌人造成持续真实伤害",
    "スマイト": "惩戒（Smite），打野必备召唤师技能，对野怪造成高额伤害",
    "バリア": "屏障（Barrier），召唤师技能，提供临时护盾",
    "クレンズ": "净化（Cleanse），召唤师技能，移除控制效果",
    "テレポート": "传送（Teleport），召唤师技能，传送到友方单位",
    "サモナースペル": "召唤师技能（summoner spell），每局可选两个的额外技能",
    "インビジブル": "隐形（invisible），无法被敌人看见的状态",
    "ローム": "游走（roam），离开自己的路线去支援其他路",
    "スノーボール": "雪球效应（snowball），优势不断积累扩大最终碾压",
    "スローイング": "送优势（throwing），优势局因失误被翻盘",
    "メタ": "版本答案（meta），当前版本最强势的打法和英雄选择",
    "パッチ": "版本更新（patch），游戏平衡性调整补丁",
    "ナーフ": "削弱（nerf），降低英雄或装备的强度",
    "バフ": "增强（buff），提高英雄或装备的强度",
    "OP": "过强（overpowered），英雄或装备过于强力",
    "チャンス": "有机会！（it's a chance!），可以打的信号",
    "フィード": "送人头（feed），不断被杀让对面发育起来",
    "コーチング": "指导（coaching），高手教新手提升游戏水平",
    "カウンター": "克制（counter），针对性地选择英雄或打法反制对手",
    "フリーズ": "控线（freeze），保持兵线位置不变以安全发育",
    "プッシュ力": "推线能力（wave clear），快速清除小兵的能力",
    "ファーム": "发育；刷钱（farm），击杀小兵积累经济和经验",
    "ローム": "游走（roam），离开自己路线去其他路支援gank",
    "チャット": "游戏内聊天（in-game chat），与队友或对手文字交流",
    "トロール": "故意捣乱（troll），恶意破坏游戏体验的行为",
    "レジスト": "抗性（resistance），减少受到某类伤害的属性",
    "ワンチャン": "说不定有机会（one chance），劣势中仍有一线翻盘希望",
    "無理": "不行；没戏（impossible），形容情况无法挽回",
    "引いて": "后撤（back off），往后退拉开距离",
    "ファイト": "加油！（fight!），鼓励队友的喊话",
    "川": "河道（river），地图中央的河流区域",
    "弱体化": "削弱（nerf/debuff），能力被降低",
    "強化": "增强（buff/strengthen），能力得到提升",
    "ミスった": "我失误了（I messed up），承认自己的操作失误",
    "行くよ": "我要上了（I'm going in），准备发起进攻的信号",
    "行ける": "能赢（we can win），判断可以打赢的局面",
    "任せて": "交给我（leave it to me），表示自己可以处理",
    "待って": "等一下（wait），请求队友等待的信号",
    "逃げて": "快跑（run away），撤退的紧急呼叫",
    "うまい": "玩得好（well played），夸奖队友的操作",
    "ごめん": "对不起（sorry），为自己的失误道歉",
    "お疲れ様です": "辛苦了（good game），对局结束时的礼貌用语",
    "ダメージ": "伤害（damage），对敌人造成的生命值损失",
    "カメラ": "摄像头（camera），监控或观察特定区域的设备",
}


def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")


def fix_tags(conn, word_set, remove_tag, add_tag):
    """将匹配 words 的词条从 remove_tag 移到 add_tag"""
    count = 0
    with conn.cursor() as cur:
        for word in word_set:
            # 先查看当前tags
            cur.execute(
                "SELECT id, tags FROM vocab_library WHERE word = %s AND tags @> ARRAY[%s]::text[]",
                (word, remove_tag)
            )
            for row_id, tags in cur.fetchall():
                new_tags = [t for t in tags if t != remove_tag]
                if add_tag not in new_tags:
                    new_tags.append(add_tag)
                cur.execute(
                    "UPDATE vocab_library SET tags = %s WHERE id = %s",
                    (new_tags, row_id)
                )
                count += 1
    return count


def fix_meaning(conn, word, new_meaning):
    """修正释义"""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE vocab_library SET meaning = %s WHERE word = %s AND tags && ARRAY['gaming_valorant','gaming_lol']::text[]",
            (new_meaning, word)
        )
        return cur.rowcount


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只预览不执行")
    args = parser.parse_args()

    conn = get_conn()
    conn.autocommit = True

    # ── 预览：查看当前分类 ──
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tag, COUNT(*) FROM vocab_library, unnest(tags) AS tag
            WHERE tag LIKE 'gaming_val_%' OR tag LIKE 'gaming_lol_%'
            GROUP BY tag ORDER BY tag
        """)
        log.info("=== 当前分类分布 ===")
        for tag, cnt in cur.fetchall():
            log.info(f"  {tag}: {cnt}")

    if args.dry_run:
        log.info("\n=== DRY RUN === 未实际修改")
        conn.close()
        return

    # ── Step 1: Valorant 重新分类 ──
    reclass_count = 0

    log.info("\n--- Valorant: tactics → agents (abilities/roles) ---")
    n = fix_tags(conn, VAL_TACTICS_TO_AGENTS, "gaming_val_tactics", "gaming_val_agents")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- Valorant: tactics → weapons ---")
    n = fix_tags(conn, VAL_TACTICS_TO_WEAPONS, "gaming_val_tactics", "gaming_val_weapons")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- Valorant: tactics → settings ---")
    n = fix_tags(conn, VAL_TACTICS_TO_SETTINGS, "gaming_val_tactics", "gaming_val_settings")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- Valorant: tactics → maps ---")
    n = fix_tags(conn, VAL_TACTICS_TO_MAPS, "gaming_val_tactics", "gaming_val_maps")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- Valorant: tactics → ranks ---")
    n = fix_tags(conn, VAL_TACTICS_TO_RANKS, "gaming_val_tactics", "gaming_val_ranks")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    # ── Step 2: LoL 重新分类 ──
    log.info("\n--- LoL: tactics → champions ---")
    n = fix_tags(conn, LOL_TACTICS_TO_CHAMPIONS, "gaming_lol_tactics", "gaming_lol_champions")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- LoL: tactics → items ---")
    n = fix_tags(conn, LOL_TACTICS_TO_ITEMS, "gaming_lol_tactics", "gaming_lol_items")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- LoL: tactics → objectives ---")
    n = fix_tags(conn, LOL_TACTICS_TO_OBJECTIVES, "gaming_lol_tactics", "gaming_lol_objectives")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    log.info("--- LoL: tactics → roles ---")
    n = fix_tags(conn, LOL_TACTICS_TO_ROLES, "gaming_lol_tactics", "gaming_lol_roles")
    log.info(f"  修正 {n} 条")
    reclass_count += n

    # ── Step 3: 修正释义 ──
    log.info("\n--- 修正释义 ---")
    meaning_count = 0
    for word, new_meaning in MEANING_FIXES.items():
        n = fix_meaning(conn, word, new_meaning)
        if n > 0:
            meaning_count += n
            log.info(f"  {word}: 修正 {n} 条")
    log.info(f"  共修正 {meaning_count} 条释义")

    # ── 最终分布 ──
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tag, COUNT(*) FROM vocab_library, unnest(tags) AS tag
            WHERE tag LIKE 'gaming_val_%' OR tag LIKE 'gaming_lol_%'
            GROUP BY tag ORDER BY tag
        """)
        log.info("\n=== 修正后分类分布 ===")
        for tag, cnt in cur.fetchall():
            log.info(f"  {tag}: {cnt}")

    conn.close()
    log.info(f"\n全部完成！重新分类 {reclass_count} 条，修正释义 {meaning_count} 条")


if __name__ == "__main__":
    main()
