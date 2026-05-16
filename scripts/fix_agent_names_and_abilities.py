#!/usr/bin/env python3
"""
修正 5 个特工中文译名 + 补充所有特工的 4 个技能词条。

Usage:
  python3 scripts/fix_agent_names_and_abilities.py --dry-run
  python3 scripts/fix_agent_names_and_abilities.py
"""

import os
import sys
import argparse
import logging
import traceback

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
log = logging.getLogger("fix_agents")

# ── 5 个特工中文名修正 ──
AGENT_NAME_FIXES = {
    "オーメン": "幽影 | Omen，暗影控场，外号「鬼」「暗影」",
    "KAY/O": "KO | KAY/O，机器人先锋，技能压制，外号「机器人」",
    "フェイド": "黑梦 | Fade，土耳其先锋，恐惧侦查，外号「眼魔」",
    "ハーバー": "哈勃 | Harbor，印度控场，水系，外号「水男」",
    "クローヴ": "暮蝶 | Clove，苏格兰控场，死后仍可放烟，外号「不死女」",
    "デッドロック": "铁壁 | Deadlock，挪威哨位，声波陷阱，外号「网女」",
}

# ── 25 特工 × 4 技能 ──
# 每条: (日文技能名, 日文读音, 中文名 | 日文原名，特工中文名+键位技能，简短说明)
ABILITIES = [
    # ── 决斗者 ──
    # 捷风 Jett
    ("クラウドバースト", "クラウドバースト", "瞬云 | Cloudburst，捷风C技能，投掷一枚烟雾弹遮挡视野"),
    ("アップドラフト", "アップドラフト", "上升气流 | Updraft，捷风Q技能，利用气流瞬间跃起"),
    ("テイルウィンド", "テイルウィンド", "顺风 | Tailwind，捷风E技能，向移动方向快速冲刺"),
    ("ブレードストーム", "ブレードストーム", "剑刃风暴 | Blade Storm，捷风X大招，投掷多把致命飞刀"),

    # 不死鸟 Phoenix
    ("ブレイズ", "ブレイズ", "燃烧墙 | Blaze，不死鸟C技能，召唤一道火焰墙阻挡视野并造成伤害"),
    ("カーブボール", "カーブボール", "闪光曲球 | Curveball，不死鸟Q技能，投掷弧形闪光弹致盲敌人"),
    ("ホットハンズ", "ホットハンズ", "炙热之手 | Hot Hands，不死鸟E技能，投掷火球形成燃烧区域并自我治疗"),
    ("ラン・イット・バック", "ラン・イット・バック", "浴火重生 | Run It Back，不死鸟X大招，标记当前位置，死亡后在标记处复活"),

    # 雷兹 Raze
    ("ブームボット", "ブームボット", "爆破机器人 | Boom Bot，雷兹C技能，释放追踪机器人撞击爆炸"),
    ("ブラストパック", "ブラストパック", "炸药包 | Blast Pack，雷兹Q技能，投掷炸药包造成伤害并弹跳位移"),
    ("ペイント弾", "ペイントだん", "漆彩榴弹 | Paint Shells，雷兹E技能，投掷集束榴弹造成范围伤害"),
    ("ショーストッパー", "ショーストッパー", "晚安焰火 | Showstopper，雷兹X大招，发射火箭筒造成巨大范围伤害"),

    # 芮娜 Reyna
    ("リーア", "リーア", "睥睨 | Leer，芮娜C技能，发射一颗可破坏的视线限制眼球"),
    ("デバウアー", "デバウアー", "噬魂 | Devour，芮娜Q技能，吸取敌方魂珠恢复生命"),
    ("ディスミス", "ディスミス", "放逐 | Dismiss，芮娜E技能，消耗魂珠进入无形状态"),
    ("エンプレス", "エンプレス", "女王降临 | Empress，芮娜X大招，进入狂暴状态提升射速并强化技能"),

    # 夜露 Yoru
    ("フェイクアウト", "フェイクアウト", "假面 | Fakeout，夜露C技能，发射模仿脚步声的诱饵弹"),
    ("ブラインドサイド", "ブラインドサイド", "致盲弹 | Blindside，夜露Q技能，弹射闪光弹致盲敌人"),
    ("ゲートクラッシュ", "ゲートクラッシュ", "穿梭门 | Gatecrash，夜露E技能，放置可传送的裂隙锚点"),
    ("ディメンショナルドリフト", "ディメンショナルドリフト", "次元漂移 | Dimensional Drift，夜露X大招，进入异次元空间自由移动"),

    # 霓虹 Neon
    ("ファストレーン", "ファストレーン", "闪电封墙 | Fast Lane，霓虹C技能，放出两道电流屏障阻挡视野"),
    ("リレーボルト", "リレーボルト", "电能冲击 | Relay Bolt，霓虹Q技能，发射会弹射的电击球"),
    ("ハイギア", "ハイギア", "超速模式 | High Gear，霓虹E技能，爆发加速奔跑"),
    ("オーバードライブ", "オーバードライブ", "超限 | Overdrive，霓虹X大招，从指尖发射持续雷电光束"),

    # 壹决 Iso
    ("コンティンジェンシー", "コンティンジェンシー", "隔断 | Contingency，壹决C技能，召唤一道移动的能量墙"),
    ("アンダーカット", "アンダーカット", "破绽 | Undercut，壹决Q技能，发射穿透墙壁的能量球施加脆弱"),
    ("ダブルタップ", "ダブルタップ", "能量护盾 | Double Tap，壹决E技能，激活吸收伤害的能量护盾"),
    ("キルコントラクト", "キルコントラクト", "决斗领域 | Kill Contract，壹决X大招，将一名敌人拉入1v1决斗领域"),

    # ── 先锋 ──
    # 铁臂 Breach
    ("アフターショック", "アフターショック", "余震 | Aftershock，铁臂C技能，穿透墙壁的爆炸冲击波"),
    ("フラッシュポイント", "フラッシュポイント", "致盲点 | Flashpoint，铁臂Q技能，穿墙致盲闪光"),
    ("フォールトライン", "フォールトライン", "断层 | Fault Line，铁臂E技能，释放沿地面延伸的震波"),
    ("ローリングサンダー", "ローリングサンダー", "滚雷 | Rolling Thunder，铁臂X大招，释放覆盖大片区域的连续震击"),

    # 斯凯 Skye
    ("リグロウス", "リグロウス", "自然治愈 | Regrowth，斯凯C技能，持续治疗周围队友"),
    ("トレイルブレイザー", "トレイルブレイザー", "开路先锋 | Trailblazer，斯凯Q技能，操控一只可跳跃爆炸的召唤兽"),
    ("ガイディングライト", "ガイディングライト", "引路之光 | Guiding Light，斯凯E技能，操控一只可闪光致盲的飞鹰"),
    ("シーカー", "シーカー", "追踪猎手 | Seekers，斯凯X大招，放出三只追踪敌人的致盲精灵"),

    # 猎枭 Sova
    ("オウルドローン", "オウルドローン", "枭型无人机 | Owl Drone，猎枭C技能，控制侦查无人机发射标记镖"),
    ("ショックダート", "ショックダート", "电击箭 | Shock Dart，猎枭Q技能，发射会爆炸的电击箭"),
    ("リコンボルト", "リコンボルト", "侦查箭 | Recon Bolt，猎枭E技能，发射脉冲扫描箭暴露敌人位置"),
    ("ハンターズフューリー", "ハンターズフューリー", "猎人之怒 | Hunter's Fury，猎枭X大招，发射三道穿透全图的能量冲击波"),

    # KAY/O → KO
    ("フラグメント", "フラグメント", "碎片手雷 | FRAG/ment，KOC技能，投掷会多次爆炸的碎片手雷"),
    ("フラッシュドライブ", "フラッシュドライブ", "闪光驱动器 | FLASH/drive，KOQ技能，投掷致盲闪光弹"),
    ("ゼロポイント", "ゼロポイント", "零/点 | ZERO/point，KOE技能，发射抑制刀刃沉默敌人技能"),
    ("ヌルコマンド", "ヌルコマンド", "压制命令 | NULL/cmd，KOX大招，释放大范围脉冲波压制并沉默所有敌人"),

    # 盖可 Gekko
    ("モッシュ", "モッシュ", "爆爆 | Mosh Pit，盖可C技能，投掷酸液炸弹造成范围伤害"),
    ("ウィングマン", "ウィングマン", "鸟人 | Wingman，盖可Q技能，派出可下包/拆包的飞行精灵"),
    ("スラッシュ", "スラッシュ", "闪光精灵 | Dizzy，盖可E技能，派出致盲敌人的飞行精灵"),
    ("スラッシュパック", "スラッシュパック", "鲨鲨 | Thrash，盖可X大招，操控一只可跳跃禁锢敌人的鲨鱼"),

    # 黑梦 Fade
    ("プラウラー", "プラウラー", "潜伏兽 | Prowler，黑梦C技能，释放追踪敌人的恐惧猎兽"),
    ("シーズ", "シーズ", "困缚 | Seize，黑梦Q技能，投掷会束缚并腐蚀敌人的锚球"),
    ("ホーント", "ホーント", "窥探之眼 | Haunt，黑梦E技能，投掷暴露并标记敌人的侦察之眼"),
    ("ナイトフォール", "ナイトフォール", "夜幕降临 | Nightfall，黑梦X大招，释放大范围恐惧波暴露并弱化敌人"),

    # ── 哨兵 ──
    # 贤者 Sage
    ("バリアオーブ", "バリアオーブ", "屏障法球 | Barrier Orb，贤者C技能，放置一面可升起的冰墙"),
    ("スローオーブ", "スローオーブ", "减速法球 | Slow Orb，贤者Q技能，投掷造成地面减速的冰球"),
    ("ヒールオーブ", "ヒールオーブ", "治愈法球 | Healing Orb，贤者E技能，为友方恢复生命值"),
    ("リザレクション", "リザレクション", "复活 | Resurrection，贤者X大招，复活一名阵亡的队友"),

    # 奇乐 Killjoy
    ("ナノスワーム", "ナノスワーム", "纳米蜂群 | Nanoswarm，奇乐C技能，投掷隐形的纳米手雷造成持续伤害"),
    ("アラームボット", "アラームボット", "警报机器人 | Alarmbot，奇乐Q技能，放置会追踪爆炸的警报机器人"),
    ("タレット", "タレット", "自动炮台 | Turret，奇乐E技能，放置自动射击的哨兵炮台"),
    ("ロックダウン", "ロックダウン", "全面封锁 | Lockdown，奇乐X大招，放置大范围强制禁锢的封锁装置"),

    # 零 Cypher
    ("トラップワイヤー", "トラップワイヤー", "绊索 | Trapwire，零C技能，布置隐形绊索暴露并束缚敌人"),
    ("サイバーケージ", "サイバーケージ", "电子烟 | Cyber Cage，零Q技能，远程激活的视野遮蔽装置"),
    ("スパイカメラ", "スパイカメラ", "监视摄像头 | Spycam，零E技能，放置可发射标记镖的监视摄像头"),
    ("ニューラルセフト", "ニューラルセフト", "窃取情报 | Neural Theft，零X大招，从敌方尸体窃取情报暴露所有敌人"),

    # 尚博勒 Chamber
    ("トレードマーク", "トレードマーク", "商标 | Trademark，尚博勒C技能，放置扫描敌人的减速陷阱"),
    ("ヘッドハンター", "ヘッドハンター", "猎头手枪 | Headhunter，尚博勒Q技能，装备一把高精度重型手枪"),
    ("ランデヴー", "ランデヴー", "传送锚点 | Rendezvous，尚博勒E技能，放置两个可瞬间传送的锚点"),
    ("ツール・ド・フォース", "ツール・ド・フォース", "终极之力 | Tour De Force，尚博勒X大招，装备一把一击必杀的强力狙击枪"),

    # 铁壁 Deadlock
    ("グラヴネット", "グラヴネット", "重力网 | GravNet，铁壁C技能，投掷令敌人蹲伏的重力网手雷"),
    ("ソニックセンサー", "ソニックセンサー", "声波传感器 | Sonic Sensor，铁壁Q技能，放置声波传感器探测并眩晕敌人"),
    ("バリアメッシュ", "バリアメッシュ", "屏障网 | Barrier Mesh，铁壁E技能，放置一道阻挡移动的X形屏障"),
    ("アナイアレーション", "アナイアレーション", "湮灭 | Annihilation，铁壁X大招，发射一道轨迹球束缚并消灭首个命中敌人"),

    # 维斯 Vyse
    ("アークローズ", "アークローズ", "弧玫瑰 | Arc Rose，维斯C技能，放置隐形的致盲陷阱"),
    ("レイザーバイン", "レイザーバイン", "剃刀藤 | Razorvine，维斯Q技能，投掷覆盖地面的剃刀荆棘"),
    ("シアー", "シアー", "剪断 | Shear，维斯E技能，召唤一道穿透墙壁的切割墙"),
    ("スティールガーデン", "スティールガーデン", "钢铁花园 | Steel Garden，维斯X大招，释放大范围金属风暴封锁区域"),

    # ── 控场 ──
    # 炼狱 Brimstone
    ("スティムビーコン", "スティムビーコン", "振奋信标 | Stim Beacon，炼狱C技能，放置提升射速的振奋装置"),
    ("インセンディアリー", "インセンディアリー", "燃烧弹 | Incendiary，炼狱Q技能，发射燃烧榴弹造成持续火焰伤害"),
    ("スカイスモーク", "スカイスモーク", "天幕烟雾 | Sky Smoke，炼狱E技能，利用卫星投放远程烟雾"),
    ("オービタルストライク", "オービタルストライク", "轨道打击 | Orbital Strike，炼狱X大招，呼叫卫星轨道激光打击大范围区域"),

    # 蝰蛇 Viper
    ("スネークバイト", "スネークバイト", "蛇咬 | Snake Bite，蝰蛇C技能，发射造成腐蚀伤害的毒液弹"),
    ("ポイズンクラウド", "ポイズンクラウド", "毒云 | Poison Cloud，蝰蛇Q技能，投掷可开关的毒气发射器"),
    ("トキシックスクリーン", "トキシックスクリーン", "毒幕 | Toxic Screen，蝰蛇E技能，释放穿越墙壁的毒气屏障"),
    ("ヴァイパーズピット", "ヴァイパーズピット", "蝰蛇之坑 | Viper's Pit，蝰蛇X大招，制造大面积毒雾区域隐藏自身"),

    # 幽影 Omen
    ("シュラウドステップ", "シュラウドステップ", "暗影步 | Shrouded Step，幽影C技能，短距离传送"),
    ("パラノイア", "パラノイア", "偏执 | Paranoia，幽影Q技能，发射穿透墙壁的致盲球"),
    ("ダークカバー", "ダークカバー", "暗黑屏障 | Dark Cover，幽影E技能，投掷可穿过墙壁的暗影烟雾球"),
    ("フロム・ザ・シャドウ", "フロム・ザ・シャドウ", "黑暗降临 | From the Shadows，幽影X大招，传送到地图任意位置"),

    # 星礈 Astra
    ("グラビティウェル", "グラビティウェル", "重力井 | Gravity Well，星礈C技能，将敌人拉向中心并脆弱化"),
    ("ノヴァパルス", "ノヴァパルス", "新星脉冲 | Nova Pulse，星礈Q技能，释放震荡冲击造成眩晕"),
    ("ネビュラ", "ネビュラ", "星云 | Nebula，星礈E技能，召唤持续的星云烟雾"),
    ("コズミックディバイド", "コズミックディバイド", "宇宙分割 | Cosmic Divide，星礈X大招，从星体形态释放分割战场的大型屏障"),

    # 暮蝶 Clove
    ("ルース", "ルース", "干扰弹 | Ruse，暮蝶C技能，投掷造成衰减的分裂碎片弹"),
    ("ピック・ミー・アップ", "ピック・ミー・アップ", "汲取 | Pick Me Up，暮蝶Q技能，击杀或助攻后激活临时生命值加速"),
    ("メドル", "メドル", "兴奋剂 | Meddle，暮蝶E技能，投掷造成腐烂效果的化学手雷"),
    ("ノット・デッド・イェット", "ノット・デッド・イェット", "尚未死亡 | Not Dead Yet，暮蝶X大招，死后仍可放烟，一段时间后原地复活"),

    # 哈勃 Harbor
    ("カスケード", "カスケード", "瀑布 | Cascade，哈勃C技能，召唤一道可移动的水墙"),
    ("コーヴ", "コーヴ", "湾流 | Cove，哈勃Q技能，投掷一个可阻挡子弹的水球护盾"),
    ("ハイタイド", "ハイタイド", "涨潮 | High Tide，哈勃E技能，释放一道曲线水墙阻挡视野"),
    ("レコニング", "レコニング", "清算 | Reckoning，哈勃X大招，释放追踪敌人的水柱冲击波"),
]

TAGS = ["gaming_valorant", "gaming_val_agents"]


def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")


def fix_agent_names(cur, dry_run):
    """修正 5 个特工的中文译名"""
    fixed = 0
    for word, new_meaning in AGENT_NAME_FIXES.items():
        cur.execute("""
            SELECT id, word, meaning FROM vocab_library
            WHERE word = %s AND tags @> ARRAY['gaming_valorant','gaming_val_agents']::text[]
        """, (word,))
        rows = cur.fetchall()
        for rid, w, old_meaning in rows:
            if dry_run:
                log.info(f"  [DRY] {w}: 「{old_meaning or ''}」→「{new_meaning}」")
            else:
                cur.execute("UPDATE vocab_library SET meaning = %s WHERE id = %s",
                            (new_meaning, rid))
            fixed += 1
        if not rows:
            log.info(f"  ⚠ 未找到词条: {word}")
    return fixed


def insert_abilities(cur, dry_run):
    """插入所有技能词条，已存在则跳过"""
    inserted = 0
    skipped = 0
    for word, reading, meaning in ABILITIES:
        # 检查是否已存在
        cur.execute("""
            SELECT id FROM vocab_library
            WHERE word = %s AND tags @> ARRAY['gaming_valorant','gaming_val_agents']::text[]
        """, (word,))
        if cur.fetchone():
            skipped += 1
            continue

        if dry_run:
            inserted += 1
            if inserted <= 5:
                log.info(f"  [DRY] INSERT: {word} → {meaning[:60]}...")
            continue

        cur.execute("""
            INSERT INTO vocab_library (word, reading, meaning, tags, level)
            VALUES (%s, %s, %s, %s, '一般')
        """, (word, reading, meaning, TAGS))
        inserted += 1

    return inserted, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="仅预览不执行")
    args = parser.parse_args()

    conn = get_conn()
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            # 1. 修正特工名
            log.info("── 修正 5 个特工中文译名 ──")
            fixed = fix_agent_names(cur, args.dry_run)
            log.info(f"修正译名: {fixed} 条")

            # 2. 插入技能词条
            log.info("── 插入特工技能词条 ──")
            inserted, skipped = insert_abilities(cur, args.dry_run)
            log.info(f"技能词条: 新增 {inserted} 条，已存在跳过 {skipped} 条")
    except Exception as e:
        log.error(f"执行出错: {e}")
        traceback.print_exc()
    finally:
        conn.close()

    if args.dry_run:
        log.info("\n=== DRY RUN === 未实际修改")
    else:
        log.info("\n全部完成！特工译名修正 + 技能词条补充")


if __name__ == "__main__":
    main()
