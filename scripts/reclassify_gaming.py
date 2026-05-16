#!/usr/bin/env python3
"""
将现有 gaming_valorant / gaming_lol 词条分配到子分类，并补充常用新词。

Valorant 子分类:
  gaming_val_agents   — 特工名称与特工技能
  gaming_val_weapons  — 枪械
  gaming_val_tactics  — 战术术语与对局交流
  gaming_val_settings — 游戏设置
  gaming_val_ranks    — 段位名称
  gaming_val_maps     — 地图与地图报点

LoL 子分类:
  gaming_lol_champions  — 英雄名称与技能
  gaming_lol_items      — 装备
  gaming_lol_tactics    — 战术与对局交流
  gaming_lol_roles      — 位置与职责
  gaming_lol_objectives — 地图资源与目标

Usage:
  python3 scripts/reclassify_gaming.py          # 全量执行
  python3 scripts/reclassify_gaming.py --dry-run  # 只看统计
"""

import os
import sys
import re
import argparse
import logging
import json
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
log = logging.getLogger("reclassify")

# ── 分类规则 ──
# Valorant Agent names (known)
VAL_AGENTS = {
    "ソーヴァ", "フェイド", "アストラ", "アイソ", "ヨル", "オーメン",
    "サイファー", "キルジョイ", "チェンバー", "ジェット", "ネオン",
    "ヴァイス", "セージ", "ゲッコー", "レイナ", "ブリムストーン",
    "クローヴ", "ヴァイパー", "フェニックス", "デッドロック",
    "レイズ", "スカイ", "ブリーチ", "ハーバー", "KAY/O",
    "フェード", "チェンバー", "ゲッコー",
}

VAL_WEAPON_KEYWORDS = [
    "ヴァンダル", "ファントム", "オペレーター", "マーシャル",
    "ショーティー", "フレンジー", "ゴースト", "バッキー",
    "ブルドッグ", "ジャッジ", "スティンガー", "クラシック",
    "ガーディアン", "シェリフ", "スペクター", "ナイフ",
    "アーマー", "ライトアーマー", "ヘビーアーマー", "フルアーマー",
    "リコイル", "スプレー", "クロスヘア", "ヘッドショット",
    "タップ撃ち", "ワンタップ", "バースト", "プリエイム",
    "置きエイム", "エイム練習", "エイム",
]

VAL_MAP_KEYWORDS = [
    "アセント", "バインド", "サンセット", "ブリーズ", "パール",
    "アイスボックス", "フラクチャー", "ロータス", "ヘイヴン",
    "アビス", "スプリット", "ヘブン",
    "サイト", "Aサイト", "Bサイト", "Cサイト",
    "ロング", "ショート", "ミッド", "ガレージ", "エルボー",
    "コネクター", "ランプ", "バックサイト", "ベント",
    "スポーン", "スポーン地点", "設置地点",
]

VAL_TACTIC_KEYWORDS = [
    "リテイク", "クラッチ", "エコラウンド", "フォースバイ",
    "フルバイ", "セーブ", "フラッシュ", "スモーク", "ダッシュ",
    "プッシュ", "攻める", "守って", "詰める", "引く", "引いて",
    "前に出る", "固まって", "バラけて", "時間稼いで",
    "カバー", "トレード", "ピーク", "ワイドピーク", "ジグルピーク",
    "ラーク", "ホールド", "フェイク", "カニ歩き",
    "情報", "音する", "敵", "味方", "全員",
    "エントリー", "ローテーション", "イニシエーター",
    "デュエリスト", "コントローラー", "センチネル",
    "ラウンド", "クレジット", "GG", "ナイストライ",
    "グッドラック", "ありがとう", "ごめん",
    "解除", "設置", "即解除", "ディフューズ中",
    "アビリティ使って", "投げ物", "スキル",
    "瀕死", "当たってる", "ダメージ", "キル",
]

VAL_SETTINGS_KEYWORDS = [
    "クロスヘア", "DPI", "感度", "設定", "解像度",
    "グラフィック", "フレームレート", "FPS",
    "キー設定", "キーバインド", "マウス",
]

VAL_RANK_KEYWORDS = [
    "ランク", "段位", "アイアン", "ブロンズ", "シルバー",
    "ゴールド", "プラチナ", "ダイヤモンド", "アセンダント",
    "イモータル", "レディアント", "コンペティティブ",
    "競技", "ランクマ", "ランクマッチ", "MMR",
    "アップランク", "ダウンランク",
]

# LoL classification keywords
LOL_CHAMPION_KEYWORDS = [
    "チャンピオン", "ピック", "バン", "スキン",
    "パッシブ", "Q", "W", "E", "R",
    "スキルショット", "ポイントクリック",
    "アサシン", "メイジ", "マークスマン", "ファイター",
    "タンク", "サポート",
]

LOL_ITEM_KEYWORDS = [
    "アイテム", "コアアイテム", "スタートアイテム",
    "トリンケット", "ポーション", "ブーツ", "ワード",
    "ピンクワード", "デワード", "オーブ",
    "ドランブレード", "ドランリング", "ドランシールド",
    "ラバドンデスキャップ", "ゾーニャの砂時計", "ゾーニャ",
    "インフィニティエッジ", "ブラッドサースター",
    "ステラックの篭手", "モレロノミコン", "サンファイアイギス",
    "リッチベイン", "ヴォイドスタッフ", "ルーデンテンペスト",
    "ソーンメイル", "ワーモグアーマー", "ブラッククリーバー",
    "ガーディアンエンジェル", "ナッシャー トゥース",
    "トリニティフォース",
]

LOL_TACTIC_KEYWORDS = [
    "ガンク", "ローム", "ファーム", "フリーズ", "プッシュ",
    "スプリットプッシュ", "ファストプッシュ", "スロープッシュ",
    "レーンフリーズ", "ウェーブクリア", "ウェーブ",
    "チームファイト", "集団戦", "エンゲージ", "ディスエンゲージ",
    "トレード", "有利トレード", "ダイブ", "スノーボール",
    "フィード", "スローイング", "コール", "チャット",
    "ピン", "マップ確認", "視界", "視界取り",
    "コーチング", "メタ", "構成", "シナジー",
    "カウンター", "パッチ", "降参", "サレンダー",
    "ファイト", "ナイス", "行ける", "行くよ", "任せて",
    "待って", "逃げて", "うまい", "ありがとう", "ごめん",
    "お疲れ様です", "ミスった",
]

LOL_ROLE_KEYWORDS = [
    "トップ", "ミッド", "ジャングル", "ボット", "ADC",
    "サポート", "キャリー", "レーン", "サイドレーン",
    "レーン戦", "レーニングフェーズ",
]

LOL_OBJECTIVE_KEYWORDS = [
    "ドラゴン", "バロン", "ヘラルド", "ネクサス",
    "インヒビター", "タワー", "タワー下", "ミニオン",
    "オブジェクト", "オブジェクト管理", "ブルー", "レッド",
    "ブッシュ", "リスポーン", "リコール",
    "ファーストブラッド", "キル", "ダブルキル", "トリプルキル",
    "クアッドラキル", "ペンタキル", "エース",
    "CS", "キルスコア", "KDA", "ヴィジョンスコア",
    "ゴールド", "経験値", "レベル差", "アイテム差",
]


def classify_valorant(word, reading, meaning):
    """将 Valorant 词条分类到子标签"""
    w = (word or "").strip()
    r = (reading or "").strip()
    m = (meaning or "").strip().lower()
    text = f"{w} {r} {m}"

    tags = []

    # 特工名称精确匹配
    if w in VAL_AGENTS:
        tags.append("gaming_val_agents")

    # 枪械
    for kw in VAL_WEAPON_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_val_weapons")
            break

    # 地图
    for kw in VAL_MAP_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_val_maps")
            break

    # 战术
    for kw in VAL_TACTIC_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_val_tactics")
            break

    # 设置
    for kw in VAL_SETTINGS_KEYWORDS:
        if kw in w or kw in r or kw in m:
            tags.append("gaming_val_settings")
            break

    # 段位
    for kw in VAL_RANK_KEYWORDS:
        if kw in w or kw in r or kw in m:
            tags.append("gaming_val_ranks")
            break

    # Fallback: 如果没匹配到任何子分类，默认放到战术
    if not tags:
        tags.append("gaming_val_tactics")

    return tags


def classify_lol(word, reading, meaning):
    """将 LoL 词条分类到子标签"""
    w = (word or "").strip()
    r = (reading or "").strip()
    m = (meaning or "").strip().lower()
    text = f"{w} {r} {m}"

    tags = []

    for kw in LOL_ITEM_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_lol_items")
            break

    for kw in LOL_OBJECTIVE_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_lol_objectives")
            break

    for kw in LOL_ROLE_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_lol_roles")
            break

    for kw in LOL_CHAMPION_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_lol_champions")
            break

    for kw in LOL_TACTIC_KEYWORDS:
        if kw in w or kw in r:
            tags.append("gaming_lol_tactics")
            break

    if not tags:
        tags.append("gaming_lol_tactics")

    return tags


# ── 新增词汇 ──
NEW_VAL_WORDS = [
    # 特工名称与特工技能 (gaming_val_agents)
    ("ブリーチ", "ブリーチ", "Breach（ initiator agent ），突破型特工，擅长穿墙技能", "gaming_val_agents"),
    ("スカイ", "スカイ", "Skye（ initiator agent ），自然系特工，可治疗和侦查", "gaming_val_agents"),
    ("チェンバー", "チェンバー", "Chamber（ sentinel agent ），法式特工，使用定制武器", "gaming_val_agents"),
    ("フェニックス", "フェニックス", "Phoenix（ duelist agent ），火系决斗者，可自我治疗", "gaming_val_agents"),
    ("ヨル", "ヨル", "Yoru（ duelist agent ），日本特工，擅长传送与迷惑", "gaming_val_agents"),
    ("レイナ", "レイナ", "Reyna（ duelist agent ），吸血型决斗者，高击杀回报", "gaming_val_agents"),
    ("ネオン", "ネオン", "Neon（ duelist agent ），菲律宾特工，电光速度型", "gaming_val_agents"),
    ("ジェット", "ジェット", "Jett（ duelist agent ），韩国特工，风系高机动", "gaming_val_agents"),
    ("セージ", "セージ", "Sage（ sentinel agent ），中国特工，治疗与复活", "gaming_val_agents"),
    ("ソーヴァ", "ソーヴァ", "Sova（ initiator agent ），俄罗斯特工，弓箭侦查", "gaming_val_agents"),
    ("ヴァイパー", "ヴァイパー", "Viper（ controller agent ），毒系控场特工", "gaming_val_agents"),
    ("オーメン", "オーメン", "Omen（ controller agent ），暗影系控场特工", "gaming_val_agents"),
    ("キルジョイ", "キルジョイ", "Killjoy（ sentinel agent ），德国特工，机械防御", "gaming_val_agents"),
    ("サイファー", "サイファー", "Cypher（ sentinel agent ），摩洛哥特工，陷阱侦查", "gaming_val_agents"),
    ("レイズ", "レイズ", "Raze（ duelist agent ），巴西特工，爆破型决斗者", "gaming_val_agents"),
    ("アストラ", "アストラ", "Astra（ controller agent ），加纳特工，宇宙系控场", "gaming_val_agents"),
    ("KAY/O", "ケイオー", "KAY/O（ initiator agent ），机器人特工，技能压制", "gaming_val_agents"),
    ("ハーバー", "ハーバー", "Harbor（ controller agent ），印度特工，水系控场", "gaming_val_agents"),
    ("フェイド", "フェイド", "Fade（ initiator agent ），土耳其特工，恐惧侦查", "gaming_val_agents"),
    ("ゲッコー", "ゲッコー", "Gekko（ initiator agent ），宠物型特工，可回收技能", "gaming_val_agents"),
    ("デッドロック", "デッドロック", "Deadlock（ sentinel agent ），挪威特工，声波陷阱", "gaming_val_agents"),
    ("アイソ", "アイソ", "Iso（ duelist agent ），中国特工，能量护盾", "gaming_val_agents"),
    ("クローヴ", "クローヴ", "Clove（ controller agent ），苏格兰特工，死后仍可放烟", "gaming_val_agents"),
    ("ヴァイス", "ヴァイス", "Vyse（ sentinel agent ），金属控制型特工", "gaming_val_agents"),
    ("アビリティ", "アビリティ", "能力；技能（ability），泛指特工的各种技能", "gaming_val_agents"),
    ("ウルト", "ウルト", "大招；终极技能（ultimate的缩略）", "gaming_val_agents"),
    ("スキル", "スキル", "技能（skill），指游戏中的主动或被动技能", "gaming_val_agents"),
    ("シグネチャー", "シグネチャー", "标志技能；免费技能（signature ability）", "gaming_val_agents"),
    ("リコン", "リコン", "侦查技能；信息收集能力（recon）", "gaming_val_agents"),
    ("ドローン", "ドローン", "无人机；侦查道具（Sova的侦查箭或无人侦查机）", "gaming_val_agents"),
    ("フラッシュ", "フラッシュ", "闪光弹；致盲技能（flash）", "gaming_val_agents"),
    ("スモーク", "スモーク", "烟雾弹；遮蔽视野的技能（smoke）", "gaming_val_agents"),
    ("ヒール", "ヒール", "治疗技能（heal），回复生命值", "gaming_val_agents"),
    ("テレポート", "テレポート", "传送技能（teleport），瞬间移动", "gaming_val_agents"),
    ("トラップ", "トラップ", "陷阱技能（trap），用于侦查或限制敌人移动", "gaming_val_agents"),
    ("ウォール", "ウォール", "墙体技能；屏障（wall），如Sage的冰墙", "gaming_val_agents"),
    ("モロトフ", "モロトフ", "燃烧弹；火焰伤害技能（molotov/incendiary）", "gaming_val_agents"),
    ("グレネード", "グレネード", "手榴弹；范围伤害技能（grenade）", "gaming_val_agents"),
    ("デコイ", "デコイ", "诱饵；假目标技能（decoy），迷惑敌人", "gaming_val_agents"),
    ("ブラインド", "ブラインド", "致盲效果（blind），使敌人视野受限", "gaming_val_agents"),
    ("スロー", "スロー", "减速效果（slow），降低敌人移动速度", "gaming_val_agents"),
    ("スタン", "スタン", "眩晕效果（stun），使敌人无法行动", "gaming_val_agents"),

    # 枪械 (gaming_val_weapons)
    ("クラシック", "クラシック", "Classic，默认手枪，免费获得", "gaming_val_weapons"),
    ("ショーティー", "ショーティー", "Shorty，短管霰弹枪，近距离高伤害", "gaming_val_weapons"),
    ("フレンジー", "フレンジー", "Frenzy，全自动手枪，射速快", "gaming_val_weapons"),
    ("ゴースト", "ゴースト", "Ghost，消音手枪，精准度高", "gaming_val_weapons"),
    ("シェリフ", "シェリフ", "Sheriff，左轮手枪，爆头一枪死", "gaming_val_weapons"),
    ("スティンガー", "スティンガー", "Stinger，廉价冲锋枪，近距离爆发高", "gaming_val_weapons"),
    ("スペクター", "スペクター", "Spectre，消音冲锋枪，中近距离通用", "gaming_val_weapons"),
    ("バッキー", "バッキー", "Bucky，泵动霰弹枪，便宜可靠", "gaming_val_weapons"),
    ("ジャッジ", "ジャッジ", "Judge，全自动霰弹枪，近距离秒杀", "gaming_val_weapons"),
    ("ブルドッグ", "ブルドッグ", "Bulldog，三连发步枪，中距离好用", "gaming_val_weapons"),
    ("ガーディアン", "ガーディアン", "Guardian，半自动步枪，高伤害高精度", "gaming_val_weapons"),
    ("ファントム", "ファントム", "Phantom，消音步枪，弹道稳定适合压枪", "gaming_val_weapons"),
    ("ヴァンダル", "ヴァンダル", "Vandal，主力步枪，爆头一枪死", "gaming_val_weapons"),
    ("マーシャル", "マーシャル", "Marshal，杠杆式狙击枪，打身体高伤", "gaming_val_weapons"),
    ("オペレーター", "オペレーター", "Operator，重型狙击枪，打身体一枪死（AWP）", "gaming_val_weapons"),
    ("ナイフ", "ナイフ", "近战武器；刀（knife），基础近战攻击", "gaming_val_weapons"),
    ("アーマー", "アーマー", "护甲（armor），减少受到的伤害", "gaming_val_weapons"),
    ("ライトアーマー", "ライトアーマー", "轻甲（light shield），400元基础防护", "gaming_val_weapons"),
    ("ヘビーアーマー", "ヘビーアーマー", "重甲（heavy shield），1000元全防护", "gaming_val_weapons"),
    ("フルアーマー", "フルアーマー", "满甲（full armor），完整护甲状态", "gaming_val_weapons"),
    ("リコイル", "リコイル", "后坐力；枪械反冲（recoil）", "gaming_val_weapons"),
    ("スプレー", "スプレー", "扫射；连射时的子弹散布（spray）", "gaming_val_weapons"),
    ("タップ撃ち", "タップうち", "点射；单发射击（tap shooting）", "gaming_val_weapons"),
    ("バースト", "バースト", "连发；爆发射击（burst），如Bulldog的三连发", "gaming_val_weapons"),
    ("ヘッドショット", "ヘッドショット", "爆头（headshot），俗称HS", "gaming_val_weapons"),
    ("ワンタップ", "ワンタップ", "一击爆头（one tap），一发子弹击杀", "gaming_val_weapons"),
    ("エイム", "エイム", "瞄准；准度（aim），射击精度的总称", "gaming_val_weapons"),
    ("クロスヘア", "クロスヘア", "十字准星（crosshair），屏幕中央的瞄准标识", "gaming_val_weapons"),
    ("プリエイム", "プリエイム", "预瞄（pre-aim），提前将准星对准敌人可能位置", "gaming_val_weapons"),
    ("置きエイム", "おきエイム", "架枪；固定预瞄（holding an angle）", "gaming_val_weapons"),
    ("エイム練習", "エイムれんしゅう", "瞄准练习（aim training）", "gaming_val_weapons"),
    ("フリック", "フリック", "甩枪；快速瞄准并射击（flick shot）", "gaming_val_weapons"),

    # 战术术语与对局交流 (gaming_val_tactics)
    ("ラウンド", "ラウンド", "回合（round），一局游戏的基本单位", "gaming_val_tactics"),
    ("エコラウンド", "エコラウンド", "经济局；存钱局（eco round），少买或不买装备", "gaming_val_tactics"),
    ("フォースバイ", "フォースバイ", "强制购买；半起（force buy），钱不够全买但也要买一些", "gaming_val_tactics"),
    ("フルバイ", "フルバイ", "全起（full buy），买满装备武器", "gaming_val_tactics"),
    ("セーブ", "セーブ", "存钱（save），不买装备为下局攒钱", "gaming_val_tactics"),
    ("クレジット", "クレジット", "金钱；经济（credits），游戏内货币", "gaming_val_tactics"),
    ("リテイク", "リテイク", "回防；重夺（retake），夺回已失守的点位", "gaming_val_tactics"),
    ("クラッチ", "クラッチ", "残局翻盘（clutch），少打多的局面", "gaming_val_tactics"),
    ("エントリー", "エントリー", "进点；突破（entry），最先冲入点位的行动", "gaming_val_tactics"),
    ("ローテーション", "ローテーション", "轮转；转点（rotation），从一个点位转移到另一个", "gaming_val_tactics"),
    ("イニシエーター", "イニシエーター", "先锋；突破手（initiator），负责收集信息开路", "gaming_val_tactics"),
    ("デュエリスト", "デュエリスト", "决斗者（duelist），主攻击杀型特工", "gaming_val_tactics"),
    ("コントローラー", "コントローラー", "控场（controller），负责烟雾和区域控制", "gaming_val_tactics"),
    ("センチネル", "センチネル", "守卫（sentinel），防守型特工", "gaming_val_tactics"),
    ("ピーク", "ピーク", "探头；拉枪（peek），从掩体后探出查看", "gaming_val_tactics"),
    ("ワイドピーク", "ワイドピーク", "大拉（wide peek），大幅探出掩体", "gaming_val_tactics"),
    ("ジグルピーク", "ジグルピーク", "小身位晃（jiggle peek），快速露头收头获取信息", "gaming_val_tactics"),
    ("カニ歩き", "カニあるき", "横移；螃蟹步（side strafe），左右移动对枪", "gaming_val_tactics"),
    ("ラーク", "ラーク", "蹲守；埋伏（lurk），在敌人意想不到的位置等待", "gaming_val_tactics"),
    ("フェイク", "フェイク", "假打；佯攻（fake），假装进攻某点实际打另一点", "gaming_val_tactics"),
    ("詰める", "つめる", "前压；推前（push forward），向前推进压缩敌人空间", "gaming_val_tactics"),
    ("攻める", "せめる", "进攻；进攻方行动（attack）", "gaming_val_tactics"),
    ("守って", "まもって", "防守！守点（defend/hold）", "gaming_val_tactics"),
    ("前に出る", "まえにでる", "往前压（advance），向敌方方向推进", "gaming_val_tactics"),
    ("引く", "ひく", "后退；拉开距离（fall back）", "gaming_val_tactics"),
    ("バラけて", "バラけて", "散开！别站一起（spread out）", "gaming_val_tactics"),
    ("固まって", "かたまって", "抱团（group up），集中在一起行动", "gaming_val_tactics"),
    ("カバー", "カバー", "掩护（cover），帮队友看角度或补枪", "gaming_val_tactics"),
    ("トレード", "トレード", "补枪；交换击杀（trade kill）", "gaming_val_tactics"),
    ("味方", "みかた", "队友（ally/teammate），我方人员", "gaming_val_tactics"),
    ("敵", "てき", "敌人（enemy），对方玩家", "gaming_val_tactics"),
    ("情報", "じょうほう", "信息；情报（info），获取敌人位置等信息", "gaming_val_tactics"),
    ("音する", "おとする", "有声音！听到脚步/技能声（audio cue）", "gaming_val_tactics"),
    ("当たってる", "あたってる", "打中了！命中敌人（tagged/hit）", "gaming_val_tactics"),
    ("瀕死", "ひんし", "残血；濒死（low HP），HP极低的状态", "gaming_val_tactics"),
    ("解除", "かいじょ", "拆包；拆除炸弹（defuse）", "gaming_val_tactics"),
    ("設置", "せっち", "下包；安放炸弹（plant the spike）", "gaming_val_tactics"),
    ("即解除", "そくかいじょ", "马上拆！正在拆包（defusing now）", "gaming_val_tactics"),
    ("ディフューズ中", "ディフューズちゅう", "拆包中（defusing），正在解除炸弹", "gaming_val_tactics"),
    ("時間稼いで", "じかんかせいで", "拖时间（stall for time），消耗回合时间", "gaming_val_tactics"),
    ("ホールド", "ホールド", "稳住；架住（hold position），不冒进", "gaming_val_tactics"),
    ("プッシュ", "プッシュ", "推进；前压（push），向前施压", "gaming_val_tactics"),
    ("GG", "ジージー", "好游戏（good game），对局结束时的礼貌用语", "gaming_val_tactics"),
    ("ナイストライ", "ナイストライ", "打得不错（nice try），鼓励队友", "gaming_val_tactics"),

    # 地图与地图报点 (gaming_val_maps)
    ("アセント", "アセント", "Ascent，双点地图，以中路控制为核心", "gaming_val_maps"),
    ("バインド", "バインド", "Bind，双点地图，设有单向传送门", "gaming_val_maps"),
    ("サンセット", "サンセット", "Sunset，双点地图，以中路大道为特色", "gaming_val_maps"),
    ("スプリット", "スプリット", "Split，双点地图，日本风格高度差", "gaming_val_maps"),
    ("ヘイヴン", "ヘイヴン", "Haven，三点地图，唯一有三处下包点的图", "gaming_val_maps"),
    ("ブリーズ", "ブリーズ", "Breeze，大型双点地图，加勒比海滩风", "gaming_val_maps"),
    ("パール", "パール", "Pearl，葡萄牙水下城市风格地图", "gaming_val_maps"),
    ("フラクチャー", "フラクチャー", "Fracture，H型双点地图，防守方被夹击", "gaming_val_maps"),
    ("ロータス", "ロータス", "Lotus，三点地图，印度风格遗迹", "gaming_val_maps"),
    ("アイスボックス", "アイスボックス", "Icebox，双点地图，以垂直战斗为特点", "gaming_val_maps"),
    ("アビス", "アビス", "Abyss，双点地图，北欧深渊主题", "gaming_val_maps"),
    ("Aサイト", "エーサイト", "A点（A bombsite），A下包区域", "gaming_val_maps"),
    ("Bサイト", "ビーサイト", "B点（B bombsite），B下包区域", "gaming_val_maps"),
    ("Cサイト", "シーサイト", "C点（C bombsite），仅Haven/Lotus有", "gaming_val_maps"),
    ("ロング", "ロング", "长道（long），通往点位的长距离路径", "gaming_val_maps"),
    ("ショート", "ショート", "短道（short），通往点位的短路径", "gaming_val_maps"),
    ("ミッド", "ミッド", "中路（mid），地图中央区域", "gaming_val_maps"),
    ("ガレージ", "ガレージ", "车库（garage），Haven的重生点附近区域", "gaming_val_maps"),
    ("エルボー", "エルボー", "肘部通道（elbow），拐角区域", "gaming_val_maps"),
    ("コネクター", "コネクター", "连接通道（connector），连接两个区域的通道", "gaming_val_maps"),
    ("ランプ", "ランプ", "斜坡（ramp），上坡或下坡区域", "gaming_val_maps"),
    ("バックサイト", "バックサイト", "点位后方（back site），下包区的后方", "gaming_val_maps"),
    ("ベント", "ベント", "通风口（vent），狭小的通道", "gaming_val_maps"),
    ("ヘブン", "ヘブン", "天堂位（heaven），高处点内位置", "gaming_val_maps"),
    ("スポーン", "スポーン", "出生点（spawn），每回合出生位置", "gaming_val_maps"),

    # 段位名称 (gaming_val_ranks)
    ("ランク", "ランク", "段位（rank），玩家的竞技等级", "gaming_val_ranks"),
    ("ランクマッチ", "ランクマッチ", "排位赛（ranked match），竞技模式", "gaming_val_ranks"),
    ("コンペティティブ", "コンペティティブ", "竞技模式（competitive），Valorant的排位玩法", "gaming_val_ranks"),
    ("アイアン", "アイアン", "铁牌；Valorant最低段位（Iron）", "gaming_val_ranks"),
    ("ブロンズ", "ブロンズ", "铜牌（Bronze），Valorant第二段位", "gaming_val_ranks"),
    ("シルバー", "シルバー", "银牌（Silver），Valorant第三段位", "gaming_val_ranks"),
    ("ゴールド", "ゴールド", "金牌（Gold），Valorant第四段位", "gaming_val_ranks"),
    ("プラチナ", "プラチナ", "白金（Platinum），Valorant第五段位", "gaming_val_ranks"),
    ("ダイヤモンド", "ダイヤモンド", "钻石（Diamond），Valorant第六段位", "gaming_val_ranks"),
    ("アセンダント", "アセンダント", "超凡（Ascendant），Valorant第七段位", "gaming_val_ranks"),
    ("イモータル", "イモータル", "不朽（Immortal），Valorant第八段位", "gaming_val_ranks"),
    ("レディアント", "レディアント", "辐能（Radiant），Valorant最高段位", "gaming_val_ranks"),
    ("アップランク", "アップランク", "升段（rank up），晋升到更高段位", "gaming_val_ranks"),
    ("ダウンランク", "ダウンランク", "掉段（derank），降级到更低段位", "gaming_val_ranks"),
    ("MMR", "エムエムアール", "隐藏分（Matchmaking Rating），匹配系统评分", "gaming_val_ranks"),
    ("RR", "アールアール", "段位分（Rank Rating），段位内的分数", "gaming_val_ranks"),
    ("ダブルランクアップ", "ダブルランクアップ", "跳段（double rank up），一次升两个段位", "gaming_val_ranks"),
    ("プレイスメント", "プレイスメント", "定级赛（placement matches），确定初始段位", "gaming_val_ranks"),
    ("スマーフ", "スマーフ", "小号；炸鱼（smurf），高段位玩家用低段位号", "gaming_val_ranks"),
    ("ブースト", "ブースト", "带上分（boost），高手带低段位上分", "gaming_val_ranks"),

    # 游戏设置 (gaming_val_settings)
    ("感度", "かんど", "灵敏度（sensitivity），鼠标或瞄准灵敏度", "gaming_val_settings"),
    ("DPI", "ディーピーアイ", "鼠标DPI（dots per inch），鼠标硬件灵敏度", "gaming_val_settings"),
    ("eDPI", "イーディーピーアイ", "有效DPI（effective DPI），DPI×游戏内感度", "gaming_val_settings"),
    ("解像度", "かいぞうど", "分辨率（resolution），画面分辨率设置", "gaming_val_settings"),
    ("フレームレート", "フレームレート", "帧率（FPS/framerate），每秒画面更新次数", "gaming_val_settings"),
    ("FPS", "エフピーエス", "帧数；每秒帧数（frames per second）", "gaming_val_settings"),
    ("グラフィック設定", "グラフィックせってい", "画质设置（graphics settings）", "gaming_val_settings"),
    ("キーバインド", "キーバインド", "按键绑定（key bind），自定义按键设置", "gaming_val_settings"),
    ("マウスパッド", "マウスパッド", "鼠标垫（mousepad）", "gaming_val_settings"),
    ("リフレッシュレート", "リフレッシュレート", "刷新率（refresh rate），显示器刷新频率", "gaming_val_settings"),
    ("144Hz", "ひゃくよんじゅうよんヘルツ", "144Hz刷新率，电竞显示器标配", "gaming_val_settings"),
    ("240Hz", "にひゃくよんじゅうヘルツ", "240Hz刷新率，高刷电竞显示器", "gaming_val_settings"),
    ("クロスヘア設定", "クロスヘアせってい", "准星设置（crosshair settings），自定义准星外观", "gaming_val_settings"),
    ("ミニマップ", "ミニマップ", "小地图（minimap），显示队友和敌人信息", "gaming_val_settings"),
    ("画質", "がしつ", "画质（graphics quality），游戏画面质量", "gaming_val_settings"),
    ("垂直同期", "すいちょくどうき", "垂直同步（V-Sync），防画面撕裂但增加延迟", "gaming_val_settings"),
]

NEW_LOL_WORDS = [
    # 英雄名称与技能 (gaming_lol_champions)
    ("チャンピオン", "チャンピオン", "英雄（champion），玩家操控的角色", "gaming_lol_champions"),
    ("ピック", "ピック", "选择英雄（pick），在对局中选定某个英雄", "gaming_lol_champions"),
    ("バン", "バン", "禁用英雄（ban），禁止对局中出现某英雄", "gaming_lol_champions"),
    ("スキルショット", "スキルショット", "指向性技能（skillshot），需手动瞄准的技能", "gaming_lol_champions"),
    ("ポイントクリック", "ポイントクリック", "点选技能（point-and-click），点击目标即可释放", "gaming_lol_champions"),
    ("パッシブ", "パッシブ", "被动技能（passive），自动生效的能力", "gaming_lol_champions"),
    ("Qスキル", "キュースキル", "Q技能（first ability），第一个主动技能", "gaming_lol_champions"),
    ("Wスキル", "ダブリュースキル", "W技能（second ability），第二个主动技能", "gaming_lol_champions"),
    ("Eスキル", "イースキル", "E技能（third ability），第三个主动技能", "gaming_lol_champions"),
    ("Rスキル", "アールスキル", "R技能；大招（ultimate），终极技能", "gaming_lol_champions"),
    ("ウルト", "ウルト", "大招（ult），终极技能的简称", "gaming_lol_champions"),
    ("クールダウン", "クールダウン", "冷却时间（cooldown），技能再次可用的等待时间", "gaming_lol_champions"),
    ("CD", "シーディー", "冷却简称（cooldown缩写）", "gaming_lol_champions"),
    ("マナ", "マナ", "法力值（mana），释放技能消耗的资源", "gaming_lol_champions"),
    ("スタック", "スタック", "层数；叠加（stack），累积的技能或物品效果", "gaming_lol_champions"),
    ("レベルアップ", "レベルアップ", "升级（level up），英雄等级提升", "gaming_lol_champions"),
    ("スキン", "スキン", "皮肤（skin），英雄的外观装扮", "gaming_lol_champions"),
    ("ダッシュ", "ダッシュ", "突进技能（dash），快速位移", "gaming_lol_champions"),
    ("ブリンク", "ブリンク", "闪现类技能（blink），瞬间位移", "gaming_lol_champions"),
    ("CC", "シーシー", "控制技能（crowd control），限制敌人行动的技能", "gaming_lol_champions"),
    ("ノックアップ", "ノックアップ", "击飞（knock-up），将敌人打飞的控制效果", "gaming_lol_champions"),
    ("ノックバック", "ノックバック", "击退（knock-back），将敌人推开的控制效果", "gaming_lol_champions"),
    ("スタン", "スタン", "眩晕（stun），使目标无法行动", "gaming_lol_champions"),
    ("スネア", "スネア", "缠绕（snare/root），使目标无法移动但可攻击", "gaming_lol_champions"),
    ("サプレス", "サプレス", "压制（suppression），最强的控制效果", "gaming_lol_champions"),
    ("チャーム", "チャーム", "魅惑（charm），使目标走向施法者", "gaming_lol_champions"),
    ("フィアー", "フィアー", "恐惧（fear），使目标随机移动", "gaming_lol_champions"),
    ("タウント", "タウント", "嘲讽（taunt），强制目标攻击施法者", "gaming_lol_champions"),
    ("スロー", "スロー", "减速（slow），降低目标移动速度", "gaming_lol_champions"),
    ("シールド", "シールド", "护盾（shield），吸收伤害的临时HP", "gaming_lol_champions"),
    ("回復スキル", "かいふくスキル", "回复技能（heal），恢复生命值的技能", "gaming_lol_champions"),
    ("テレポート", "テレポート", "传送技能（teleport），召唤师技能之一", "gaming_lol_champions"),

    # 装备 (gaming_lol_items)
    ("コアアイテム", "コアアイテム", "核心装备（core item），英雄最关键的装备", "gaming_lol_items"),
    ("スタートアイテム", "スタートアイテム", "出门装（starting item），开局购买的装备", "gaming_lol_items"),
    ("ミシックアイテム", "ミシックアイテム", "神话装备（mythic item），现已移除的装备等级", "gaming_lol_items"),
    ("レジェンダリー", "レジェンダリー", "传说装备（legendary item），高级装备", "gaming_lol_items"),
    ("ブーツ", "ブーツ", "鞋子（boots），提供移动速度的基础装备", "gaming_lol_items"),
    ("ポーション", "ポーション", "药水（potion），回复生命值的消耗品", "gaming_lol_items"),
    ("トリニティフォース", "トリニティフォース", "三项之力（Trinity Force），战士核心装备", "gaming_lol_items"),
    ("インフィニティエッジ", "インフィニティエッジ", "无尽之刃（Infinity Edge），暴击核心装备", "gaming_lol_items"),
    ("ラバドンデスキャップ", "ラバドンデスキャップ", "灭世者的死亡之帽（Rabadon's Deathcap），AP核心装", "gaming_lol_items"),
    ("ゾーニャの砂時計", "ゾーニャのすなどけい", "中亚沙漏（Zhonya's Hourglass），金身装备", "gaming_lol_items"),
    ("ヴォイドスタッフ", "ヴォイドスタッフ", "虚空之杖（Void Staff），法穿装备", "gaming_lol_items"),
    ("モレロノミコン", "モレロノミコン", "莫雷洛秘典（Morellonomicon），重伤装备", "gaming_lol_items"),
    ("ブラッドサースター", "ブラッドサースター", "饮血剑（Bloodthirster），吸血装备", "gaming_lol_items"),
    ("ガーディアンエンジェル", "ガーディアンエンジェル", "守护天使（Guardian Angel），复活甲", "gaming_lol_items"),
    ("ブラッククリーバー", "ブラッククリーバー", "黑色切割者（Black Cleaver），破甲装备", "gaming_lol_items"),
    ("ステラックの篭手", "ステラックのこて", "斯特拉克的挑战护手（Sterak's Gage），战士保命装", "gaming_lol_items"),
    ("サンファイアイギス", "サンファイアイギス", "日炎圣盾（Sunfire Aegis），坦克装备", "gaming_lol_items"),
    ("ソーンメイル", "ソーンメイル", "荆棘之甲（Thornmail），反伤装备", "gaming_lol_items"),
    ("ワーモグアーマー", "ワーモグアーマー", "狂徒铠甲（Warmog's Armor），回血装备", "gaming_lol_items"),
    ("リッチベイン", "リッチベイン", "巫妖之祸（Lich Bane），技能后强化普攻的装备", "gaming_lol_items"),
    ("ルーデンテンペスト", "ルーデンテンペスト", "卢登的激荡（Luden's Tempest），AP poke装", "gaming_lol_items"),
    ("ナッシャートゥース", "ナッシャートゥース", "纳什之牙（Nashor's Tooth），攻速+AP装备", "gaming_lol_items"),
    ("ドランブレード", "ドランブレード", "多兰之刃（Doran's Blade），AD出门装", "gaming_lol_items"),
    ("ドランリング", "ドランリング", "多兰之戒（Doran's Ring），AP出门装", "gaming_lol_items"),
    ("ドランシールド", "ドランシールド", "多兰之盾（Doran's Shield），坦克出门装", "gaming_lol_items"),
    ("トリンケット", "トリンケット", "饰品（trinket），免费的视野装备", "gaming_lol_items"),
    ("ワード", "ワード", "眼（ward），提供视野的消耗品", "gaming_lol_items"),
    ("ピンクワード", "ピンクワード", "真眼（pink ward/control ward），可看见隐形单位", "gaming_lol_items"),
    ("デワード", "デワード", "排眼（de-ward），清除敌方眼位", "gaming_lol_items"),
    ("アイテム差", "アイテムさ", "装备差距（item gap），双方装备价值的差值", "gaming_lol_items"),
    ("ゴールド", "ゴールド", "金币（gold），游戏内购买装备的货币", "gaming_lol_items"),

    # 位置与职责 (gaming_lol_roles)
    ("トップ", "トップ", "上路（top lane），地图上方的单人对线位置", "gaming_lol_roles"),
    ("ミッド", "ミッド", "中路（mid lane），地图中央的单人对线位置", "gaming_lol_roles"),
    ("ジャングル", "ジャングル", "打野（jungle），在野区游走发育的位置", "gaming_lol_roles"),
    ("ボット", "ボット", "下路（bot lane），双人线位置所在的路线", "gaming_lol_roles"),
    ("ADC", "エーディーシー", "AD射手（AD Carry），下路物理输出核心", "gaming_lol_roles"),
    ("サポート", "サポート", "辅助（support），保护队友和做视野的角色", "gaming_lol_roles"),
    ("レーナー", "レーナー", "对线者（laner），泛指在线上的玩家", "gaming_lol_roles"),
    ("サイドレーン", "サイドレーン", "边路（side lane），上下路的统称", "gaming_lol_roles"),
    ("ファイター", "ファイター", "战士（fighter），近战输出型英雄", "gaming_lol_roles"),
    ("タンク", "タンク", "坦克（tank），肉盾型英雄，吸收伤害", "gaming_lol_roles"),
    ("アサシン", "アサシン", "刺客（assassin），高爆发秒脆皮英雄", "gaming_lol_roles"),
    ("メイジ", "メイジ", "法师（mage），魔法输出型英雄", "gaming_lol_roles"),
    ("マークスマン", "マークスマン", "射手（marksman），远程物理输出英雄", "gaming_lol_roles"),
    ("キャリー", "キャリー", "核心输出（carry），队伍的主要伤害来源", "gaming_lol_roles"),
    ("オートフィル", "オートフィル", "自动补位（autofill），系统自动分配位置", "gaming_lol_roles"),
    ("ローム", "ローム", "游走（roam），离开自己路线支援其他路", "gaming_lol_roles"),

    # 地图资源与目标 (gaming_lol_objectives)
    ("ネクサス", "ネクサス", "主基地（nexus），摧毁即可获胜", "gaming_lol_objectives"),
    ("インヒビター", "インヒビター", "抑制器（inhibitor），摧毁后对方出超级兵", "gaming_lol_objectives"),
    ("タワー", "タワー", "防御塔（tower），自动攻击敌人的建筑", "gaming_lol_objectives"),
    ("タワー下", "タワーした", "塔下（under tower），在防御塔保护范围内", "gaming_lol_objectives"),
    ("ミニオン", "ミニオン", "小兵（minion），自动推进的单位", "gaming_lol_objectives"),
    ("CS", "シーエス", "补刀数（creep score），击杀小兵的总数", "gaming_lol_objectives"),
    ("ドラゴン", "ドラゴン", "小龙（dragon），击杀后提供团队buff", "gaming_lol_objectives"),
    ("バロン", "バロン", "大龙；纳什男爵（Baron Nashor），击杀后提供强力buff", "gaming_lol_objectives"),
    ("ヘラルド", "ヘラルド", "峡谷先锋（Rift Herald），击杀后可召唤推塔", "gaming_lol_objectives"),
    ("ブルー", "ブルー", "蓝buff（blue buff），提供法力回复和技能急速", "gaming_lol_objectives"),
    ("レッド", "レッド", "红buff（red buff），提供生命回复和普攻减速", "gaming_lol_objectives"),
    ("ブッシュ", "ブッシュ", "草丛（bush），提供隐身视野的植被", "gaming_lol_objectives"),
    ("視界", "しかい", "视野（vision），对小地图区域的可见性", "gaming_lol_objectives"),
    ("視界取り", "しがいどり", "控视野（warding/vision control），通过放眼控制地图信息", "gaming_lol_objectives"),
    ("オブジェクト", "オブジェクト", "地图目标（objective），如龙、塔等战略点", "gaming_lol_objectives"),
    ("オブジェクト管理", "オブジェクトかんり", "资源控制（objective control），管理地图资源", "gaming_lol_objectives"),
    ("ファーストブラッド", "ファーストブラッド", "一血（first blood），全场第一个击杀", "gaming_lol_objectives"),
    ("リスポーン", "リスポーン", "重生（respawn），死亡后重新出现在基地", "gaming_lol_objectives"),
    ("リコール", "リコール", "回城（recall），传送回基地", "gaming_lol_objectives"),
    ("ピン", "ピン", "信号（ping），在小地图上标记位置", "gaming_lol_objectives"),
    ("マップ", "マップ", "地图（map），可指小地图或整张地图", "gaming_lol_objectives"),
    ("マップ確認", "マップかくにん", "看小地图（checking minimap），关注地图动态", "gaming_lol_objectives"),

    # 战术与对局交流 (gaming_lol_tactics)
    ("レーン戦", "レーンせん", "对线期（laning phase），游戏前期的对线阶段", "gaming_lol_tactics"),
    ("レーニングフェーズ", "レーニングフェーズ", "对线阶段（laning phase），同上", "gaming_lol_tactics"),
    ("ゲーム終盤", "ゲームしゅうばん", "后期（late game），游戏后期阶段", "gaming_lol_tactics"),
    ("チームファイト", "チームファイト", "团战（teamfight），多人参与的大规模战斗", "gaming_lol_tactics"),
    ("集団戦", "しゅうだんせん", "团战（日文说法）", "gaming_lol_tactics"),
    ("ガンク", "ガンク", "抓人（gank），打野或游走英雄配合队友击杀", "gaming_lol_tactics"),
    ("ファーム", "ファーム", "发育；刷钱（farm），通过击杀小兵积累经济", "gaming_lol_tactics"),
    ("フリーズ", "フリーズ", "控线（freeze），保持兵线位置不变", "gaming_lol_tactics"),
    ("プッシュ", "プッシュ", "推线（push），快速清除小兵推向前方", "gaming_lol_tactics"),
    ("スプリットプッシュ", "スプリットプッシュ", "分带（split push），多人分别带不同路线", "gaming_lol_tactics"),
    ("ファストプッシュ", "ファストプッシュ", "速推（fast push），快速清线推塔", "gaming_lol_tactics"),
    ("スロープッシュ", "スロープッシュ", "慢推（slow push），缓慢积累大波兵线", "gaming_lol_tactics"),
    ("レーンフリーズ", "レーンフリーズ", "控线（lane freeze），冻结兵线位置", "gaming_lol_tactics"),
    ("ウェーブクリア", "ウェーブクリア", "清线（wave clear），清除一波小兵", "gaming_lol_tactics"),
    ("ダイブ", "ダイブ", "越塔强杀（dive），在对方塔下发起攻击", "gaming_lol_tactics"),
    ("エンゲージ", "エンゲージ", "开团（engage），主动发起团战", "gaming_lol_tactics"),
    ("ディスエンゲージ", "ディスエンゲージ", "撤退脱离（disengage），脱离战斗", "gaming_lol_tactics"),
    ("キル", "キル", "击杀（kill），消灭敌方英雄", "gaming_lol_tactics"),
    ("ダブルキル", "ダブルキル", "双杀（double kill）", "gaming_lol_tactics"),
    ("トリプルキル", "トリプルキル", "三杀（triple kill）", "gaming_lol_tactics"),
    ("クアッドラキル", "クアッドラキル", "四杀（quadra kill）", "gaming_lol_tactics"),
    ("ペンタキル", "ペンタキル", "五杀（penta kill）", "gaming_lol_tactics"),
    ("エース", "エース", "团灭（ace），消灭对方全部五人", "gaming_lol_tactics"),
    ("アシスト", "アシスト", "助攻（assist），参与击杀但非最后一击", "gaming_lol_tactics"),
    ("デス", "デス", "死亡（death），英雄被击杀", "gaming_lol_tactics"),
    ("KDA", "ケーディーエー", "击杀/死亡/助攻比（K/D/A ratio）", "gaming_lol_tactics"),
    ("スノーボール", "スノーボール", "雪球效应（snowball），优势不断积累扩大", "gaming_lol_tactics"),
    ("フィード", "フィード", "送人头（feed），不断被杀让对面发育", "gaming_lol_tactics"),
    ("スローイング", "スローイング", "送优势（throwing），优势局被翻盘", "gaming_lol_tactics"),
    ("メタ", "メタ", "版本打法（meta），当前版本最强的策略和英雄", "gaming_lol_tactics"),
    ("構成", "こうせい", "阵容（team composition），队伍的攻守结构", "gaming_lol_tactics"),
    ("シナジー", "シナジー", "协同效应（synergy），英雄之间的配合效果", "gaming_lol_tactics"),
    ("パッチ", "パッチ", "版本更新（patch），游戏平衡性调整", "gaming_lol_tactics"),
    ("降参", "こうさん", "投降（surrender），同意提前结束对局", "gaming_lol_tactics"),
    ("サレンダー", "サレンダー", "认输（surrender），同投降", "gaming_lol_tactics"),
    ("OP", "オーピー", "过强（overpowered），英雄或物品过于强力", "gaming_lol_tactics"),
    ("ナーフ", "ナーフ", "削弱（nerf），降低英雄或装备强度", "gaming_lol_tactics"),
    ("バフ", "バフ", "增强（buff），提高英雄或装备强度", "gaming_lol_tactics"),
]

# ── 去重（同一个词可能出现在多个来源）──
def dedup_words(words):
    seen = set()
    result = []
    for w, r, m, t in words:
        key = (w.strip(), t)
        if key not in seen:
            seen.add(key)
            result.append((w, r, m, t))
    return result


def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行")
    args = parser.parse_args()

    conn = get_conn()
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, word, reading, meaning, tags
                FROM vocab_library
                WHERE tags && ARRAY['gaming_valorant','gaming_lol']::text[]
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    log.info(f"找到 {len(rows)} 个游戏词条")

    # Step 1: 分类现有词条
    reclassify_updates = []  # (id, new_tags[])
    val_count = 0
    lol_count = 0
    val_sub_stats = {}
    lol_sub_stats = {}

    for row_id, word, reading, meaning, tags in rows:
        existing_tags = list(tags) if tags else []
        new_sub_tags = []

        if "gaming_valorant" in existing_tags:
            val_count += 1
            new_sub_tags = classify_valorant(word, reading, meaning)
            for t in new_sub_tags:
                val_sub_stats[t] = val_sub_stats.get(t, 0) + 1
            # 保留原 gaming_valorant 标签 + 新子分类
            final_tags = existing_tags + [t for t in new_sub_tags if t not in existing_tags]

        elif "gaming_lol" in existing_tags:
            lol_count += 1
            new_sub_tags = classify_lol(word, reading, meaning)
            for t in new_sub_tags:
                lol_sub_stats[t] = lol_sub_stats.get(t, 0) + 1
            final_tags = existing_tags + [t for t in new_sub_tags if t not in existing_tags]

        else:
            continue

        if any(t not in existing_tags for t in new_sub_tags):
            reclassify_updates.append((row_id, final_tags))

    log.info(f"  Valorant: {val_count} 词, 分布: {val_sub_stats}")
    log.info(f"  LoL: {lol_count} 词, 分布: {lol_sub_stats}")
    log.info(f"  需要更新标签: {len(reclassify_updates)} 个")

    # Step 2: 新增词条
    new_val_words = dedup_words(NEW_VAL_WORDS)
    new_lol_words = dedup_words(NEW_LOL_WORDS)

    log.info(f"  新增 Valorant 词: {len(new_val_words)}")
    log.info(f"  新增 LoL 词: {len(new_lol_words)}")

    if args.dry_run:
        log.info("\n=== DRY RUN === 未实际修改")
        return

    # Step 3: 批量更新现有词条
    if reclassify_updates:
        conn = get_conn()
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '120s'")
            updated = 0
            for i in range(0, len(reclassify_updates), 100):
                batch = reclassify_updates[i:i + 100]
                with conn.cursor() as cur:
                    for row_id, new_tags in batch:
                        cur.execute(
                            "UPDATE vocab_library SET tags = %s WHERE id = %s",
                            (new_tags, row_id)
                        )
                updated += len(batch)
                if i % 500 == 0 and i > 0:
                    log.info(f"  标签更新进度: {updated}/{len(reclassify_updates)}")
            log.info(f"  标签更新完成: {updated} 个")
        finally:
            conn.close()

    # Step 4: 插入新词条
    all_new = new_val_words + new_lol_words
    if all_new:
        conn = get_conn()
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '120s'")
            inserted = 0
            skipped = 0
            for i in range(0, len(all_new), 100):
                batch = all_new[i:i + 100]
                with conn.cursor() as cur:
                    for word, reading, meaning, tag in batch:
                        # 检查是否已存在（同词+同标签）
                        cur.execute(
                            "SELECT id FROM vocab_library WHERE word = %s AND tags @> ARRAY[%s]::text[] LIMIT 1",
                            (word, tag)
                        )
                        if cur.fetchone():
                            skipped += 1
                            continue
                        new_id = str(uuid.uuid4())
                        cur.execute(
                            """INSERT INTO vocab_library (id, word, reading, meaning, tags, level, pos, frequency)
                               VALUES (%s, %s, %s, %s, ARRAY[%s]::text[], '游戏', '名词', 3)""",
                            (new_id, word, reading, meaning, tag)
                        )
                        inserted += 1
                if i % 500 == 0 and i > 0:
                    log.info(f"  新词插入进度: {inserted + skipped}/{len(all_new)}")
            log.info(f"  新词插入完成: {inserted} 个（跳过重复: {skipped} 个）")
        finally:
            conn.close()

    log.info(f"\n全部完成！")


if __name__ == "__main__":
    main()
