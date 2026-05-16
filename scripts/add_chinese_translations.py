#!/usr/bin/env python3
"""
为所有游戏词条添加中文游戏译名，并清理重复/冗余。

改动:
1. 每个游戏词条的 meaning 改为: {中文游戏译名} | {原日语释义}
2. 删除重复词条（同一词出现多次的，合并为一条）
3. 删除"未分级"和"一般"标签
4. 补充游戏中常用的俚语/外号

Usage:
  python3 scripts/add_chinese_translations.py --dry-run
  python3 scripts/add_chinese_translations.py
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
log = logging.getLogger("add_cn")

# ── Valorant 特工: Japanese → (Chinese official name, nicknames/slang, original JP description) ──
VAL_AGENT_CN = {
    "ジェット": ("捷风", "Jett，风系高机动决斗者，外号「风女」"),
    "セージ": ("贤者", "Sage，中国哨位，可治疗复活，外号「奶妈」"),
    "ソーヴァ": ("猎枭", "Sova，俄罗斯先锋，弓箭侦查，外号「无人机」「老鹰」"),
    "ヴァイパー": ("蝰蛇", "Viper，毒系控场，外号「毒女」「蛇女」"),
    "オーメン": ("黑梦", "Omen，暗影控场，外号「鬼」「暗影」"),
    "キルジョイ": ("奇乐", "Killjoy，德国哨位，机械防御，外号「炮台妹」"),
    "サイファー": ("零", "Cypher，摩洛哥哨位，陷阱侦查，外号「摄像头」「老六」"),
    "レイズ": ("雷兹", "Raze，巴西决斗者，爆破专家，外号「炸弹妹」"),
    "ブリーチ": ("铁臂", "Breach，瑞典先锋，穿墙技能，外号「大壮」"),
    "ブリムストーン": ("炼狱", "Brimstone，美国控场，卫星轨道打击，外号「烟男」"),
    "フェニックス": ("不死鸟", "Phoenix，英国决斗者，火焰自愈，外号「火男」"),
    "レイナ": ("芮娜", "Reyna，墨西哥决斗者，吸血型，外号「吸血女」"),
    "ヨル": ("夜露", "Yoru，日本决斗者，传送迷惑，外号「日本仔」"),
    "アストラ": ("星礈", "Astra，加纳控场，宇宙系，外号「星女」"),
    "KAY/O": ("凯欧", "KAY/O，机器人先锋，技能压制，外号「机器人」"),
    "チェンバー": ("尚博勒", "Chamber，法国哨位，定制武器，外号「法国人」「狙击手」"),
    "ネオン": ("霓虹", "Neon，菲律宾决斗者，电光速度，外号「电女」"),
    "フェイド": ("黑市", "Fade，土耳其先锋，恐惧侦查，外号「眼魔」"),
    "ハーバー": ("港湾", "Harbor，印度控场，水系，外号「水男」"),
    "ゲッコー": ("盖可", "Gekko，宠物型先锋，可回收技能，外号「宠物男」"),
    "デッドロック": ("铁臂", "Deadlock，挪威哨位，声波陷阱，外号「网女」"),
    "アイソ": ("壹决", "Iso，中国决斗者，能量护盾，外号「盾男」"),
    "クローヴ": ("丁香", "Clove，苏格兰控场，死后仍可放烟，外号「不死女」"),
    "ヴァイス": ("维斯", "Vyse，金属控制型哨位"),
    "スカイ": ("斯凯", "Skye，澳大利亚先锋，自然系治疗侦查，外号「鸟女」"),
}

# ── Valorant 武器 ──
VAL_WEAPON_CN = {
    "クラシック": ("标配", "Classic，默认手枪，免费，外号「小手枪」"),
    "ショーティー": ("短管", "Shorty，短霰弹枪，近距离秒杀，外号「喷子」"),
    "フレンジー": ("狂怒", "Frenzy，全自动手枪，外号「连发手枪」"),
    "ゴースト": ("鬼魅", "Ghost，消音手枪，精准度高，外号「消音手枪」"),
    "シェリフ": ("警长", "Sheriff，左轮手枪，爆头一枪死，外号「沙鹰」"),
    "スティンガー": ("毒刺", "Stinger，廉价冲锋枪，近距爆发，外号「小冲锋」"),
    "スペクター": ("幻影", "Spectre，消音冲锋枪，中近通用，外号「消音冲锋」"),
    "バッキー": ("雄鹿", "Bucky，泵动霰弹枪，便宜可靠，外号「单喷」"),
    "ジャッジ": ("判官", "Judge，全自动霰弹枪，近距离秒杀，外号「连喷」"),
    "ブルドッグ": ("斗牛犬", "Bulldog，三连发步枪，中距离，外号「三连发」"),
    "ガーディアン": ("守护", "Guardian，半自动步枪，高伤害高精度，外号「半自动」"),
    "ファントム": ("幻象", "Phantom，消音步枪，弹道稳定，外号「M4」"),
    "ヴァンダル": ("暴徒", "Vandal，主力步枪，爆头一枪死，外号「AK」"),
    "マーシャル": ("警长", "Marshal，杠杆狙击枪，外号「鸟狙」"),
    "オペレーター": ("冥驹", "Operator，重型狙击枪，打身体一枪死，外号「大狙」「OP」"),
    "ナイフ": ("近战武器", "Knife，基础近战攻击，外号「刀」"),
}

VAL_ARMOR_CN = {
    "アーマー": ("护甲", "Armor，减少受到的伤害"),
    "ライトアーマー": ("轻甲", "Light Shield，400元的轻甲"),
    "ヘビーアーマー": ("重甲", "Heavy Shield，1000元的全甲"),
    "フルアーマー": ("满甲", "Full Armor，完整的护甲状态"),
}

VAL_WEAPON_TERMS_CN = {
    "リコイル": ("后坐力", "Recoil，枪械反冲力"),
    "スプレー": ("扫射", "Spray，连射时的子弹散布，俗称「泼水」"),
    "タップ撃ち": ("点射", "Tap Shooting，单发射击"),
    "バースト": ("连发", "Burst，爆发射击，如Bulldog的三连发"),
    "ヘッドショット": ("爆头", "Headshot，俗称HS，一击致命"),
    "HS": ("爆头", "Headshot的缩写"),
    "ワンタップ": ("一枪头", "One Tap，一发爆头击杀"),
    "エイム": ("瞄准", "Aim，射击精度，马枪的反义词"),
    "クロスヘア": ("准星", "Crosshair，屏幕中央的瞄准标识"),
    "プリエイム": ("预瞄", "Pre-aim，提前将准星对准敌人可能出现的位置"),
    "置きエイム": ("架枪", "Holding an Angle，固定预瞄某个位置"),
    "エイム練習": ("练枪", "Aim Training，瞄准练习"),
    "フリック": ("甩枪", "Flick Shot，快速甩动鼠标瞄准射击"),
    "ワンショット": ("一枪死", "One-Shot，一发击杀"),
}

# ── Valorant 地图 ──
VAL_MAP_CN = {
    "アセント": ("亚海悬城", "Ascent，双点地图，以中路控制为核心，俗称「源工」"),
    "バインド": ("源工重镇", "Bind，双点地图，设有单向传送门，俗称「绑定」"),
    "サンセット": ("日落之城", "Sunset，双点地图，以中路大道为特色"),
    "スプリット": ("霓虹町", "Split，双点地图，日本风格，高度差明显"),
    "ヘイヴン": ("隐世修所", "Haven，三点地图，唯一有三处下包点"),
    "ブリーズ": ("微风岛屿", "Breeze，大型双点地图，加勒比海滩风"),
    "パール": ("深海明珠", "Pearl，葡萄牙水下城市风格地图"),
    "フラクチャー": ("裂变峡谷", "Fracture，H型双点，防守方被夹击"),
    "ロータス": ("莲华古城", "Lotus，三点地图，印度风格遗迹"),
    "アイスボックス": ("冰库", "Icebox，双点地图，以垂直战斗为特点"),
    "アビス": ("深渊", "Abyss，双点地图，北欧深渊主题"),
}

VAL_MAP_CALLOUT_CN = {
    "Aサイト": ("A点", "A Bombsite，A下包区域"),
    "Bサイト": ("B点", "B Bombsite，B下包区域"),
    "Cサイト": ("C点", "C Bombsite，仅Haven/Lotus有"),
    "ロング": ("长道", "Long，通往点位的长距离路径"),
    "ショート": ("短道", "Short，通往点位的短路径"),
    "ミッド": ("中路", "Mid，地图中央区域"),
    "ガレージ": ("车库", "Garage，Haven的C点附近区域"),
    "エルボー": ("拐角", "Elbow，L型拐角通道"),
    "コネクター": ("连接", "Connector，两个区域的连接通道"),
    "ランプ": ("斜坡", "Ramp，上坡或下坡区域"),
    "バックサイト": ("点位后方", "Back Site，下包区后方"),
    "ベント": ("通风口", "Vent，狭小通道"),
    "ヘブン": ("天堂", "Heaven，高处点内位置，俗称「高台」"),
    "スポーン": ("出生点", "Spawn，每回合出生的位置"),
}

# ── Valorant 段位 ──
VAL_RANK_CN = {
    "アイアン": ("黑铁", "Iron，最低段位"),
    "ブロンズ": ("青铜", "Bronze，第二段位"),
    "シルバー": ("白银", "Silver，第三段位"),
    "ゴールド": ("黄金", "Gold，第四段位"),
    "プラチナ": ("铂金", "Platinum，第五段位"),
    "ダイヤモンド": ("钻石", "Diamond，第六段位"),
    "アセンダント": ("超凡", "Ascendant，第七段位"),
    "イモータル": ("不朽", "Immortal，第八段位"),
    "レディアント": ("辐能", "Radiant，最高段位，俗称「王者」"),
    "ランク": ("段位", "Rank，玩家的竞技等级"),
    "ランクマッチ": ("排位赛", "Ranked Match，竞技模式，俗称「排位」"),
    "コンペティティブ": ("竞技模式", "Competitive，Valorant的排位玩法"),
    "アップランク": ("升段", "Rank Up，晋升到更高段位，俗称「上分」"),
    "ダウンランク": ("掉段", "Derank，降级，俗称「掉分」"),
    "MMR": ("隐藏分", "MMR (Matchmaking Rating)，匹配系统评分"),
    "RR": ("段位分", "RR (Rank Rating)，段位内的分数"),
    "ダブルランクアップ": ("跳段", "Double Rank Up，一次升两个段位"),
    "プレイスメント": ("定级赛", "Placement Matches，确定初始段位的比赛"),
    "スマーフ": ("炸鱼", "Smurf，高段位玩家用低段位号虐菜"),
    "ブースト": ("带上分", "Boost，高段位带低段位上分，俗称「代练」"),
}

# ── Valorant 游戏设置 ──
VAL_SETTINGS_CN = {
    "感度": ("灵敏度", "Sensitivity，鼠标或游戏内灵敏度，俗称「灵敏度」"),
    "DPI": ("鼠标DPI", "DPI (Dots Per Inch)，鼠标硬件灵敏度"),
    "eDPI": ("有效DPI", "eDPI (Effective DPI)，DPI×游戏内灵敏度"),
    "解像度": ("分辨率", "Resolution，画面分辨率"),
    "フレームレート": ("帧率", "Framerate / FPS，每秒画面更新次数"),
    "FPS": ("帧数", "FPS (Frames Per Second)，每秒帧数"),
    "グラフィック設定": ("画质设置", "Graphics Settings"),
    "キーバインド": ("按键绑定", "Key Bind，自定义按键设置"),
    "マウスパッド": ("鼠标垫", "Mousepad"),
    "リフレッシュレート": ("刷新率", "Refresh Rate，显示器刷新频率，如144Hz/240Hz"),
    "144Hz": ("144Hz", "144Hz刷新率，电竞显示器标配"),
    "240Hz": ("240Hz", "240Hz刷新率，高刷电竞显示器"),
    "クロスヘア設定": ("准星设置", "Crosshair Settings，自定义准星外观"),
    "ミニマップ": ("小地图", "Minimap"),
    "画質": ("画质", "Graphics Quality"),
    "垂直同期": ("垂直同步", "V-Sync，防画面撕裂但增加延迟"),
}

# ── Valorant 战术/能力术语 ──
VAL_TACTICS_CN = {
    "ラウンド": ("回合", "Round，一局游戏的基本单位"),
    "エコラウンド": ("经济局", "Eco Round，存钱不买装备的回合"),
    "フォースバイ": ("强起", "Force Buy，钱不够全买但也要买一些"),
    "フルバイ": ("全起", "Full Buy，买满装备武器"),
    "セーブ": ("存钱", "Save，不买装备为下局攒钱"),
    "クレジット": ("经济", "Credits，游戏内金钱"),
    "リテイク": ("回防", "Retake，重新夺回已失守的点位"),
    "クラッチ": ("残局", "Clutch，少打多的翻盘局面"),
    "エントリー": ("进点", "Entry，最先冲入点位的行动，俗称「突破」"),
    "ローテーション": ("转点", "Rotation，从一个点位转移攻击另一个"),
    "イニシエーター": ("先锋", "Initiator，负责收集信息开路的角色定位"),
    "デュエリスト": ("决斗者", "Duelist，主力击杀型角色定位"),
    "コントローラー": ("控场", "Controller，负责烟雾和区域控制的角色定位"),
    "センチネル": ("哨位", "Sentinel，防守型角色定位"),
    "ピーク": ("探点", "Peek，从掩体后探出查看敌人"),
    "ワイドピーク": ("大拉", "Wide Peek，大幅探出掩体"),
    "ジグルピーク": ("晃身", "Jiggle Peek，快速露头收头获取信息"),
    "カニ歩き": ("横移", "Crab Walk / Strafe，左右移动对枪"),
    "ラーク": ("老六", "Lurk，在敌人意想不到的位置蹲守"),
    "フェイク": ("假打", "Fake，假装进攻某点实际打另一点，俗称「佯攻」"),
    "詰める": ("前压", "Push Forward，向前推进压缩敌人空间"),
    "攻める": ("进攻", "Attack，进攻方行动"),
    "守って": ("防守", "Defend / Hold，守住点位"),
    "前に出る": ("往前顶", "Advance，向敌方方向推进"),
    "引く": ("后退", "Fall Back，拉开距离"),
    "バラけて": ("散开", "Spread Out，别站一起避免被集火"),
    "固まって": ("抱团", "Group Up，集中在一起行动"),
    "カバー": ("掩护", "Cover，帮队友看角度或补枪"),
    "トレード": ("补枪", "Trade Kill，队友被杀后立刻击杀敌人"),
    "味方": ("队友", "Ally / Teammate，我方人员"),
    "敵": ("敌人", "Enemy，对方玩家"),
    "情報": ("信息", "Info，获取敌人位置等信息"),
    "音する": ("有脚步", "Audio Cue，听到敌人脚步声或技能声"),
    "当たってる": ("打中了", "Tagged / Hit，命中敌人"),
    "瀕死": ("残血", "Low HP，血量极低的状态，俗称「大残」"),
    "解除": ("拆包", "Defuse，拆除炸弹"),
    "設置": ("下包", "Plant the Spike，安放炸弹"),
    "即解除": ("秒拆", "Defusing Now，正在拆包的紧急呼叫"),
    "ディフューズ中": ("拆包中", "Defusing，正在解除炸弹"),
    "時間稼いで": ("拖时间", "Stall for Time，消耗回合时间"),
    "ホールド": ("架住", "Hold Position，稳住不冒进"),
    "プッシュ": ("推进", "Push，向前施压"),
    "GG": ("GG", "Good Game，对局结束礼貌用语"),
    "ナイストライ": ("尽力了", "Nice Try，鼓励队友的用语"),
    "グッドラック": ("祝好运", "Good Luck，开局祝福"),
    "アビリティ": ("技能", "Ability，特工的各种技能"),
    "ウルト": ("大招", "Ult，终极技能的简称"),
    "シグネチャー": ("专属技能", "Signature Ability，每回合免费刷新的技能"),
    "スキル": ("技能", "Skill，游戏中的主动或被动技能"),
    "リコン": ("侦查", "Recon，信息收集能力"),
    "ドローン": ("无人机", "Drone，侦查道具，如Sova的侦查无人机"),
    "フラッシュ": ("闪光", "Flash，致盲技能"),
    "スモーク": ("烟雾", "Smoke，遮蔽视野的技能"),
    "ヒール": ("治疗", "Heal，回复生命值的技能"),
    "テレポート": ("传送", "Teleport，瞬间移动技能"),
    "トラップ": ("陷阱", "Trap，侦查或限制敌人移动的装备"),
    "ウォール": ("墙体", "Wall，屏障技能，如Sage的冰墙"),
    "モロトフ": ("燃烧弹", "Molotov / Incendiary，火焰伤害技能"),
    "グレネード": ("手雷", "Grenade，范围伤害技能"),
    "デコイ": ("诱饵", "Decoy，制造假象迷惑敌人"),
    "ブラインド": ("致盲", "Blind，使敌人视野受限"),
    "スロー": ("减速", "Slow，降低敌人移动速度"),
    "スタン": ("眩晕", "Stun，使敌人无法行动"),
    "ダッシュ": ("突进", "Dash，快速位移技能"),
    "リバイブ": ("复活", "Revive，复活技能，如Sage的大招"),
    "カメラ": ("摄像头", "Camera，监控技能，如Cypher的摄像头"),
    "スパイク": ("炸弹", "Spike，进攻方需要安放的目标物"),
    "アルティメット": ("终极技能", "Ultimate，每个特工最强大的技能"),
    "アビリティ使って": ("交技能", "Use Your Abilities，让队友使用技能"),
    "投げ物": ("投掷物", "Utility / Nades，闪光烟雾等投掷类技能道具"),
    "全員": ("全体", "Everyone，全部人员"),
    "ラスト": ("最后一人", "Last，残局最后存活的玩家"),
    "ハーフ": ("半场", "Half，12回合后攻守互换"),
    "ヘル": ("地狱", "Hell，吐槽局面极其糟糕"),
    "OP": ("过强", "Overpowered，英雄或武器过于强力，俗称「太超模了」"),
    "ダメージ": ("伤害", "Damage，对敌人造成的生命值损失"),
}

# ── LoL Champions (common terms) ──
LOL_CHAMP_CN = {
    "チャンピオン": ("英雄", "Champion，玩家操控的角色"),
    "ピック": ("选择", "Pick，选择英雄"),
    "バン": ("禁用", "Ban，禁止出现某英雄"),
    "スキルショット": ("指向性技能", "Skillshot，需手动瞄准方向的技能"),
    "ポイントクリック": ("点选技能", "Point-and-Click，点击目标直接释放"),
    "パッシブ": ("被动", "Passive，自动生效的技能"),
    "Qスキル": ("Q技能", "Q Ability，第一个主动技能"),
    "Wスキル": ("W技能", "W Ability，第二个主动技能"),
    "Eスキル": ("E技能", "E Ability，第三个主动技能"),
    "Rスキル": ("大招", "R Ability / Ultimate，终极技能"),
    "クールダウン": ("冷却", "Cooldown，技能再次可用前的等待时间"),
    "CD": ("冷却", "Cooldown缩写"),
    "マナ": ("法力值", "Mana，释放技能消耗的资源，俗称「蓝」"),
    "スタック": ("层数", "Stack，累积的技能或物品效果"),
    "レベルアップ": ("升级", "Level Up，英雄等级提升"),
    "スキン": ("皮肤", "Skin，英雄外观装扮"),
    "ダッシュ": ("突进", "Dash，快速位移"),
    "ブリンク": ("闪现", "Blink，瞬间位移技能"),
    "CC": ("控制", "CC (Crowd Control)，限制敌人行动的技能"),
    "ノックアップ": ("击飞", "Knock-Up，将敌人打飞的控制效果"),
    "ノックバック": ("击退", "Knock-Back，将敌人推开的控制效果"),
    "スタン": ("眩晕", "Stun，使目标无法行动"),
    "スネア": ("定身", "Snare / Root，使目标无法移动但可攻击"),
    "サプレス": ("压制", "Suppression，最强控制效果，无法任何操作"),
    "チャーム": ("魅惑", "Charm，使目标走向施法者"),
    "フィアー": ("恐惧", "Fear，使目标随机移动"),
    "タウント": ("嘲讽", "Taunt，强制目标攻击施法者"),
    "スロー": ("减速", "Slow，降低移动速度"),
    "シールド": ("护盾", "Shield，吸收伤害的临时HP"),
    "回復スキル": ("回复", "Heal，恢复生命值的技能"),
    "テレポート": ("传送", "Teleport，召唤师技能之一"),
    "イグナイト": ("点燃", "Ignite，召唤师技能，对敌人造成持续真实伤害"),
    "スマイト": ("惩戒", "Smite，打野必备召唤师技能"),
    "バリア": ("屏障", "Barrier，召唤师技能，提供临时护盾"),
    "クレンズ": ("净化", "Cleanse，召唤师技能，移除控制效果"),
    "サモナースペル": ("召唤师技能", "Summoner Spell，每局可选两个"),
    "インビジブル": ("隐身", "Invisible，无法被敌人看见"),
    "AD": ("物理伤害", "AD (Attack Damage)，物理输出"),
    "AP": ("法术伤害", "AP (Ability Power)，法术输出"),
    "DPS": ("每秒伤害", "DPS (Damage Per Second)，每秒输出"),
    "魔法ダメージ": ("魔法伤害", "Magic Damage，俗称AP伤害"),
    "物理ダメージ": ("物理伤害", "Physical Damage，俗称AD伤害"),
    "確定ダメージ": ("真实伤害", "True Damage，无视护甲魔抗的伤害"),
    "無敵": ("无敌", "Invincible，免疫所有伤害"),
    "レジスト": ("抗性", "Resistance，减少受到的某类伤害"),
    "強化": ("强化", "Buff / Strengthen，能力得到提升"),
    "弱体化": ("削弱", "Nerf / Debuff，能力被降低"),
}

# ── LoL 装备 ──
LOL_ITEM_CN = {
    "コアアイテム": ("核心装备", "Core Item，英雄最关键的装备"),
    "スタートアイテム": ("出门装", "Starting Item，开局购买的第一个装备"),
    "ミシックアイテム": ("神话装备", "Mythic Item，已移除的装备等级"),
    "レジェンダリー": ("传说装备", "Legendary Item，高级装备"),
    "ブーツ": ("鞋子", "Boots，提供移动速度的基础装备"),
    "ポーション": ("药水", "Potion，回复生命值的消耗品，俗称「血瓶」"),
    "トリニティフォース": ("三相之力", "Trinity Force，战士核心装备，俗称「三项」"),
    "インフィニティエッジ": ("无尽之刃", "Infinity Edge，暴击核心装备，俗称「无尽」"),
    "ラバドンデスキャップ": ("灭世者的死亡之帽", "Rabadon's Deathcap，AP核心装，俗称「帽子」"),
    "ゾーニャの砂時計": ("中娅沙漏", "Zhonya's Hourglass，金身装备，俗称「金身」"),
    "ヴォイドスタッフ": ("虚空之杖", "Void Staff，法穿装备，俗称「法穿棒」"),
    "モレロノミコン": ("莫雷洛秘典", "Morellonomicon，重伤装备，俗称「鬼书」"),
    "ブラッドサースター": ("饮血剑", "Bloodthirster，吸血装备，俗称「饮血」"),
    "ガーディアンエンジェル": ("守护天使", "Guardian Angel，复活甲，俗称「春哥甲」"),
    "ブラッククリーバー": ("黑色切割者", "Black Cleaver，破甲装备，俗称「黑切」"),
    "ステラックの篭手": ("斯特拉克的挑战护手", "Sterak's Gage，战士保命装，俗称「血手」"),
    "サンファイアイギス": ("日炎圣盾", "Sunfire Aegis，坦克装备，俗称「日炎」"),
    "ソーンメイル": ("荆棘之甲", "Thornmail，反伤装备，俗称「反甲」"),
    "ワーモグアーマー": ("狂徒铠甲", "Warmog's Armor，回血装备，俗称「狂徒」"),
    "リッチベイン": ("巫妖之祸", "Lich Bane，技能后强化普攻，俗称「巫妖」"),
    "ルーデンテンペスト": ("卢登的激荡", "Luden's Tempest，AP poke装备，俗称「卢登」"),
    "ナッシャートゥース": ("纳什之牙", "Nashor's Tooth，攻速AP装备，俗称「纳什」"),
    "ドランブレード": ("多兰之刃", "Doran's Blade，AD出门装，俗称「多兰剑」"),
    "ドランリング": ("多兰之戒", "Doran's Ring，AP出门装，俗称「多兰戒」"),
    "ドランシールド": ("多兰之盾", "Doran's Shield，坦克出门装，俗称「多兰盾」"),
    "トリンケット": ("饰品", "Trinket，免费的视野装备"),
    "ワード": ("眼", "Ward，提供视野的消耗品"),
    "ピンクワード": ("真眼", "Control Ward / Pink Ward，可看见隐形单位"),
    "デワード": ("排眼", "De-Ward，清除敌方眼位"),
    "アイテム差": ("装备差", "Item Gap，双方装备价值的差距"),
    "ゴールド": ("金币", "Gold，游戏内购买装备的货币"),
    "ボイドスタッフ": ("虚空之杖", "Void Staff，法穿装备"),
    "ゾーニャ": ("金身", "Zhonya的简称，中娅沙漏的主动效果"),
}

# ── LoL 位置与职责 ──
LOL_ROLE_CN = {
    "トップ": ("上单", "Top Lane，地图上方单人线"),
    "ミッド": ("中单", "Mid Lane，地图中央单人线"),
    "ジャングル": ("打野", "Jungle，在野区游走发育的位置"),
    "ボット": ("下路", "Bot Lane，双人线所在路线"),
    "ADC": ("ADC", "AD Carry，下路物理输出核心，俗称「射手」"),
    "サポート": ("辅助", "Support，保护队友和做视野的角色"),
    "レーナー": ("线上玩家", "Laner，在线上的玩家"),
    "サイドレーン": ("边路", "Side Lane，上下路的统称"),
    "ファイター": ("战士", "Fighter，近战输出型英雄，俗称「战士」"),
    "タンク": ("坦克", "Tank，肉盾型英雄，吸收伤害，俗称「肉」"),
    "アサシン": ("刺客", "Assassin，高爆发秒脆皮英雄"),
    "メイジ": ("法师", "Mage，魔法输出型英雄"),
    "マークスマン": ("射手", "Marksman，远程物理输出英雄，俗称「ADC」"),
    "キャリー": ("核心", "Carry，队伍的主要伤害来源"),
    "オートフィル": ("自动补位", "Autofill，系统自动分配位置"),
    "ローム": ("游走", "Roam，离开自己路线支援其他路，俗称「游走」"),
    "レーン戦": ("对线", "Laning Phase，游戏前期的对线阶段"),
    "レーニングフェーズ": ("对线期", "Laning Phase，同上"),
}

# ── LoL 地图资源 ──
LOL_OBJECTIVE_CN = {
    "ネクサス": ("基地", "Nexus，摧毁即可获胜，俗称「水晶」"),
    "インヒビター": ("高地塔", "Inhibitor，摧毁后对方出超级兵，俗称「兵营」"),
    "タワー": ("防御塔", "Tower，自动攻击敌人的建筑，俗称「塔」"),
    "タワー下": ("塔下", "Under Tower，在防御塔保护范围内"),
    "ミニオン": ("小兵", "Minion，自动推进的单位，俗称「兵线」"),
    "CS": ("补刀", "CS (Creep Score)，击杀小兵的总数，俗称「补兵」"),
    "ドラゴン": ("小龙", "Dragon，击杀后提供团队buff，俗称「龙」"),
    "バロン": ("大龙", "Baron Nashor，击杀后提供强力buff，俗称「男爵」"),
    "ヘラルド": ("先锋", "Rift Herald，击杀后可召唤推塔，俗称「峡谷先锋」"),
    "ブルー": ("蓝buff", "Blue Buff，提供法力回复和技能急速，俗称「蓝」"),
    "レッド": ("红buff", "Red Buff，提供生命回复和普攻减速，俗称「红」"),
    "ブッシュ": ("草丛", "Bush，提供隐身视野的植被，俗称「草」"),
    "視界": ("视野", "Vision，对小地图区域的可见性"),
    "視界取り": ("控视野", "Vision Control，通过放眼控制地图信息，俗称「做视野」"),
    "オブジェクト": ("地图资源", "Objective，如龙、塔等战略目标"),
    "オブジェクト管理": ("资源控制", "Objective Control，管理地图资源"),
    "ファーストブラッド": ("一血", "First Blood，全场第一个击杀"),
    "リスポーン": ("重生", "Respawn，死亡后重新出现在基地"),
    "リコール": ("回城", "Recall，传送回基地，俗称「B回家」"),
    "ピン": ("信号", "Ping，在小地图上标记位置"),
    "マップ": ("地图", "Map，小地图或整张地图"),
    "マップ確認": ("看小地图", "Checking Minimap，观察地图动态"),
}

# ── LoL 战术 ──
LOL_TACTIC_CN = {
    "ガンク": ("抓人", "Gank，打野或游走英雄配合队友击杀，俗称「gank」"),
    "ファーム": ("发育", "Farm，击杀小兵积累经济和经验，俗称「刷钱」"),
    "フリーズ": ("控线", "Freeze，保持兵线位置不变以安全发育"),
    "プッシュ": ("推线", "Push，快速清除小兵向前推进"),
    "スプリットプッシュ": ("分带", "Split Push，多人分别带不同路线，俗称「四一分推」"),
    "ファストプッシュ": ("速推", "Fast Push，快速清线推塔"),
    "スロープッシュ": ("慢推", "Slow Push，缓慢积累大波兵线"),
    "レーンフリーズ": ("控线", "Lane Freeze，冻结兵线位置"),
    "ウェーブクリア": ("清线", "Wave Clear，清除一波小兵"),
    "ダイブ": ("越塔", "Dive，在对方塔下发起强杀"),
    "エンゲージ": ("开团", "Engage，主动发起团战，俗称「开」"),
    "ディスエンゲージ": ("撤退", "Disengage，脱离战斗，俗称「拉开」"),
    "キル": ("击杀", "Kill，消灭敌方英雄，俗称「人头」"),
    "ダブルキル": ("双杀", "Double Kill"),
    "トリプルキル": ("三杀", "Triple Kill"),
    "クアッドラキル": ("四杀", "Quadra Kill"),
    "ペンタキル": ("五杀", "Penta Kill"),
    "エース": ("团灭", "Ace，消灭对方全部五人"),
    "アシスト": ("助攻", "Assist，参与击杀但非最后一击"),
    "デス": ("死亡", "Death，英雄被击杀"),
    "KDA": ("KDA", "K/D/A比率，击杀/死亡/助攻比"),
    "スノーボール": ("雪球", "Snowball，优势不断积累扩大，俗称「滚雪球」"),
    "フィード": ("送人头", "Feed，不断被杀让对面发育起来，俗称「送」"),
    "スローイング": ("浪输", "Throwing，优势局被翻盘，俗称「浪了」"),
    "メタ": ("版本答案", "Meta，当前版本最强打法，俗称「版本强势」"),
    "構成": ("阵容", "Team Composition，队伍的攻守结构"),
    "シナジー": ("配合", "Synergy，英雄之间的协同效应"),
    "パッチ": ("更新", "Patch，游戏平衡性调整补丁"),
    "降参": ("投降", "Surrender，同意提前结束对局，俗称「点了」"),
    "サレンダー": ("认输", "Surrender，同投降"),
    "ナーフ": ("削弱", "Nerf，降低英雄或装备强度，俗称「砍」"),
    "バフ": ("增强", "Buff，提高英雄或装备强度"),
    "チームファイト": ("团战", "Teamfight，多人参与的大规模战斗"),
    "集団戦": ("团战", "Teamfight，日文说法"),
    "ウェーブ": ("兵线", "Wave，一波小兵"),
    "ゲーム終盤": ("后期", "Late Game，游戏后期阶段"),
    "プッシュ力": ("推线能力", "Wave Clear Power，快速清线的能力"),
    "チャット": ("聊天", "In-Game Chat，游戏内聊天系统"),
    "コーチング": ("指导", "Coaching，高手教新手提升"),
    "カウンター": ("克制", "Counter，针对性地反制对手"),
    "チャンス": ("机会", "It's a Chance，可以打的信号"),
    "コール": ("指挥", "Call，战术指挥或呼叫"),
    "トロール": ("演员", "Troll，故意捣乱破坏游戏体验，俗称「演员」"),
    "ワンチャン": ("有机会", "One Chance，劣势中还有一线翻盘希望"),
}

# ── LoL 通用交流 ──
LOL_COMM_CN = {
    "ありがとう": ("谢谢", "Thanks，感谢队友"),
    "ごめん": ("对不起", "Sorry，为失误道歉"),
    "うまい": ("漂亮", "Well Played，夸奖队友操作好"),
    "ナイス": ("漂亮", "Nice，好操作的意思"),
    "行ける": ("能赢", "We Can Win，判断能打赢"),
    "行くよ": ("上了", "I'm Going In，准备发起进攻"),
    "任せて": ("交给我", "Leave It to Me，让队友放心"),
    "待って": ("等等", "Wait，请求等待的信号"),
    "逃げて": ("快跑", "Run Away，撤退的紧急呼叫"),
    "引いて": ("后撤", "Back Off，往后退拉开距离"),
    "ファイト": ("加油", "Fight，鼓励队友"),
    "お疲れ様です": ("辛苦了", "GG，对局结束礼貌用语"),
    "ミスった": ("失误了", "I Messed Up，承认操作失误"),
    "無理": ("没戏", "Impossible，局势无法挽回"),
    "川": ("河道", "River，地图中央的河流区域，俗称「河」"),
}

# 合并所有映射
ALL_CN_MAPPINGS = {}
ALL_CN_MAPPINGS.update(VAL_AGENT_CN)
ALL_CN_MAPPINGS.update(VAL_WEAPON_CN)
ALL_CN_MAPPINGS.update(VAL_ARMOR_CN)
ALL_CN_MAPPINGS.update(VAL_WEAPON_TERMS_CN)
ALL_CN_MAPPINGS.update(VAL_MAP_CN)
ALL_CN_MAPPINGS.update(VAL_MAP_CALLOUT_CN)
ALL_CN_MAPPINGS.update(VAL_RANK_CN)
ALL_CN_MAPPINGS.update(VAL_SETTINGS_CN)
ALL_CN_MAPPINGS.update(VAL_TACTICS_CN)
ALL_CN_MAPPINGS.update(LOL_CHAMP_CN)
ALL_CN_MAPPINGS.update(LOL_ITEM_CN)
ALL_CN_MAPPINGS.update(LOL_ROLE_CN)
ALL_CN_MAPPINGS.update(LOL_OBJECTIVE_CN)
ALL_CN_MAPPINGS.update(LOL_TACTIC_CN)
ALL_CN_MAPPINGS.update(LOL_COMM_CN)

# 要去除的标签
TAGS_TO_REMOVE = {"未分级", "一般"}


def get_conn():
    return psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                           options="-c plan_cache_mode=force_custom_plan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只预览不执行")
    args = parser.parse_args()

    conn = get_conn()
    conn.autocommit = True

    # 1. 去除 "未分级" "一般" 标签
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, word, tags FROM vocab_library
            WHERE tags && ARRAY['未分级','一般']::text[]
        """)
        bad_tags = cur.fetchall()
    log.info(f"去除「未分级」「一般」标签: {len(bad_tags)} 条")

    if not args.dry_run and bad_tags:
        with conn.cursor() as cur:
            for rid, word, tags in bad_tags:
                new_tags = [t for t in tags if t not in TAGS_TO_REMOVE]
                if not new_tags:
                    new_tags = ["未分类"]
                cur.execute("UPDATE vocab_library SET tags = %s WHERE id = %s", (new_tags, rid))
        log.info(f"  标签已更新: {len(bad_tags)} 条")

    # 2. 更新游戏词条 meaning 为中文译名格式
    updated = 0
    not_found = []

    for word, (cn_name, description) in ALL_CN_MAPPINGS.items():
        # 查找所有有这个word的游戏词条
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, word, meaning FROM vocab_library
                WHERE word = %s AND (tags && ARRAY['gaming_valorant','gaming_lol']::text[])
            """, (word,))
            rows = cur.fetchall()

        for rid, w, old_meaning in rows:
            new_meaning = f"{cn_name} | {description}"
            if args.dry_run:
                if len(not_found) < 10:
                    log.info(f"  {w}: 「{old_meaning or ''}」→「{new_meaning}」")
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE vocab_library SET meaning = %s WHERE id = %s",
                        (new_meaning, rid)
                    )
            updated += 1

        if not rows:
            not_found.append(word)

    log.info(f"更新 meaning: {updated} 条")
    if not_found:
        log.info(f"  未在DB找到（可能不需要更新）: {len(not_found)} 个")

    # 3. 合并重复词条（同word+同tag的多条记录保留一条最完整的）
    with conn.cursor() as cur:
        cur.execute("""
            SELECT word, tags, COUNT(*) as cnt,
                   ARRAY_AGG(id ORDER BY length(COALESCE(meaning,'')) DESC) as ids
            FROM vocab_library
            WHERE tags && ARRAY['gaming_valorant','gaming_lol']::text[]
            GROUP BY word, tags
            HAVING COUNT(*) > 1
        """)
        dups = cur.fetchall()
    log.info(f"重复游戏词条组: {len(dups)}")

    if not args.dry_run and dups:
        deleted = 0
        with conn.cursor() as cur:
            for word, tags, cnt, ids in dups:
                keep = ids[0]  # 保留meaning最长的那条
                for dup_id in ids[1:]:
                    cur.execute("DELETE FROM vocab_library WHERE id = %s", (dup_id,))
                    deleted += 1
        log.info(f"  删除重复: {deleted} 条（保留最完整的一条）")

    conn.close()

    if args.dry_run:
        log.info("\n=== DRY RUN === 未实际修改")
    else:
        log.info("\n全部完成！")


if __name__ == "__main__":
    main()
