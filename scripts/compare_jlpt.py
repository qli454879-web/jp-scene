#!/usr/bin/env python3
"""
JLPT N4/N3 词表对比 & 打标签工具

用法:
  python3 scripts/compare_jlpt.py --dry-run     # 只对比，不修改 DB
  python3 scripts/compare_jlpt.py --tag          # 对比 + 给已有词打标签
  python3 scripts/compare_jlpt.py --missing      # 输出缺失词列表
"""

import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg
import psycopg.rows
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

# ── N4 词表 (571词, 来源 jlptsensei.com) ──
N4_WORDS = [
    ("あ","あ","Ah; oh"), ("ああ","ああ","ah; yes"), ("アフリカ","afurika","Africa"),
    ("上がる","あがる","to rise"), ("挨拶","あいさつ","to greet"), ("味","あじ","flavor; taste"),
    ("アジア","ajia","Asia"), ("赤ちゃん","あかちゃん","baby; infant"), ("赤ん坊","あかんぼう","baby; infant"),
    ("アクセサリー","akusesarii","accessory"), ("アメリカ","amerika","America"), ("アナウンサー","anaunsaa","announcer"),
    ("あんな","anna","such"), ("案内","あんない","to guide"), ("安心","あんしん","peace of mind"),
    ("安全","あんぜん","safety; security"), ("アルバイト","arubaito","part-time job"), ("アルコール","arukooru","alcohol"),
    ("浅い","あさい","shallow"), ("遊び","あそび","playing"), ("集まる","あつまる","to gather"),
    ("集める","あつめる","to collect"), ("謝る","あやまる","to apologize"), ("倍","ばい","double"),
    ("番組","ばんぐみ","program (e.g. TV)"), ("場所","ばしょ","place"), ("ベル","beru","bell"),
    ("美術館","びじゅつかん","art gallery"), ("びっくり","bikkuri","to be surprised"), ("ビル","biru","building"),
    ("僕","ぼく","I (used by males)"), ("貿易","ぼうえき","trade"), ("部長","ぶちょう","manager"),
    ("ぶどう","budou","grapes"), ("文学","ぶんがく","literature"), ("文化","ぶんか","culture"),
    ("文法","ぶんぽう","grammar"), ("ちゃん","chan","suffix for familiar female person"), ("チェック","chekku","to check"),
    ("血","ち","blood"), ("力","ちから","energy; force"), ("地理","ちり","geography"),
    ("中学校","ちゅうがっこう","junior high school"), ("注意","ちゅうい","caution"), ("注射","ちゅうしゃ","injection"),
    ("駐車場","ちゅうしゃじょう","parking lot"), ("大分","だいぶ","considerably"), ("大学生","だいがくせい","university student"),
    ("大事","だいじ","important"), ("大体","だいたい","roughly"), ("暖房","だんぼう","heating"),
    ("男性","だんせい","man; male"), ("できるだけ","dekiru dake","as much as possible"), ("電報","でんぽう","telegram"),
    ("電灯","でんとう","electric light"), ("どんどん","dondon","rapidly"), ("泥棒","どろぼう","thief"),
    ("動物園","どうぶつえん","zoo"), ("道具","どうぐ","tool"), ("枝","えだ","branch"),
    ("遠慮","えんりょ","reserve; refraining"), ("選ぶ","えらぶ","to choose"), ("エスカレーター","esukareetaa","escalator"),
    ("ファックス","fakkusu","fax"), ("不便","ふべん","inconvenience"), ("増える","ふえる","to increase"),
    ("深い","ふかい","deep"), ("復習","ふくしゅう","review"), ("複雑","ふくざつ","complexity"),
    ("踏む","ふむ","to step on"), ("船","ふね","ship"), ("降り出す","ふりだす","to start to rain"),
    ("布団","ふとん","futon"), ("太る","ふとる","to become fat"), ("普通","ふつう","usually"),
    ("ガラス","garasu","a glass"), ("ガソリン","gasorin","gasoline"), ("ガソリンスタンド","gasorin sutando","petrol station"),
    ("ガス","gasu","petrol"), ("原因","げんいん","cause"), ("下宿","げしゅく","lodging"),
    ("技術","ぎじゅつ","technology; skill"), ("ごちそう","gochisou","a feast"), ("ごみ","gomi","rubbish"),
    ("ご覧になる","ごらんになる","(respectful) to see"), ("ご主人","ごしゅじん","your husband"), ("ご存じ","ごぞんじ","knowing"),
    ("具合","ぐあい","condition; health"), ("葉","は","leaves"), ("拝見","はいけん","seeing"),
    ("歯医者","はいしゃ","dentist"), ("はっきり","hakkiri","clearly"), ("運ぶ","はこぶ","to carry"),
    ("花見","はなみ","cherry blossom viewing"), ("ハンドバッグ","handobaggu","handbag"), ("反対","はんたい","opposition"),
    ("払う","はらう","to pay"), ("発音","はつおん","pronunciation"), ("林","はやし","woods; forest"),
    ("恥ずかしい","はずかしい","embarrassed"),
    # Page 2 (101-200)
    ("変","へん","strange"), ("返事","へんじ","reply"), ("火","ひ","fire"),
    ("酷い","ひどい","terrible"), ("冷える","ひえる","to grow cold"), ("髭","ひげ","beard"),
    ("非常に","ひじょうに","extremely"), ("光","ひかり","light"), ("光る","ひかる","to shine"),
    ("引き出し","ひきだし","drawer"), ("引き出す","ひきだす","to withdraw"), ("引っ越す","ひっこす","to move house"),
    ("飛行場","ひこうじょう","airfield"), ("開く","ひらく","to open"), ("拾う","ひろう","to pick up"),
    ("昼間","ひるま","daytime"), ("昼休み","ひるやすみ","lunch break"), ("久しぶり","ひさしぶり","after a long time"),
    ("褒める","ほめる","to praise"), ("翻訳","ほんやく","translation"), ("星","ほし","star"),
    ("ほとんど","ほとんど","mostly"), ("法律","ほうりつ","law"), ("放送","ほうそう","to broadcast"),
    ("一度","いちど","once"), ("以外","いがい","excepting"), ("医学","いがく","medical science"),
    ("いじめる","いじめる","to tease"), ("以上","いじょう","and more"), ("以下","いか","not exceeding"),
    ("意見","いけん","opinion"), ("生き物","いきもの","living thing"), ("生きる","いきる","to live"),
    ("以内","いない","within"), ("田舎","いなか","countryside"), ("祈る","いのる","to pray"),
    ("いっぱい","いっぱい","full"), ("色んな","いろんな","various"), ("石","いし","stone"),
    ("急ぐ","いそぐ","to hurry"), ("一生懸命","いっしょうけんめい","with utmost effort"), ("頂く","いただく","(humble) to receive"),
    ("致す","いたす","(humble) to do"), ("糸","いと","thread"), ("ジャム","ジャム","jam"),
    ("字","じ","character"), ("時代","じだい","period"), ("事故","じこ","accident"),
    ("事務所","じむしょ","office"), ("神社","じんじゃ","Shinto shrine"), ("人口","じんこう","population"),
    ("人生","じんせい","human life"), ("地震","じしん","earthquake"), ("辞典","じてん","dictionary"),
    ("自由","じゆう","freedom"), ("女性","じょせい","woman"), ("準備","じゅんび","to prepare"),
    ("十分","じゅうぶん","enough"), ("柔道","じゅうどう","judo"), ("住所","じゅうしょ","address"),
    ("カーテン","カーテン","curtain"), ("壁","かべ","wall"), ("課長","かちょう","section manager"),
    ("帰り","かえり","return"), ("変える","かえる","to change"), ("科学","かがく","science"),
    ("鏡","かがみ","mirror"), ("海岸","かいがん","coast"), ("会議","かいぎ","meeting"),
    ("会議室","かいぎしつ","conference room"), ("会場","かいじょう","venue"), ("会話","かいわ","conversation"),
    ("火事","かじ","fire"), ("格好","かっこう","appearance"), ("構う","かまう","to mind"),
    ("髪","かみ","hair"), ("噛む","かむ","to bite"), ("家内","かない","(my) wife"),
    ("必ず","かならず","always"), ("悲しい","かなしい","sad"), ("考える","かんがえる","to think"),
    ("看護婦","かんごふ","female nurse"), ("関係","かんけい","relationship"), ("彼女","かのじょ","she; her"),
    ("簡単","かんたん","simple"), ("彼","かれ","he"), ("彼ら","かれら","they"),
    ("形","かたち","shape"), ("片付ける","かたづける","to tidy up"), ("硬い","かたい","hard"),
    ("勝つ","かつ","to win"), ("乾く","かわく","to get dry"), ("代わり","かわり","instead"),
    ("変わる","かわる","to change"), ("通う","かよう","to commute"), ("飾る","かざる","to decorate"),
    ("毛","け","hair or fur"), ("ケーキ","ケーキ","cake"), ("怪我","けが","to injure"),
    ("計画","けいかく","to plan"),
    # Page 3 (201-300)
    ("経験","けいけん","to experience"), ("警察","けいさつ","police"), ("経済","けいざい","economy"),
    ("見物","けんぶつ","sightseeing"), ("喧嘩","けんか","to quarrel"), ("研究","けんきゅう","research"),
    ("研究室","けんきゅうしつ","laboratory"), ("消しゴム","けしごむ","eraser"), ("景色","けしき","scenery"),
    ("気","き","spirit"), ("厳しい","きびしい","strict"), ("気分","きぶん","feeling; mood"),
    ("機会","きかい","chance"), ("危険","きけん","danger"), ("聞こえる","きこえる","to be heard"),
    ("決まる","きまる","to be decided"), ("決める","きめる","to decide"), ("君","きみ","you"),
    ("気持ち","きもち","feeling"), ("着物","きもの","kimono"), ("近所","きんじょ","neighbourhood"),
    ("絹","きぬ","silk"), ("季節","きせつ","season"), ("汽車","きしゃ","train"),
    ("規則","きそく","rule"), ("子","こ","child"), ("心","こころ","heart"),
    ("国際","こくさい","international"), ("細かい","こまかい","small, fine"), ("米","こめ","rice"),
    ("込む","こむ","to be crowded"), ("今度","こんど","this time; next time"), ("この間","このあいだ","recently"),
    ("このごろ","このごろ","these days"), ("コンピュータ","コンピュータ","computer"), ("コンサート","コンサート","concert"),
    ("今夜","こんや","tonight"), ("これから","これから","after this"), ("故障","こしょう","to break down"),
    ("答え","こたえ","response"), ("小鳥","ことり","small bird"), ("こう","こう","this way"),
    ("校長","こうちょう","principal"), ("講堂","こうどう","auditorium"), ("郊外","こうがい","suburb"),
    ("講義","こうぎ","lecture"), ("工業","こうぎょう","industry"), ("工場","こうじょう","factory"),
    ("高校","こうこう","senior high school"), ("高校生","こうこうせい","high school student"), ("公務員","こうむいん","government worker"),
    ("高等学校","こうとうがっこう","high school"), ("交通","こうつう","traffic"), ("怖い","こわい","frightening"),
    ("壊れる","こわれる","to be broken"), ("壊す","こわす","to break"), ("首","くび","neck"),
    ("下さる","くださる","(respectful) to give"), ("雲","くも","cloud"), ("君","くん","suffix for familiar young male"),
    ("比べる","くらべる","to compare"), ("暮れる","くれる","to get dark"), ("草","くさ","grass"),
    ("空気","くうき","air"), ("空港","くうこう","airport"), ("客","きゃく","guest"),
    ("教育","きょういく","education"), ("教会","きょうかい","church"), ("興味","きょうみ","interest"),
    ("競争","きょうそう","competition"), ("急","きゅう","sudden"), ("急行","きゅうこう","hurrying"),
    ("間違える","まちがえる","to make a mistake"), ("参る","まいる","(humble) to go"), ("負ける","まける","to lose"),
    ("漫画","まんが","comic"), ("間に合う","まにあう","to be in time"), ("真ん中","まんなか","middle"),
    ("周り","まわり","around"), ("回る","まわる","to go around"), ("まず","まず","first of all"),
    ("召し上がる","めしあがる","to eat; to drink"), ("珍しい","めずらしい","unusual"), ("見える","みえる","to be seen"),
    ("港","みなと","harbour"), ("味噌","みそ","miso"), ("見つかる","みつかる","to be found"),
    ("見つける","みつける","to discover"), ("都","みやこ","capital"), ("湖","みずうみ","lake"),
    ("戻る","もどる","to turn back"), ("木綿","もめん","cotton"), ("森","もり","forest"),
    ("もし","もし","if"), ("申し上げる","もうしあげる","to say; to offer"), ("申す","もうす","to be called"),
    ("もうすぐ","もうすぐ","soon"), ("迎える","むかえる","to go out to meet"), ("昔","むかし","olden days"),
    ("向かう","むかう","to head towards"),
    # Page 4 (301-400)
    ("無理","むり","impossible"), ("虫","むし","insect"), ("息子","むすこ","son"),
    ("娘","むすめ","daughter"), ("投げる","なげる","to throw"), ("泣く","なく","to weep"),
    ("無くなる","なくなる","to disappear"), ("亡くなる","なくなる","to die"), ("生","なま","raw"),
    ("直る","なおる","to be fixed"), ("治る","なおる","to be cured"), ("慣れる","なれる","to get used to"),
    ("鳴る","なる","to sound"), ("なるほど","なるほど","I understand"), ("寝坊","ねぼう","oversleeping"),
    ("値段","ねだん","price"), ("眠い","ねむい","sleepy"), ("眠る","ねむる","to sleep"),
    ("熱","ねつ","fever"), ("苦い","にがい","bitter"), ("逃げる","にげる","to escape"),
    ("二階建て","にかいだて","two-storied building"), ("人形","にんぎょう","doll"), ("匂い","におい","a smell"),
    ("似る","にる","to be similar"), ("喉","のど","throat"), ("残る","のこる","to remain"),
    ("乗り換える","のりかえる","to change (bus/train)"), ("乗り物","のりもの","vehicle"), ("濡れる","ぬれる","to get wet"),
    ("塗る","ぬる","to paint"), ("盗む","ぬすむ","to steal"), ("入学","にゅうがく","entry to school"),
    ("入院","にゅういん","hospitalization"), ("落ちる","おちる","to fall"), ("踊り","おどり","a dance"),
    ("驚く","おどろく","to be surprised"), ("踊る","おどる","to dance"), ("お出でになる","おいでになる","(respectful) to be"),
    ("お祝い","おいわい","congratulation"), ("お嬢さん","おじょうさん","(another's) daughter"), ("可笑しい","おかしい","strange or funny"),
    ("行う","おこなう","to perform"), ("怒る","おこる","to be angry"), ("起こす","おこす","to wake"),
    ("億","おく","one hundred million"), ("屋上","おくじょう","rooftop"), ("遅れる","おくれる","to be late"),
    ("贈り物","おくりもの","gift"), ("送る","おくる","to send"), ("お祭り","おまつり","festival"),
    ("お見舞い","おみまい","visiting ill people"), ("お土産","おみやげ","souvenir"), ("おもちゃ","おもちゃ","toy"),
    ("思い出す","おもいだす","to remember"), ("表","おもて","the front"), ("オートバイ","オートバイ","motorcycle"),
    ("お礼","おれい","thanks"), ("折れる","おれる","to be broken"), ("下りる","おりる","to get off"),
    ("折る","おる","to break or fold"), ("押し入れ","おしいれ","closet"), ("仰る","おっしゃる","(respectful) to say"),
    ("お宅","おたく","your home"), ("音","おと","sound"), ("落とす","おとす","to drop"),
    ("お釣り","おつり","change"), ("夫","おっと","husband"), ("終わり","おわり","the end"),
    ("親","おや","parents"), ("泳ぎ方","およぎかた","way of swimming"), ("パート","パート","part time"),
    ("パソコン","パソコン","personal computer"), ("ピアノ","ピアノ","piano"), ("プレゼント","プレゼント","gift"),
    ("冷房","れいぼう","air conditioning"), ("レジ","レジ","cashier"), ("歴史","れきし","history"),
    ("連絡","れんらく","to contact"), ("レポート","レポート","report"), ("利用","りよう","use"),
    ("理由","りゆう","reason"), ("留守","るす","absence"), ("旅館","りょかん","traditional inn"),
    ("両方","りょうほう","both sides"), ("寂しい","さびしい","lonely"), ("下がる","さがる","to get down"),
    ("探す","さがす","to look for"), ("下げる","さげる","to lower"), ("最後","さいご","end"),
    ("最近","さいきん","recently"), ("最初","さいしょ","beginning"), ("坂","さか","slope"),
    ("盛ん","さかん","prosperous"), ("昨夜","さくや","last night"), ("サンダル","サンダル","sandal"),
    ("サンドイッチ","サンドイッチ","sandwich"), ("産業","さんぎょう","industry"), ("サラダ","サラダ","salad"),
    ("再来月","さらいげつ","month after next"),
    # Page 5 (401-500)
    ("再来週","さらいしゅう","week after next"), ("差し上げる","さしあげる","to give"), ("騒ぐ","さわぐ","to make noise"),
    ("触る","さわる","to touch"), ("生物","せいぶつ","living thing"), ("政治","せいじ","politics"),
    ("生活","せいかつ","to live"), ("生命","せいめい","life"), ("生産","せいさん","production"),
    ("西洋","せいよう","the west"), ("世界","せかい","the world"), ("席","せき","seat"),
    ("線","せん","line"), ("背中","せなか","back of body"), ("先輩","せんぱい","senior"),
    ("戦争","せんそう","war"), ("説明","せつめい","explanation"), ("社長","しゃちょう","company president"),
    ("社会","しゃかい","society"), ("市","し","city"), ("試合","しあい","match"),
    ("叱る","しかる","to scold"), ("仕方","しかた","way; method"), ("試験","しけん","examination"),
    ("しっかり","しっかり","firmly"), ("島","しま","island"), ("市民","しみん","citizen"),
    ("品物","しなもの","goods"), ("新聞社","しんぶんしゃ","newspaper company"), ("親切","しんせつ","kindness"),
    ("失敗","しっぱい","failure"), ("調べる","しらべる","to investigate"), ("知らせる","しらせる","to notify"),
    ("下着","したぎ","underwear"), ("食料品","しょくりょうひん","groceries"), ("小学校","しょうがっこう","elementary school"),
    ("生じる","しょうじる","to produce"), ("紹介","しょうかい","introduction"), ("将来","しょうらい","future"),
    ("小説","しょうせつ","novel"), ("趣味","しゅみ","hobby"), ("習慣","しゅうかん","habit"),
    ("祖母","そぼ","grandmother"), ("育てる","そだてる","to bring up"), ("祖父","そふ","grandfather"),
    ("ソフト","ソフト","soft"), ("そんな","そんな","that sort of"), ("それで","それで","because of that"),
    ("それほど","それほど","to that extent"), ("そろそろ","そろそろ","soon"), ("卒業","そつぎょう","graduation"),
    ("相談","そうだん","to discuss"), ("素晴らしい","すばらしい","wonderful"), ("滑る","すべる","to slide"),
    ("凄い","すごい","terrific"), ("水道","すいどう","water supply"), ("水泳","すいえい","swimming"),
    ("すっかり","すっかり","completely"), ("空く","すく","to be hungry"), ("スクリーン","スクリーン","screen"),
    ("隅","すみ","corner"), ("済む","すむ","to finish"), ("砂","すな","sand"),
    ("すり","すり","pickpocket"), ("スーツケース","スーツケース","suitcase"), ("進む","すすむ","to make progress"),
    ("ステーキ","ステーキ","steak"), ("ステレオ","ステレオ","stereo"), ("捨てる","すてる","to throw away"),
    ("数学","すうがく","mathematics"), ("スーツ","スーツ","suit"), ("正しい","ただしい","correct"),
    ("退院","たいいん","leaving hospital"), ("台風","たいふう","typhoon"), ("タイプ","タイプ","type"),
    ("たいてい","たいてい","usually"), ("たまに","たまに","occasionally"), ("棚","たな","shelves"),
    ("誕生","たんじょう","birth"), ("楽しみ","たのしみ","looking forward to"), ("倒れる","たおれる","to fall"),
    ("足りる","たりる","to be sufficient"), ("足す","たす","to add"), ("畳","たたみ","Japanese straw mat"),
    ("建てる","たてる","to build"), ("尋ねる","たずねる","to ask"), ("訪ねる","たずねる","to visit"),
    ("手袋","てぶくろ","glove"), ("丁寧","ていねい","polite"), ("テキスト","テキスト","text"),
    ("適当","てきとう","suitable"), ("点","てん","point"), ("店員","てんいん","clerk"),
    ("テニス","テニス","tennis"), ("天気予報","てんきよほう","weather forecast"), ("展覧会","てんらんかい","exhibition"),
    ("寺","てら","temple"), ("手伝う","てつだう","to help"), ("途中","とちゅう","on the way"),
    ("届ける","とどける","to send"),
    # Page 6 (501-571)
    ("特急","とっきゅう","limited express"), ("床屋","とこや","barber"), ("特別","とくべつ","special"),
    ("特に","とくに","particularly"), ("泊まる","とまる","to stay at"), ("止める","とめる","to stop"),
    ("遠く","とおく","distant"), ("通る","とおる","to go through"), ("取り替える","とりかえる","to exchange"),
    ("到頭","とうとう","finally"), ("続ける","つづける","to continue"), ("続く","つづく","to continue"),
    ("都合","つごう","convenience"), ("捕まえる","つかまえる","to catch"), ("漬ける","つける","to soak"),
    ("月","つき","moon"), ("付く","つく","to be attached"), ("妻","つま","wife"),
    ("連れる","つれる","to take with one"), ("釣る","つる","to fish"), ("伝える","つたえる","to report"),
    ("包む","つつむ","to wrap"), ("腕","うで","arm"), ("植える","うえる","to plant"),
    ("動く","うごく","to move"), ("伺う","うかがう","to visit"), ("受ける","うける","to take a test"),
    ("受付","うけつけ","reception"), ("生まれ","うまれ","birth"), ("運転手","うんてんしゅ","driver"),
    ("裏","うら","reverse side"), ("嬉しい","うれしい","happy"), ("売り場","うりば","selling area"),
    ("嘘","うそ","a lie"), ("打つ","うつ","to hit"), ("美しい","うつくしい","beautiful"),
    ("移る","うつる","to move house"), ("写す","うつす","to copy"), ("ワープロ","ワープロ","word processor"),
    ("別れる","わかれる","to separate"), ("沸かす","わかす","to boil"), ("沸く","わく","to boil"),
    ("笑う","わらう","to laugh"), ("割れる","われる","to break"), ("割合","わりあい","rate; ratio"),
    ("忘れ物","わすれもの","lost article"), ("焼ける","やける","to burn"), ("焼く","やく","to bake"),
    ("役に立つ","やくにたつ","to be helpful"), ("約束","やくそく","promise"), ("止む","やむ","to stop"),
    ("やっぱり","やっぱり","as I thought"), ("優しい","やさしい","kind"), ("痩せる","やせる","to become thin"),
    ("柔らかい","やわらかい","soft"), ("汚れる","よごれる","to get dirty"), ("喜ぶ","よろこぶ","to be delighted"),
    ("寄る","よる","to visit"), ("予習","よしゅう","preparation"), ("予定","よてい","plan"),
    ("用","よう","business; errand"), ("用意","ようい","preparation"), ("用事","ようじ","tasks"),
    ("予約","よやく","reservation"), ("湯","ゆ","hot water"), ("指","ゆび","finger"),
    ("指輪","ゆびわ","finger ring"), ("夢","ゆめ","dream"), ("揺れる","ゆれる","to shake"),
    ("残念","ざんねん","regrettable"), ("全然","ぜんぜん","not entirely"),
]

# ── N3 词表 (192词, 来源 jlptsensei.com) ──
N3_WORDS = [
    ("明かり","あかり","light; illumination"), ("明ける","あける","to dawn"), ("明らか","あきらか","clear; obvious"),
    ("悪魔","あくま","devil; demon"), ("暗記","あんき","memorization"), ("新た","あらた","new; fresh"),
    ("有らゆる","あらゆる","all; every"), ("集まり","あつまり","gathering"), ("部分","ぶぶん","portion; section"),
    ("分","ぶん","part; segment"), ("文明","ぶんめい","civilization"), ("分析","ぶんせき","analysis"),
    ("分野","ぶんや","field; sphere"), ("父親","ちちおや","father"), ("地平線","ちへいせん","horizon"),
    ("地位","ちい","position; status"), ("長期","ちょうき","long-term"), ("中","ちゅう","during; medium"),
    ("中学","ちゅうがく","junior high school"), ("昼食","ちゅうしょく","lunch"), ("大部分","だいぶぶん","most part"),
    ("駄目","だめ","no good"), ("男子","だんし","youth"), ("出会い","であい","meeting"),
    ("出会う","であう","to meet by chance"), ("読書","どくしょ","reading"), ("努力","どりょく","effort"),
    ("同一","どういつ","identical"), ("円","えん","yen; circle"), ("不利","ふり","disadvantage"),
    ("不足","ふそく","insufficiency"), ("再び","ふたたび","again"), ("外交","がいこう","diplomacy"),
    ("外出","がいしゅつ","going out"), ("学期","がっき","school term"), ("学","がく","learning"),
    ("学問","がくもん","scholarship"), ("学者","がくしゃ","scholar"), ("学習","がくしゅう","study"),
    ("議長","ぎちょう","chairman"), ("議会","ぎかい","congress"), ("語学","ごがく","study of languages"),
    ("激しい","はげしい","violent"), ("母親","ははおや","mother"), ("博物館","はくぶつかん","museum"),
    ("販売","はんばい","sales"), ("発明","はつめい","invention"), ("外す","はずす","to remove"),
    ("品","ひん","elegance; article"), ("一言","ひとこと","single word"), ("一人一人","ひとりひとり","one by one"),
    ("本物","ほんもの","genuine article"), ("本人","ほんにん","said person"), ("一致","いっち","agreement"),
    ("一時","いちじ","one o'clock"), ("意外","いがい","unexpected"), ("一家","いっか","a family"),
    ("今に","いまに","before long"), ("今にも","いまにも","at any moment"), ("一般","いっぱん","general"),
    ("一方","いっぽう","one of two"), ("一生","いっしょう","whole life"), ("一種","いっしゅ","kind; variety"),
    ("一瞬","いっしゅん","instant"), ("一層","いっそう","much more"), ("一体","いったい","what the heck"),
    ("所謂","いわゆる","so-called"), ("邪魔","じゃま","hindrance"), ("化学","かがく","chemistry"),
    ("会","かい","meeting; association"), ("会員","かいいん","member"), ("海外","かいがい","foreign"),
    ("会合","かいごう","meeting"), ("会計","かいけい","finance"), ("開始","かいし","start"),
    ("科目","かもく","school subject"), ("権利","けんり","right; privilege"), ("基本","きほん","basics"),
    ("記事","きじ","article"), ("気味","きみ","sensation; feeling"), ("記念","きねん","commemoration"),
    ("気に入る","きにいる","to like"), ("記入","きにゅう","filling in"), ("記憶","きおく","memory"),
    ("記者","きしゃ","reporter"), ("期待","きたい","expectation"), ("国家","こっか","state; country"),
    ("国会","こっかい","National Diet"), ("国境","こっきょう","national border"), ("国語","こくご","national language"),
    ("国民","こくみん","people of a country"), ("今後","こんご","from now on"), ("今回","こんかい","this time"),
    ("今日","こんにち","today"), ("転ぶ","ころぶ","to fall down"), ("高速","こうそく","high-speed"),
    ("訓練","くんれん","training"), ("教科書","きょうかしょ","textbook"), ("協力","きょうりょく","cooperation"),
    ("強力","きょうりょく","powerful"),
    # Page 2 (101-192)
    ("急激","きゅうげき","sudden"), ("急に","きゅうに","swiftly"), ("吸収","きゅうしゅう","absorption"),
    ("急速","きゅうそく","rapid"), ("真面目","まじめ","serious"), ("真っ赤","まっか","bright red"),
    ("学ぶ","まなぶ","to study in depth"), ("万一","まんいち","unlikely event"), ("満足","まんぞく","satisfaction"),
    ("明確","めいかく","clear; precise"), ("飯","めし","cooked rice"), ("味方","みかた","ally"),
    ("魅力","みりょく","charm"), ("木曜","もくよう","Thursday"), ("半ば","なかば","half"),
    ("熱心","ねっしん","enthusiastic"), ("日本","にほん","Japan"), ("能力","のうりょく","ability"),
    ("入場","にゅうじょう","entrance"), ("お昼","おひる","lunch"), ("収める","おさめる","to pay"),
    ("連続","れんぞく","continuation"), ("利益","りえき","profit"), ("利口","りこう","clever"),
    ("留学","りゅうがく","studying abroad"), ("作品","さくひん","work of art"), ("左右","さゆう","left and right"),
    ("成長","せいちょう","growth"), ("製品","せいひん","product"), ("青年","せいねん","youth"),
    ("刺激","しげき","stimulus"), ("資本","しほん","capital"), ("品","しな","article; item"),
    ("身長","しんちょう","body height"), ("進学","しんがく","entering higher school"), ("新鮮","しんせん","fresh"),
    ("支店","してん","branch office"), ("使用","しよう","use"), ("食品","しょくひん","food"),
    ("書物","しょもつ","book"), ("書類","しょるい","document"), ("書斎","しょさい","study"),
    ("商売","しょうばい","trade"), ("奨学金","しょうがくきん","scholarship"), ("正午","しょうご","midday"),
    ("商品","しょうひん","commodity"), ("少女","しょうじょ","little girl"), ("証明","しょうめい","proof"),
    ("少年","しょうねん","boy"), ("少々","しょうしょう","just a minute"), ("招待","しょうたい","invitation"),
    ("週","しゅう","week"), ("集中","しゅうちゅう","concentration"), ("集団","しゅうだん","group"),
    ("収穫","しゅうかく","harvest"), ("週間","しゅうかん","week"), ("週刊","しゅうかん","weekly publication"),
    ("収入","しゅうにゅう","income"), ("速度","そくど","speed"), ("少しも","すこしも","not one bit"),
    ("大半","たいはん","majority"), ("大会","たいかい","tournament"), ("大した","たいした","considerable"),
    ("単なる","たんなる","mere"), ("多少","たしょう","somewhat"), ("手品","てじな","magic trick"),
    ("哲学","てつがく","philosophy"), ("徹夜","てつや","all night"), ("土地","とち","land"),
    ("都会","とかい","big city"), ("取れる","とれる","to come off"), ("取り上げる","とりあげる","to pick up"),
    ("図書","としょ","books"), ("通学","つうがく","commuting to school"), ("受け取る","うけとる","to receive"),
    ("上手い","うまい","skillful"), ("運転","うんてん","driving"), ("売れる","うれる","to sell well"),
    ("分ける","わける","to divide"), ("悪口","わるぐち","slander"), ("夜明け","よあけ","dawn"),
    ("余分","よぶん","extra"), ("読み","よみ","reading"), ("夜中","よなか","middle of the night"),
    ("宜しい","よろしい","OK; all right"), ("唯一","ゆいいつ","only"), ("輸入","ゆにゅう","import"),
    ("輸出","ゆしゅつ","export"), ("夕べ","ゆうべ","evening"), ("有利","ゆうり","advantageous"),
    ("全国","ぜんこく","the whole country"), ("随分","ずいぶん","very; extremely"),
]


def connect_db():
    if not DB_URL:
        print("ERROR: SUPABASE_DB_URL not set")
        sys.exit(1)
    return psycopg.connect(DB_URL, prepare_threshold=None, connect_timeout=10)


def match_word(conn, word, reading):
    """在 vocab_library 中匹配单词。返回 (id, word, reading, level) 或 None。"""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # 精确匹配 word
        cur.execute("SELECT id, word, reading, level, tags FROM vocab_library WHERE word = %s LIMIT 1", (word,))
        row = cur.fetchone()
        if row:
            return row
        # 精确匹配 reading
        cur.execute("SELECT id, word, reading, level, tags FROM vocab_library WHERE reading = %s LIMIT 1", (reading,))
        row = cur.fetchone()
        if row:
            return row
        # 模糊匹配 word
        cur.execute("SELECT id, word, reading, level, tags FROM vocab_library WHERE word ILIKE %s LIMIT 1", (word + "%",))
        row = cur.fetchone()
        if row:
            return row
    return None


def add_tag(conn, word_id, tag):
    """给单词添加标签（幂等）。"""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE vocab_library SET tags = array_append(tags, %s) WHERE id = %s AND NOT (%s = ANY(tags))",
            (tag, word_id, tag)
        )


def main():
    dry_run = "--dry-run" in sys.argv
    do_tag = "--tag" in sys.argv
    do_missing = "--missing" in sys.argv

    if not (dry_run or do_tag or do_missing):
        print("用法: python3 compare_jlpt.py [--dry-run | --tag | --missing]")
        print("  --dry-run  只对比，不修改 DB")
        print("  --tag      给已有词打标签")
        print("  --missing  输出缺失词 CSV")
        sys.exit(1)

    conn = connect_db()
    all_lists = [("jlpt_n4", N4_WORDS), ("jlpt_n3", N3_WORDS)]

    existing = []
    missing = []
    tagged = 0

    for tag, wordlist in all_lists:
        print(f"\n{'='*60}")
        print(f"处理 {tag} ({len(wordlist)} 词)")
        print(f"{'='*60}")

        for word, reading, meaning in wordlist:
            matched = match_word(conn, word, reading)
            if matched:
                mid = str(matched["id"])
                db_word = matched["word"]
                db_reading = matched["reading"]
                db_level = matched.get("level") or ""
                existing_tags = list(matched.get("tags") or [])
                status = "已有标签" if tag in existing_tags else "待打标签"
                print(f"  ✅ {word}（{reading}） → 匹配到「{db_word}」[{db_level}] {status}")
                existing.append((tag, word, reading, db_word, db_reading, db_level, mid, status))

                if do_tag and tag not in existing_tags:
                    add_tag(conn, mid, tag)
                    tagged += 1
            else:
                print(f"  ❌ {word}（{reading}） → 未匹配")
                missing.append((tag, word, reading, meaning))

    if do_tag:
        conn.commit()
        print(f"\n🏷️  已打标签: {tagged} 个")

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总: 已有 {len(existing)} 词 / 缺失 {len(missing)} 词")
    print(f"{'='*60}")

    # 输出 CSV
    if existing:
        with open("jlpt_existing.csv", "w") as f:
            f.write("tag,official_word,official_reading,db_word,db_reading,db_level,db_id,status\n")
            for row in existing:
                f.write(",".join(str(c) for c in row) + "\n")
        print(f"✅ jlpt_existing.csv ({len(existing)} 条)")

    if missing:
        with open("jlpt_missing.csv", "w") as f:
            f.write("tag,word,reading,meaning\n")
            for row in missing:
                f.write(",".join(str(c) for c in row) + "\n")
        print(f"❌ jlpt_missing.csv ({len(missing)} 条)")

    conn.close()


if __name__ == "__main__":
    main()
