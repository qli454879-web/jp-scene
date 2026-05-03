from fastapi import FastAPI, Query, Body, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from ai_service import AIService
from dictionary_service import DictionaryService
from vocab_service import VocabService
from supabase import create_client
import httpx
import uvicorn
import os
import logging
import secrets
import time
import sqlite3
import json
from typing import Optional, List, Dict, Any, Tuple
import csv
import re
import mimetypes
from urllib.parse import urlparse, unquote
import psycopg
import psycopg.rows
import uuid
import string

# --- Auth Configuration ---
# 禁止硬编码凭证：若未配置 SECRET_KEY，则运行时随机生成（开发/演示用）。
# 生产环境请务必在 Render 环境变量中设置 SECRET_KEY（否则重启会使旧 token 失效）。
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

# --- Supabase Configuration (Stage 1) ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip("`")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
SUPABASE_DB_ENABLED = bool(SUPABASE_DB_URL)

# 避免每次请求都重复跑 ALTER TABLE
_VOCAB_LIBRARY_SCHEMA_OK = False
# 避免每次评分/收藏都重复跑 CREATE TABLE/INDEX（会导致同步明显变慢）
_LIBRARY_USER_LISTS_SCHEMA_OK = False

# 管理员写操作密钥（仅后端校验，不要硬编码到前端代码里）
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

# 简体→日文常用汉字映射（可按需要持续扩充；避免引入冷门第三方库）
# 目标：让用户输入简体（如“强/强化”）也能命中日文词条（如“強/強化”）
S2J_KANJI_MAP: Dict[str, str] = {
    "强": "強",
    "来": "来",  # 兼容：日文常用也是“来”（繁体为“來”）
    "习": "習",
    "学": "学",
    "说": "說",
    "书": "書",
    "听": "聴",
    "写": "書",  # 粗略映射（按需再精细化）
    "气": "気",
    "车": "車",
    "国": "国",
    "会": "会",
    "发": "発",
    "动": "動",
    "过": "過",
    "时": "時",
    "间": "間",
}


def _map_s2j(text: str) -> str:
    return "".join(S2J_KANJI_MAP.get(ch, ch) for ch in (text or ""))


def _is_kana_only(text: str) -> bool:
    """Return True if text consists of only hiragana/katakana/ー."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FFー]+", t))


def _int0(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _normalize_library_selector(level: str) -> str:
    return str(level or "").strip().upper()


def _is_kaoyan_selector(level: str) -> bool:
    return _normalize_library_selector(level) == "KAOYAN"

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# --- Invitation codes ---
_INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # avoid 0/O/1/I


def _gen_invite_code(prefix: str = "JP") -> str:
    """Generate a hard-to-guess invite code, e.g. JP-ABCD-EFGH-IJKL."""
    raw = "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(12))
    groups = "-".join([raw[i : i + 4] for i in range(0, 12, 4)])
    return f"{prefix}-{groups}".upper()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")
bearer_scheme = HTTPBearer(auto_error=False)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    # Cache for AI results
    c.execute('''CREATE TABLE IF NOT EXISTS ai_cache 
                 (word TEXT PRIMARY KEY, result TEXT, timestamp REAL)''')
    
    # User accounts
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  hashed_password TEXT,
                  created_at REAL)''')
    
    # User learning progress
    c.execute('''CREATE TABLE IF NOT EXISTS user_progress 
                 (user_id INTEGER, 
                  word TEXT, 
                  level TEXT,
                  status TEXT, -- 'learning', 'mastered'
                  last_seen REAL,
                  next_review REAL,
                  review_count INTEGER DEFAULT 0,
                  PRIMARY KEY (user_id, word))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS vocab_bank 
                 (word TEXT,
                  kana TEXT,
                  meaning TEXT,
                  level TEXT,
                  order_no INTEGER,
                  PRIMARY KEY (word, level))''')

    c.execute('''CREATE TABLE IF NOT EXISTS daily_goals
                 (user_id INTEGER,
                  level TEXT,
                  date TEXT,
                  target_count INTEGER,
                  done_new INTEGER DEFAULT 0,
                  done_review INTEGER DEFAULT 0,
                  PRIMARY KEY (user_id, level, date))''')

    c.execute('''CREATE TABLE IF NOT EXISTS study_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  level TEXT,
                  date TEXT,
                  started_at REAL,
                  ended_at REAL,
                  total_new INTEGER,
                  total_review INTEGER,
                  know_count INTEGER,
                  fuzzy_count INTEGER,
                  dont_know_count INTEGER)''')

    c.execute('''CREATE TABLE IF NOT EXISTS vocab_meta_cache
                 (word TEXT PRIMARY KEY,
                  meaning_zh TEXT,
                  origin TEXT,
                  updated_at REAL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS forum_posts_local
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_name TEXT,
                  title TEXT,
                  content TEXT NOT NULL,
                  parent_id INTEGER,
                  created_at REAL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS feedbacks_local
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_name TEXT,
                  content TEXT NOT NULL,
                  created_at REAL)''')
    
    conn.commit()
    conn.close()

init_db()

def _ensure_vocab_bank_columns():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE vocab_bank ADD COLUMN order_no INTEGER")
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

_ensure_vocab_bank_columns()

def _migrate_vocab_bank_pk_if_needed():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    try:
        c.execute("PRAGMA table_info(vocab_bank)")
        cols = c.fetchall()
        if not cols:
            return
        pk_cols = [row[1] for row in cols if int(row[5] or 0) > 0]
        if pk_cols == ["word", "level"]:
            return
        if pk_cols != ["word"]:
            return
        c.execute('''CREATE TABLE IF NOT EXISTS vocab_bank_new 
                     (word TEXT,
                      kana TEXT,
                      meaning TEXT,
                      level TEXT,
                      order_no INTEGER,
                      PRIMARY KEY (word, level))''')
        c.execute(
            "INSERT OR IGNORE INTO vocab_bank_new (word, kana, meaning, level, order_no) SELECT word, kana, meaning, level, order_no FROM vocab_bank"
        )
        c.execute("DROP TABLE vocab_bank")
        c.execute("ALTER TABLE vocab_bank_new RENAME TO vocab_bank")
        conn.commit()
    finally:
        conn.close()

_migrate_vocab_bank_pk_if_needed()

_VOCAB_BANK_EXTRA_COLS = [
    ("register_social", "TEXT"),
    ("scene_deep_dive", "TEXT"),
    ("example_ja", "TEXT"),
    ("example_zh", "TEXT"),
    ("usage_frequency_note", "TEXT"),
    ("audio_filename", "TEXT"),
    ("image_prompt", "TEXT"),
]


def _ensure_vocab_bank_extra_columns():
    conn = sqlite3.connect("cache.db")
    c = conn.cursor()
    try:
        for col, typ in _VOCAB_BANK_EXTRA_COLS:
            try:
                c.execute(f"ALTER TABLE vocab_bank ADD COLUMN {col} {typ}")
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()


_ensure_vocab_bank_extra_columns()

VOCAB_SOURCE_FILES = {
    "N5": "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n5.csv",
    "N4": "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n4.csv",
    "N3": "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n3.csv",
    "N2": "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n2.csv",
    "N1": "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n1.csv",
}

def _maybe_populate_vocab_bank():
    if os.getenv("SKIP_VOCAB_DOWNLOAD") == "1":
        return

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM vocab_bank")
        total = int(c.fetchone()[0] or 0)
        if total >= 5000:
            return
    finally:
        conn.close()

    try:
        import httpx
        conn = sqlite3.connect('cache.db')
        c = conn.cursor()
        c.execute("SELECT COALESCE(MAX(order_no), 0) FROM vocab_bank")
        order_no = int(c.fetchone()[0] or 0) + 1

        for level, url in VOCAB_SOURCE_FILES.items():
            r = httpx.get(url, timeout=30)
            r.raise_for_status()
            text = r.text
            reader = csv.DictReader(text.splitlines())
            for row in reader:
                expression = (row.get("expression") or "").strip()
                reading = (row.get("reading") or "").strip()
                meaning = (row.get("meaning") or "").strip()
                if not expression or not meaning:
                    continue
                word = expression.split(";")[0].strip()
                kana = reading.split(";")[0].strip()
                if not word:
                    continue
                c.execute(
                    "INSERT OR IGNORE INTO vocab_bank (word, kana, meaning, level, order_no) VALUES (?, ?, ?, ?, ?)",
                    (word, kana, meaning, level, order_no),
                )
                order_no += 1

        conn.commit()
    except Exception as e:
        print(f"Vocab download skipped: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

_maybe_populate_vocab_bank()

def _populate_vocab_bank_from_builtin_if_needed():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM vocab_bank")
        total = int(c.fetchone()[0] or 0)
        if total >= 500:
            return
    finally:
        conn.close()

    from vocab_service import JLPT_VOCAB

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT COALESCE(MAX(order_no), 0) FROM vocab_bank")
    order_no = int(c.fetchone()[0] or 0) + 1
    for level, items in JLPT_VOCAB.items():
        for item in items:
            word = (item.get("word") or "").strip()
            kana = (item.get("kana") or "").strip()
            meaning = (item.get("meaning") or "").strip()
            if not word or not meaning:
                continue
            c.execute(
                "INSERT OR IGNORE INTO vocab_bank (word, kana, meaning, level, order_no) VALUES (?, ?, ?, ?, ?)",
                (word, kana, meaning, level, order_no),
            )
            order_no += 1
    conn.commit()
    conn.close()

_populate_vocab_bank_from_builtin_if_needed()

# --- Auth Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def _get_user_id_by_username(username: str) -> Optional[int]:
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return int(row[0]) if row else None

SRS_INTERVALS_SECONDS = [
    0,
    60 * 60,
    60 * 60 * 24,
    60 * 60 * 24 * 2,
    60 * 60 * 24 * 4,
    60 * 60 * 24 * 7,
    60 * 60 * 24 * 15,
]

def _next_review_from_rating(now_ts: float, interval_index: int, rating: str):
    if rating == "dont_know":
        interval_index = 0
    elif rating == "fuzzy":
        interval_index = max(0, interval_index - 1)
    else:
        interval_index = min(interval_index + 1, len(SRS_INTERVALS_SECONDS) - 1)
    next_review = now_ts + SRS_INTERVALS_SECONDS[interval_index]
    return interval_index, next_review

# --- Cache Helpers ---
def get_cached_result(word):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT result FROM ai_cache WHERE word=?", (word,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def save_to_cache(word, result):
    if "【系统提示】" in result.get("explanation", ""):
        return
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO ai_cache VALUES (?, ?, ?)", 
              (word, json.dumps(result), time.time()))
    conn.commit()
    conn.close()

app = FastAPI()

# Enable CORS
_cors_origins_raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if _cors_origins_raw:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
else:
    # 兼容旧行为；生产环境建议设置 CORS_ALLOW_ORIGINS 为你的域名列表（逗号分隔）
    _cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
ai = AIService()
dictionary = DictionaryService()
vocab = VocabService()

# Mock In-memory database for feedback
FEEDBACKS = []

supabase_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if SUPABASE_ENABLED else None
supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY) else None

SRS_INTERVAL_DAYS = [0, 1, 2, 4, 7, 15, 30]

VOCAB_AUDIO_DIR = (os.getenv("VOCAB_AUDIO_DIR") or "").strip()
VOCAB_AUDIO_BUCKET = (os.getenv("VOCAB_AUDIO_BUCKET") or "").strip()
VOCAB_AUDIO_PREFIX = (os.getenv("VOCAB_AUDIO_PREFIX") or "").strip().strip("/")

# --- Forum moderation (敏感词/外链) ---
FORUM_BLOCKED_WORDS = [w.strip() for w in (os.getenv("FORUM_BLOCKED_WORDS") or "").split(",") if w.strip()]
if not FORUM_BLOCKED_WORDS:
    # 轻量默认词表：上线后建议你在 Render 用环境变量自定义（逗号分隔）
    FORUM_BLOCKED_WORDS = [
        "约炮", "招嫖", "裸聊", "成人视频", "未成年", "毒品", "冰毒", "大麻", "博彩", "赌博",
        "代开发票", "办证", "枪支", "爆炸物", "恐怖", "极端", "仇恨",
    ]

FORUM_ALLOW_LINK_DOMAINS = [d.strip().lower() for d in (os.getenv("FORUM_ALLOW_LINK_DOMAINS") or "").split(",") if d.strip()]
# 默认：不允许外链（最安全）。如需允许自己的域名，设置 FORUM_ALLOW_LINK_DOMAINS=yourdomain.com


def _normalize_cn_text(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"[\s\r\n\t\-_，。,.!！?？;；:：/\\|@#￥$%^&*()（）\[\]{}<>《》\"“”'‘’]+", "", t)
    return t


def _contains_blocked_words(s: str) -> bool:
    norm = _normalize_cn_text(s)
    for w in FORUM_BLOCKED_WORDS:
        if not w:
            continue
        if w.lower() in norm:
            return True
    return False


_URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)


def _extract_urls(text: str) -> List[str]:
    return [m.group(1) for m in _URL_RE.finditer(text or "")]


def _is_allowed_url(u: str) -> bool:
    if not FORUM_ALLOW_LINK_DOMAINS:
        return False
    raw = (u or "").strip()
    if raw.lower().startswith("www."):
        raw = "http://" + raw
    try:
        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in FORUM_ALLOW_LINK_DOMAINS)


def _moderate_forum_text(title: str, content: str) -> None:
    combined = f"{title}\n{content}".strip()
    if _contains_blocked_words(combined):
        raise HTTPException(status_code=400, detail="内容包含敏感词，无法发布到论坛。")
    urls = _extract_urls(combined)
    if urls:
        bad = [u for u in urls if not _is_allowed_url(u)]
        if bad:
            raise HTTPException(status_code=400, detail="论坛暂不允许发布不明外链，请删除链接后再发布（如需分享请用文字描述）。")


def _pg_conn():
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    # Supabase 的 pooler（尤其是 transaction/session pooler）可能复用后端连接，
    # 若启用 prepared statements，可能出现 DuplicatePreparedStatement（_pg3_0 already exists）。
    # psycopg3 正确的禁用方式是 prepare_threshold=None（0 反而代表“每次都 prepare”）。
    conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None)

    # 需要兼容 N3/N4 新增字段：pos / frequency / examples
    # 做一次性“自动迁移”，避免上线后列缺失导致查询报错。
    global _VOCAB_LIBRARY_SCHEMA_OK
    if not _VOCAB_LIBRARY_SCHEMA_OK:
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS pos text;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS frequency smallint;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS examples jsonb;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT '{}'::text[];")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vocab_library_tags_gin ON public.vocab_library USING gin(tags);")
            conn.commit()
            _VOCAB_LIBRARY_SCHEMA_OK = True
        except Exception:
            conn.rollback()

    return conn


def init_supabase_schema():
    if not SUPABASE_DB_ENABLED:
        return
    ddl = """
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE IF NOT EXISTS profiles (
      user_id UUID PRIMARY KEY,
      nickname TEXT NOT NULL,
      age INTEGER,
      initial_level TEXT,
      learning_goal TEXT,
      current_level TEXT,
      is_level_public BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS invitation_codes (
      code TEXT PRIMARY KEY,
      is_used BOOLEAN NOT NULL DEFAULT FALSE,
      associated_uid UUID,
      associated_refresh_token TEXT,
      associated_access_token TEXT,
      first_used_at TIMESTAMPTZ,
      expires_at TIMESTAMPTZ,
      ai_daily_limit INT NOT NULL DEFAULT 10,
      is_admin BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS words (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      level TEXT NOT NULL,
      word TEXT NOT NULL,
      kana TEXT,
      meaning_zh TEXT NOT NULL,
      origin TEXT,
      social_targets TEXT[] DEFAULT '{}',
      offense_risk INTEGER DEFAULT 0,
      usage_frequency INTEGER DEFAULT 3,
      scene_tags TEXT[] DEFAULT '{}',
      order_no INTEGER NOT NULL DEFAULT 0,
      register_social TEXT,
      scene_deep_dive TEXT,
      example_ja TEXT,
      example_zh TEXT,
      usage_frequency_note TEXT,
      audio_filename TEXT,
      image_prompt TEXT,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(level, word)
    );

    -- New unified library table (N2/N1/...): search + detail + study source
    CREATE TABLE IF NOT EXISTS vocab_library (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      level TEXT NOT NULL,
      word TEXT NOT NULL,
      reading TEXT,
      meaning TEXT NOT NULL,
      mp3 TEXT,
      social_context JSONB,
      heatmap_data JSONB,
      insight_text TEXT,
      image_url TEXT,
      is_ai_enriched BOOLEAN NOT NULL DEFAULT FALSE,
      order_no INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(level, word)
    );
    CREATE INDEX IF NOT EXISTS idx_vocab_library_word ON vocab_library(word);
    CREATE INDEX IF NOT EXISTS idx_vocab_library_level_order ON vocab_library(level, order_no);

    -- Separate progress table for vocab_library (avoid breaking existing words-based progress)
    CREATE TABLE IF NOT EXISTS library_progress (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID NOT NULL,
      entry_id UUID NOT NULL REFERENCES vocab_library(id) ON DELETE CASCADE,
      level TEXT NOT NULL,
      repetition INTEGER NOT NULL DEFAULT 0,
      interval_days INTEGER NOT NULL DEFAULT 0,
      ease_factor NUMERIC(4,2) NOT NULL DEFAULT 2.50,
      last_result TEXT,
      last_review_at TIMESTAMPTZ,
      next_review_at TIMESTAMPTZ,
      correct_streak INTEGER NOT NULL DEFAULT 0,
      lapse_count INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, entry_id)
    );

    CREATE TABLE IF NOT EXISTS library_plans (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID NOT NULL,
      level TEXT NOT NULL,
      plan_date DATE NOT NULL,
      daily_new_count INTEGER NOT NULL DEFAULT 50,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, level, plan_date)
    );

    CREATE TABLE IF NOT EXISTS user_progress (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID NOT NULL,
      word_id UUID NOT NULL REFERENCES words(id) ON DELETE CASCADE,
      level TEXT NOT NULL,
      repetition INTEGER NOT NULL DEFAULT 0,
      interval_days INTEGER NOT NULL DEFAULT 0,
      ease_factor NUMERIC(4,2) NOT NULL DEFAULT 2.50,
      last_result TEXT,
      last_review_at TIMESTAMPTZ,
      next_review_at TIMESTAMPTZ,
      correct_streak INTEGER NOT NULL DEFAULT 0,
      lapse_count INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, word_id)
    );

    CREATE TABLE IF NOT EXISTS user_plans (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID NOT NULL,
      level TEXT NOT NULL,
      plan_date DATE NOT NULL,
      daily_new_count INTEGER NOT NULL DEFAULT 50,
      daily_review_count INTEGER NOT NULL DEFAULT 0,
      generated_count INTEGER NOT NULL DEFAULT 0,
      estimated_finish_date DATE,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, level, plan_date)
    );

    CREATE TABLE IF NOT EXISTS forum_posts (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID NOT NULL,
      parent_id UUID REFERENCES forum_posts(id) ON DELETE CASCADE,
      title TEXT,
      content TEXT NOT NULL,
      word_id UUID REFERENCES words(id) ON DELETE SET NULL,
      tags TEXT[] DEFAULT '{}',
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS feedbacks (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id UUID,
      category TEXT DEFAULT 'general',
      content TEXT NOT NULL,
      rating INTEGER DEFAULT 5,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
            # 预生成普通邀请码（随机、不可枚举），数量可用环境变量 INVITE_CODE_SEED_COUNT 调整
            seed_count = int(os.getenv("INVITE_CODE_SEED_COUNT") or "200")
            cur.execute("SELECT COUNT(*) FROM invitation_codes WHERE COALESCE(is_admin, FALSE)=FALSE")
            existing = int((cur.fetchone() or [0])[0] or 0)
            need = max(seed_count - existing, 0)
            tries = 0
            inserted = 0
            while inserted < need and tries < need * 20:
                tries += 1
                code = _gen_invite_code(prefix="JP")
                cur.execute(
                    """
                    INSERT INTO invitation_codes (code, is_used, ai_daily_limit, is_admin)
                    VALUES (%s, FALSE, 10, FALSE)
                    ON CONFLICT (code) DO NOTHING
                    """,
                    (code,),
                )
                if getattr(cur, "rowcount", 0) == 1:
                    inserted += 1
            # 站长/作者码：不过期、无限制（通过 is_admin=true 生效）
            cur.execute(
                """
                INSERT INTO invitation_codes (code, is_used, is_admin, ai_daily_limit)
                VALUES ('HAKIMI-ADMIN', FALSE, TRUE, 10)
                ON CONFLICT (code) DO UPDATE SET is_admin=TRUE
                """,
            )
            cur.execute("SELECT COUNT(*) FROM words")
            words_count = int((cur.fetchone() or [0])[0] or 0)
            if words_count == 0:
                order_no = 1
                for lv in ["N5", "N4", "N3", "N2", "N1"]:
                    for item in (vocab.get_list(lv) or []):
                        w = (item.get("word") or "").strip()
                        if not w:
                            continue
                        k = (item.get("kana") or "").strip()
                        m = (item.get("meaning") or "").strip() or "未提供释义"
                        cur.execute(
                            """
                            INSERT INTO words (level, word, kana, meaning_zh, social_targets, offense_risk, usage_frequency, scene_tags, order_no)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (level, word)
                            DO UPDATE SET kana=EXCLUDED.kana, meaning_zh=EXCLUDED.meaning_zh, order_no=EXCLUDED.order_no
                            """,
                            (lv, w, k, m, ["通用"], 0, 3, ["基础"], order_no),
                        )
                        order_no += 1
        conn.commit()
    finally:
        conn.close()


def _ensure_pg_words_extra_columns():
    if not SUPABASE_DB_ENABLED:
        return
    extras = [
        ("register_social", "TEXT"),
        ("scene_deep_dive", "TEXT"),
        ("example_ja", "TEXT"),
        ("example_zh", "TEXT"),
        ("usage_frequency_note", "TEXT"),
        ("audio_filename", "TEXT"),
        ("image_prompt", "TEXT"),
    ]
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            for col, typ in extras:
                cur.execute(f"ALTER TABLE words ADD COLUMN IF NOT EXISTS {col} {typ}")
        conn.commit()
    finally:
        conn.close()


def bootstrap_supabase_data() -> Dict[str, Any]:
    init_supabase_schema()
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM invitation_codes")
            invite_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM words")
            words_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles")
            profiles_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM feedbacks")
            feedback_count = int((cur.fetchone() or {}).get("c") or 0)
            return {
                "invitation_codes": invite_count,
                "words": words_count,
                "profiles": profiles_count,
                "feedbacks": feedback_count,
            }
    finally:
        conn.close()


def _decode_supabase_token(token: str) -> Dict[str, Any]:
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET is not configured")
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid Supabase token")


def _mint_invite_recovery_session(user_id: str, email: str = "") -> Dict[str, Any]:
    """
    当邀请码绑定的 Supabase refresh_token 已失效/被轮换时，
    使用 SUPABASE_JWT_SECRET 为原 UID 签一个后端可识别的恢复会话，
    让老用户仍能进入原账号，不必因为 refresh_token 轮换而被挡在门外。
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET is not configured")
    now = datetime.utcnow()
    exp = now + timedelta(days=30)
    payload = {
        "sub": str(user_id),
        "role": "authenticated",
        "email": email or "",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    access_token = jwt.encode(payload, SUPABASE_JWT_SECRET, algorithm="HS256")
    return {
        "access_token": access_token,
        "refresh_token": "",
        "token_type": "bearer",
        "expires_in": int((exp - now).total_seconds()),
        "user": {
            "id": str(user_id),
            "email": email or "",
        },
        "session_source": "invite_recovery",
    }


def _compute_next_review(repetition: int, rating: str) -> Tuple[int, int, float]:
    repetition = max(0, repetition)
    ease = 2.5
    if rating == "dont_know":
        repetition = 0
    elif rating == "fuzzy":
        repetition = max(0, repetition - 1)
    else:
        repetition = min(repetition + 1, len(SRS_INTERVAL_DAYS) - 1)
    interval_days = SRS_INTERVAL_DAYS[repetition]
    return repetition, interval_days, ease


def _build_daily_task_queue_pg(user_id: str, level: str, daily_new_count: int) -> Dict[str, Any]:
    now_ts = time.time()
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT w.id, w.word, w.kana, w.meaning_zh, w.origin, w.social_targets, w.offense_risk, w.usage_frequency,
                       w.scene_tags, w.register_social, w.scene_deep_dive, w.example_ja, w.example_zh,
                       w.usage_frequency_note, w.audio_filename, w.image_prompt
                FROM user_progress up
                JOIN words w ON w.id = up.word_id
                WHERE up.user_id = %s::uuid
                  AND up.level = %s
                  AND up.next_review_at IS NOT NULL
                  AND up.next_review_at <= NOW()
                ORDER BY up.next_review_at ASC
                LIMIT %s
                """,
                (user_id, level, max(1, int(daily_new_count))),
            )
            review_rows = cur.fetchall()

            cur.execute(
                """
                SELECT w.id, w.word, w.kana, w.meaning_zh, w.origin, w.social_targets, w.offense_risk, w.usage_frequency,
                       w.scene_tags, w.register_social, w.scene_deep_dive, w.example_ja, w.example_zh,
                       w.usage_frequency_note, w.audio_filename, w.image_prompt
                FROM words w
                WHERE w.level = %s
                  AND NOT EXISTS (
                    SELECT 1 FROM user_progress up
                    WHERE up.user_id = %s::uuid AND up.word_id = w.id
                  )
                ORDER BY w.order_no ASC
                LIMIT %s
                """,
                (level, user_id, max(1, int(daily_new_count))),
            )
            new_rows = cur.fetchall()

            queue = []
            for row in review_rows:
                row["kind"] = "review"
                queue.append(row)
            for row in new_rows:
                row["kind"] = "new"
                queue.append(row)

            cur.execute("SELECT COUNT(*) AS c FROM words WHERE level=%s", (level,))
            total_words = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute(
                """
                SELECT COUNT(*) AS c
                FROM user_progress up
                JOIN words w ON w.id = up.word_id
                WHERE up.user_id = %s::uuid AND w.level=%s
                """,
                (user_id, level),
            )
            learned_words = int((cur.fetchone() or {}).get("c") or 0)

            remaining = max(total_words - learned_words, 0)
            days_left = max((remaining + max(1, daily_new_count) - 1) // max(1, daily_new_count), 0)
            return {
                "queue": queue,
                "total": len(queue),
                "due": len(review_rows),
                "new": len(new_rows),
                "remaining_new_words": remaining,
                "estimated_days_left": days_left,
                "generated_at": now_ts,
            }
    finally:
        conn.close()


def _build_daily_task_queue_library_pg(user_id: str, level: str, daily_new_count: int) -> Dict[str, Any]:
    now_ts = time.time()
    selector = _normalize_library_selector(level)
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if _is_kaoyan_selector(selector):
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE 'kaoyan' = ANY(tags)")
            else:
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE level=%s", (selector,))
            total_words = int((cur.fetchone() or {}).get("c") or 0)
            if total_words == 0:
                return {"queue": [], "total": 0, "due": 0, "new": 0, "remaining_new_words": 0, "estimated_days_left": 0, "generated_at": now_ts}

            # due review
            cur.execute(
                """
                SELECT v.id, v.level, v.word, v.reading, v.meaning, v.mp3,
                       v.pos, v.frequency, v.examples,
                       v.social_context, v.heatmap_data, v.insight_text, v.image_url, v.is_ai_enriched, v.order_no
                FROM library_progress lp
                JOIN vocab_library v ON v.id = lp.entry_id
                WHERE lp.user_id = %s::uuid
                  AND lp.level = %s
                  AND lp.next_review_at IS NOT NULL
                  AND lp.next_review_at <= NOW()
                ORDER BY lp.next_review_at ASC
                LIMIT %s
                """,
                (user_id, selector, max(1, int(daily_new_count))),
            )
            review_rows = cur.fetchall()

            if _is_kaoyan_selector(selector):
                cur.execute(
                    """
                    SELECT v.id, v.level, v.word, v.reading, v.meaning, v.mp3,
                           v.pos, v.frequency, v.examples,
                           v.social_context, v.heatmap_data, v.insight_text, v.image_url, v.is_ai_enriched, v.order_no
                    FROM vocab_library v
                    WHERE 'kaoyan' = ANY(v.tags)
                      AND NOT EXISTS (
                        SELECT 1 FROM library_progress lp
                        WHERE lp.user_id = %s::uuid AND lp.entry_id = v.id
                      )
                    ORDER BY v.level ASC, v.order_no ASC
                    LIMIT %s
                    """,
                    (user_id, max(1, int(daily_new_count))),
                )
            else:
                cur.execute(
                    """
                    SELECT v.id, v.level, v.word, v.reading, v.meaning, v.mp3,
                           v.pos, v.frequency, v.examples,
                           v.social_context, v.heatmap_data, v.insight_text, v.image_url, v.is_ai_enriched, v.order_no
                    FROM vocab_library v
                    WHERE v.level = %s
                      AND NOT EXISTS (
                        SELECT 1 FROM library_progress lp
                        WHERE lp.user_id = %s::uuid AND lp.entry_id = v.id
                      )
                    ORDER BY v.order_no ASC
                    LIMIT %s
                    """,
                    (selector, user_id, max(1, int(daily_new_count))),
                )
            new_rows = cur.fetchall()

            queue: List[Dict[str, Any]] = []
            for row in review_rows:
                row["kind"] = "review"
                queue.append(row)
            for row in new_rows:
                row["kind"] = "new"
                queue.append(row)

            if _is_kaoyan_selector(selector):
                cur.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM library_progress lp
                    JOIN vocab_library v ON v.id = lp.entry_id
                    WHERE lp.user_id = %s::uuid AND 'kaoyan' = ANY(v.tags)
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM library_progress lp
                    JOIN vocab_library v ON v.id = lp.entry_id
                    WHERE lp.user_id = %s::uuid AND v.level = %s
                    """,
                    (user_id, selector),
                )
            learned_words = int((cur.fetchone() or {}).get("c") or 0)
            remaining = max(total_words - learned_words, 0)
            days_left = max((remaining + max(1, daily_new_count) - 1) // max(1, daily_new_count), 0)
            return {
                "queue": queue,
                "total": len(queue),
                "due": len(review_rows),
                "new": len(new_rows),
                "remaining_new_words": remaining,
                "estimated_days_left": days_left,
                "generated_at": now_ts,
            }
    finally:
        conn.close()


def _ensure_invitation_codes_extra_columns() -> None:
    """邀请码：支持“首次使用后 7 天内可重复登录，过期失效”"""
    if not SUPABASE_DB_ENABLED:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS first_used_at timestamptz;")
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS expires_at timestamptz;")
        conn.commit()
    finally:
        conn.close()


def _ensure_invitation_codes_limits_columns() -> None:
    if not SUPABASE_DB_ENABLED:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS ai_daily_limit int NOT NULL DEFAULT 10;")
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;")
        conn.commit()
    finally:
        conn.close()


def _ensure_ai_usage_table() -> None:
    if not SUPABASE_DB_ENABLED:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.ai_usage_daily (
                    user_id uuid NOT NULL,
                    day date NOT NULL,
                    cnt int NOT NULL DEFAULT 0,
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, day)
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
async def _startup_init_pg_schema() -> None:
    """
    Render/Supabase 环境中，数据库可能短暂不可用（网络、pooler 熔断、密码更新等）。
    为避免因为“初始化表结构/补列”失败导致整个服务无法启动，这里在启动时尝试初始化，
    但捕获异常并记录日志，让服务先起来（必要时可通过管理员接口再触发 bootstrap）。
    """
    if not SUPABASE_DB_ENABLED:
        return
    try:
        init_supabase_schema()
        _ensure_pg_words_extra_columns()
        _ensure_invitation_codes_extra_columns()
        _ensure_invitation_codes_limits_columns()
        _ensure_ai_usage_table()
        # 预创建用户列表相关表/索引，避免背词时每次同步都跑 DDL 导致“同步中”很久
        try:
            conn = _pg_conn()
            try:
                _ensure_library_user_lists_tables(conn)
            finally:
                conn.close()
        except Exception:
            logging.exception("Ensure library user list tables failed (non-fatal).")
    except Exception:
        logging.exception("Postgres init/ensure failed (non-fatal).")


def _get_user_ai_limit(user_id: str) -> int | None:
    """Return daily limit. None means unlimited."""
    if not SUPABASE_DB_ENABLED:
        return None
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # 同一用户可能关联多个邀请码：只要任意一个 is_admin=true 就视为无限制；
            # 否则使用 ai_daily_limit 的最大值（默认 10）。
            cur.execute(
                """
                SELECT
                  COALESCE(BOOL_OR(is_admin), FALSE) AS is_admin_any,
                  COALESCE(MAX(ai_daily_limit), 10) AS lim
                FROM public.invitation_codes
                WHERE associated_uid=%s::uuid
                """,
                (user_id,),
            )
            row = cur.fetchone() or {}
            if row.get("is_admin_any"):
                return None
            return int(row.get("lim") or 10)
    finally:
        conn.close()


def _assert_ai_quota_available(user_id: str) -> None:
    """Only check quota, don't increment yet (avoid counting failed upstream calls)."""
    limit = _get_user_ai_limit(user_id)
    if limit is None:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT cnt FROM public.ai_usage_daily WHERE user_id=%s::uuid AND day=CURRENT_DATE", (user_id,))
            row = cur.fetchone()
            cnt = int((row or [0])[0] if row else 0)
        if cnt >= limit:
            raise HTTPException(status_code=429, detail=f"今日小雪梨已达到 {limit} 次上限，明天再来问吧。")
    finally:
        conn.close()


def _increment_ai_usage(user_id: str) -> None:
    limit = _get_user_ai_limit(user_id)
    if limit is None:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.ai_usage_daily (user_id, day, cnt)
                VALUES (%s::uuid, CURRENT_DATE, 1)
                ON CONFLICT (user_id, day)
                DO UPDATE SET cnt = public.ai_usage_daily.cnt + 1, updated_at = now()
                """,
                (user_id,),
            )
        conn.commit()
    finally:
        conn.close()


@app.get("/api/config")
async def get_public_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "supabase_enabled": SUPABASE_ENABLED,
    }


@app.get("/api/v2/system/check")
async def system_check_v2():
    return {
        "supabase_enabled": SUPABASE_ENABLED,
        "supabase_db_enabled": SUPABASE_DB_ENABLED,
        "has_supabase_url": bool(SUPABASE_URL),
        "has_supabase_anon_key": bool(SUPABASE_ANON_KEY),
        "has_supabase_jwt_secret": bool(SUPABASE_JWT_SECRET),
        "has_supabase_service_role_key": bool(SUPABASE_SERVICE_ROLE_KEY),
        "has_supabase_db_url": bool(SUPABASE_DB_URL),
    }


@app.post("/api/v2/system/bootstrap")
async def system_bootstrap_v2(x_admin_key: str | None = Header(default=None, alias="x-admin-key")):
    # 初始化数据库属于危险操作：用管理员密钥保护（避免依赖定义顺序问题）
    _require_admin_key(x_admin_key)
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL 未配置，无法初始化 Supabase 数据")
    stats = bootstrap_supabase_data()
    return {"status": "success", "stats": stats}

# --- Auth Routes ---

@app.post("/api/auth/register")
async def register(username: str = Body(...), password: str = Body(...)):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    try:
        hashed_pw = get_password_hash(password)
        c.execute("INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)", 
                  (username, hashed_pw, time.time()))
        conn.commit()
        return {"status": "success"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT hashed_password FROM users WHERE username=?", (form_data.username,))
    row = c.fetchone()
    conn.close()
    
    if not row or not verify_password(form_data.password, row[0]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


# --- Stage 1: Supabase Auth + Plan + Ebbinghaus ---
async def get_current_supabase_user(token: str = Depends(oauth2_scheme)):
    async def _fetch_user():
        if not (SUPABASE_URL and SUPABASE_ANON_KEY):
            return None
        url = f"{SUPABASE_URL}/auth/v1/user"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
            if r.status_code != 200:
                return None
            return r.json()

    user = await _fetch_user()
    if user and user.get("id"):
        return {"id": user["id"], "email": user.get("email")}

    payload = _decode_supabase_token(token)
    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid Supabase session")
    return {"id": user_id, "email": email}


async def get_optional_supabase_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if not credentials or not credentials.credentials:
        return None
    token = credentials.credentials
    try:
        if SUPABASE_URL and SUPABASE_ANON_KEY:
            url = f"{SUPABASE_URL}/auth/v1/user"
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url, headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
                if r.status_code == 200:
                    user = r.json()
                    if user and user.get("id"):
                        return {"id": user["id"], "email": user.get("email")}
        payload = _decode_supabase_token(token)
        user_id = payload.get("sub")
        email = payload.get("email")
        if not user_id:
            return None
        return {"id": user_id, "email": email}
    except Exception:
        return None


def _is_admin_uid(user_id: str) -> bool:
    """站长鉴权：只要该 uid 绑定过任意 is_admin=true 的邀请码，即视为管理员。"""
    if not SUPABASE_DB_ENABLED:
        return False
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(BOOL_OR(is_admin), FALSE) FROM public.invitation_codes WHERE associated_uid=%s::uuid",
                (user_id,),
            )
            return bool((cur.fetchone() or [False])[0])
    finally:
        conn.close()


async def require_admin_user(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    if not _is_admin_uid(current_user["id"]):
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


@app.post("/api/v2/auth/register")
async def register_v2(email: str = Body(...), password: str = Body(...)):
    if not SUPABASE_ENABLED or not supabase_auth:
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    try:
        resp = supabase_auth.auth.sign_up({"email": email, "password": password})
        user = getattr(resp, "user", None)
        return {
            "status": "success",
            "user_id": getattr(user, "id", None),
            "email": getattr(user, "email", email),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Supabase register failed: {e}")


@app.post("/api/v2/auth/login")
async def login_v2(email: str = Body(...), password: str = Body(...)):
    if not SUPABASE_ENABLED or not supabase_auth:
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    try:
        resp = supabase_auth.auth.sign_in_with_password({"email": email, "password": password})
        session = getattr(resp, "session", None)
        if not session or not getattr(session, "access_token", None):
            raise HTTPException(status_code=401, detail="Login failed")
        return {
            "access_token": session.access_token,
            "refresh_token": getattr(session, "refresh_token", None),
            "token_type": "bearer",
            "expires_in": getattr(session, "expires_in", None),
            "user": {
                "id": getattr(resp.user, "id", None) if getattr(resp, "user", None) else None,
                "email": getattr(resp.user, "email", email) if getattr(resp, "user", None) else email,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Supabase login failed: {e}")


@app.post("/api/v2/code-auth/login")
async def code_auth_login_v2(code: str = Body(..., embed=True)):
    code_clean = (code or "").strip().upper()
    if not code_clean:
        raise HTTPException(status_code=400, detail="code is required")
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT code, is_used, associated_uid, associated_refresh_token, associated_access_token, first_used_at, expires_at, is_admin FROM invitation_codes WHERE code=%s",
                (code_clean,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="邀请码不存在")
            if not row["is_used"]:
                return {"status": "new_user", "code": code_clean}
            # 已使用：若未写入过期时间，则回填（兼容历史数据）。管理员码不过期。
            if (not bool(row.get("is_admin") or False)) and row.get("expires_at") is None:
                cur.execute(
                    """
                    UPDATE invitation_codes
                    SET first_used_at=COALESCE(first_used_at, updated_at, created_at, NOW()),
                        expires_at=COALESCE(expires_at, COALESCE(first_used_at, updated_at, created_at, NOW()) + INTERVAL '30 days'),
                        updated_at=NOW()
                    WHERE code=%s
                    """,
                    (code_clean,),
                )
                conn.commit()
                cur.execute(
                    "SELECT code, is_used, associated_uid, associated_refresh_token, associated_access_token, first_used_at, expires_at, is_admin FROM invitation_codes WHERE code=%s",
                    (code_clean,),
                )
                row = cur.fetchone() or row

            # 检查是否过期（首次使用后 30 天；管理员码不过期）
            if (not bool(row.get("is_admin") or False)) and row.get("expires_at") is not None:
                cur.execute("SELECT NOW() > %s::timestamptz AS expired", (row["expires_at"],))
                expired = bool((cur.fetchone() or {}).get("expired") or False)
                if expired:
                    raise HTTPException(status_code=410, detail="邀请码已过期，请联系管理员获取新邀请码")
            associated_uid = str(row.get("associated_uid") or "")
            stored_refresh_token = str(row.get("associated_refresh_token") or "").strip()
            stored_access_token = str(row.get("associated_access_token") or "").strip()
            if not associated_uid:
                raise HTTPException(status_code=409, detail="邀请码已绑定，但缺少用户标识，请联系管理员重置")
    finally:
        conn.close()

    # 走到这里表示“老用户邀请码恢复”
    # 优先在后端直接 refresh，并把轮换后的 refresh_token 写回 DB；
    # 若遇到 Already Used / token_not_found，则自动回退为恢复会话，避免把用户卡死。
    session = None
    session_source = "supabase_refresh"
    refresh_error_detail = ""
    if stored_refresh_token:
        try:
            session = await _supabase_refresh_session(stored_refresh_token)
        except HTTPException as e:
            refresh_error_detail = str(e.detail or "")
            msg = refresh_error_detail.lower()
            if ("already used" in msg) or ("refresh_token_not_found" in msg) or ("invalid refresh token" in msg):
                session = _mint_invite_recovery_session(associated_uid)
                session_source = "invite_recovery"
            else:
                raise
    else:
        session = _mint_invite_recovery_session(associated_uid)
        session_source = "invite_recovery"

    # 若拿到了新的 Supabase session，则把轮换后的 token 回写，避免下次继续踩坑
    if session_source == "supabase_refresh" and session and session.get("refresh_token"):
        conn2 = _pg_conn()
        try:
            with conn2.cursor() as cur2:
                cur2.execute(
                    """
                    UPDATE invitation_codes
                    SET associated_refresh_token=%s,
                        associated_access_token=%s,
                        updated_at=NOW()
                    WHERE code=%s AND associated_uid=%s::uuid
                    """,
                    (session.get("refresh_token"), session.get("access_token") or None, code_clean, associated_uid),
                )
            conn2.commit()
        finally:
            conn2.close()

    # 恢复会话也把 access_token 更新一下，便于站内直接继续用 Bearer token
    if session_source == "invite_recovery" and session and session.get("access_token"):
        conn3 = _pg_conn()
        try:
            with conn3.cursor() as cur3:
                cur3.execute(
                    """
                    UPDATE invitation_codes
                    SET associated_access_token=%s,
                        updated_at=NOW()
                    WHERE code=%s AND associated_uid=%s::uuid
                    """,
                    (session.get("access_token"), code_clean, associated_uid),
                )
            conn3.commit()
        finally:
            conn3.close()

    return {
        "status": "returning_user",
        "code": code_clean,
        "associated_uid": associated_uid,
        "refresh_token": stored_refresh_token,
        "access_token": stored_access_token or (session or {}).get("access_token"),
        "session": session,
        "session_source": session_source,
        "refresh_error_detail": refresh_error_detail or None,
    }


async def _supabase_anonymous_signup() -> Dict[str, Any]:
    """不依赖前端 supabase-js：后端直接调用 Supabase Auth 匿名登录。"""
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    url = f"{SUPABASE_URL}/auth/v1/signup"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.post(url, headers=headers, json={"data": {}})
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase anonymous signup failed: {r.text[:200]}")
    data = r.json() if r.text else {}
    # 返回结构尽量模拟 supabase-js session
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "token_type": data.get("token_type"),
        "expires_in": data.get("expires_in"),
        "user": data.get("user"),
    }


async def _supabase_refresh_session(refresh_token: str) -> Dict[str, Any]:
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    rt = (refresh_token or "").strip()
    if not rt:
        raise HTTPException(status_code=400, detail="refresh_token is required")
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=12) as client:
        r = await client.post(url, headers=headers, json={"refresh_token": rt})
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase refresh failed: {r.text[:200]}")
    data = r.json() if r.text else {}
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "token_type": data.get("token_type"),
        "expires_in": data.get("expires_in"),
        "user": data.get("user"),
    }


@app.post("/api/v2/supabase/anonymous")
async def supabase_anonymous_v2(code: str = Body("", embed=True)):
    """
    仅用于邀请码新用户流程：避免前端依赖 CDN 的 supabase-js 导致“系统初始化中”。
    保护：必须提供一个存在且未使用的邀请码。
    """
    code_clean = (code or "").strip().upper()
    if not code_clean:
        raise HTTPException(status_code=400, detail="code is required")
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT is_used FROM invitation_codes WHERE code=%s", (code_clean,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="邀请码不存在")
            if bool(row.get("is_used") or False):
                raise HTTPException(status_code=409, detail="邀请码已被使用")
    finally:
        conn.close()
    session = await _supabase_anonymous_signup()
    if not session.get("access_token") or not session.get("refresh_token") or not (session.get("user") or {}).get("id"):
        raise HTTPException(status_code=502, detail="Supabase anonymous signup returned invalid session")
    return {"status": "success", "session": session}


@app.post("/api/v2/supabase/refresh")
async def supabase_refresh_v2(refresh_token: str = Body("", embed=True)):
    """前端无 supabase-js 时，用 refresh_token 换取 access_token。"""
    session = await _supabase_refresh_session(refresh_token)
    if not session.get("access_token"):
        raise HTTPException(status_code=502, detail="Supabase refresh returned invalid session")
    return {"status": "success", "session": session}


@app.post("/api/v2/code-auth/complete")
async def code_auth_complete_v2(
    code: str = Body(...),
    user_id: str = Body(...),
    refresh_token: str = Body(...),
    access_token: str = Body(""),
    nickname: str = Body(...),
    age: int = Body(...),
    initial_level: str = Body(...),
    learning_goal: str = Body(...),
):
    code_clean = (code or "").strip().upper()
    nickname_clean = (nickname or "").strip()
    goal_clean = (learning_goal or "").strip()
    if not code_clean or not user_id or not refresh_token or not nickname_clean or not initial_level or not goal_clean:
        raise HTTPException(status_code=400, detail="缺少必要字段")

    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT code, is_used, associated_uid, expires_at, is_admin FROM invitation_codes WHERE code=%s FOR UPDATE", (code_clean,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="邀请码不存在")
            if row["is_used"] and str(row.get("associated_uid") or "") != str(user_id):
                raise HTTPException(status_code=409, detail="邀请码已被使用")
            if (not bool(row.get("is_admin") or False)) and row.get("expires_at") is not None:
                cur.execute("SELECT NOW() > %s::timestamptz AS expired", (row["expires_at"],))
                if bool((cur.fetchone() or {}).get("expired") or False):
                    raise HTTPException(status_code=410, detail="邀请码已过期，请联系管理员获取新邀请码")

            cur.execute(
                """
                UPDATE invitation_codes
                SET is_used=TRUE,
                    associated_uid=%s::uuid,
                    associated_refresh_token=%s,
                    associated_access_token=%s,
                    first_used_at=COALESCE(first_used_at, NOW()),
                    expires_at=CASE WHEN %s THEN expires_at ELSE COALESCE(expires_at, NOW() + INTERVAL '30 days') END,
                    updated_at=NOW()
                WHERE code=%s
                """,
                (user_id, refresh_token, access_token or None, bool(row.get("is_admin") or False), code_clean),
            )

            cur.execute(
                """
                INSERT INTO profiles (user_id, nickname, age, initial_level, learning_goal, current_level, is_level_public, updated_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, TRUE, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                  nickname=EXCLUDED.nickname,
                  age=EXCLUDED.age,
                  initial_level=EXCLUDED.initial_level,
                  learning_goal=EXCLUDED.learning_goal,
                  current_level=COALESCE(profiles.current_level, EXCLUDED.current_level),
                  updated_at=NOW()
                """,
                (user_id, nickname_clean, int(age), initial_level, goal_clean, initial_level),
            )
        conn.commit()
        return {"status": "success", "code": code_clean}
    finally:
        conn.close()


@app.post("/api/v2/code-auth/refresh-token")
async def code_auth_update_refresh_token_v2(
    code: str = Body(...),
    user_id: str = Body(...),
    refresh_token: str = Body(...),
    access_token: str = Body(""),
):
    code_clean = (code or "").strip().upper()
    if not code_clean or not user_id or not refresh_token:
        raise HTTPException(status_code=400, detail="缺少必要字段")
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            # 过期校验
            cur.execute("SELECT expires_at, is_admin FROM invitation_codes WHERE code=%s AND associated_uid=%s::uuid", (code_clean, user_id))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="邀请码不存在或不匹配用户")
            exp = r[0]
            is_admin = bool(r[1] or False)
            if (not is_admin) and exp is None:
                cur.execute(
                    """
                    UPDATE invitation_codes
                    SET first_used_at=COALESCE(first_used_at, updated_at, created_at, NOW()),
                        expires_at=COALESCE(expires_at, COALESCE(first_used_at, updated_at, created_at, NOW()) + INTERVAL '30 days'),
                        updated_at=NOW()
                    WHERE code=%s AND associated_uid=%s::uuid
                    """,
                    (code_clean, user_id),
                )
                conn.commit()
                cur.execute("SELECT expires_at FROM invitation_codes WHERE code=%s AND associated_uid=%s::uuid", (code_clean, user_id))
                exp = (cur.fetchone() or [None])[0]
            if (not is_admin) and exp is not None:
                cur.execute("SELECT NOW() > %s::timestamptz AS expired", (exp,))
                if bool((cur.fetchone() or [False])[0]):
                    raise HTTPException(status_code=410, detail="邀请码已过期，请联系管理员获取新邀请码")
            cur.execute(
                """
                UPDATE invitation_codes
                SET associated_refresh_token=%s,
                    associated_access_token=%s,
                    updated_at=NOW()
                WHERE code=%s AND associated_uid=%s::uuid
                """,
                (refresh_token, access_token or None, code_clean, user_id),
            )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.get("/api/v2/auth/me")
async def me_v2(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    return current_user


@app.get("/api/v2/profile/me")
async def profile_me_v2(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT user_id, nickname, age, initial_level, learning_goal, current_level, is_level_public FROM profiles WHERE user_id=%s::uuid",
                (user_id,),
            )
            row = cur.fetchone()
            return row or {}
    finally:
        conn.close()


@app.post("/api/v2/profile")
async def upsert_profile_v2(
    nickname: str = Body(...),
    age: int = Body(...),
    initial_level: str = Body(...),
    learning_goal: str = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    nickname_clean = nickname.strip()
    if not nickname_clean:
        raise HTTPException(status_code=400, detail="nickname is required")

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profiles (user_id, nickname, age, initial_level, learning_goal, current_level, updated_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                  nickname=EXCLUDED.nickname,
                  age=EXCLUDED.age,
                  initial_level=EXCLUDED.initial_level,
                  learning_goal=EXCLUDED.learning_goal,
                  current_level=COALESCE(profiles.current_level, EXCLUDED.current_level),
                  updated_at=NOW()
                """,
                (user_id, nickname_clean, int(age), initial_level, learning_goal, initial_level),
            )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.post("/api/v2/profile/settings")
async def update_profile_settings_v2(
    is_level_public: bool = Body(...),
    current_level: Optional[str] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            if current_level:
                cur.execute(
                    "UPDATE profiles SET is_level_public=%s, current_level=%s, updated_at=NOW() WHERE user_id=%s::uuid",
                    (bool(is_level_public), current_level, user_id),
                )
            else:
                cur.execute(
                    "UPDATE profiles SET is_level_public=%s, updated_at=NOW() WHERE user_id=%s::uuid",
                    (bool(is_level_public), user_id),
                )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.get("/api/v2/forum/posts")
async def forum_posts_v2(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT p.id, p.user_id, p.title, p.content, p.parent_id, p.created_at,
                       pr.nickname,
                       CASE WHEN pr.is_level_public THEN COALESCE(pr.current_level, pr.initial_level) ELSE NULL END AS level_label
                FROM forum_posts p
                LEFT JOIN profiles pr ON pr.user_id = p.user_id
                WHERE p.parent_id IS NULL
                ORDER BY p.created_at DESC
                LIMIT 100
                """
            )
            posts = cur.fetchall()
            post_ids = [str(p["id"]) for p in posts]
            replies_by_parent: Dict[str, List[Dict[str, Any]]] = {}
            if post_ids:
                cur.execute(
                    """
                    SELECT r.id, r.user_id, r.content, r.parent_id, r.created_at,
                           pr.nickname,
                           CASE WHEN pr.is_level_public THEN COALESCE(pr.current_level, pr.initial_level) ELSE NULL END AS level_label
                    FROM forum_posts r
                    LEFT JOIN profiles pr ON pr.user_id = r.user_id
                    WHERE r.parent_id = ANY(%s::uuid[])
                    ORDER BY r.created_at ASC
                    """,
                    (post_ids,),
                )
                for r in cur.fetchall():
                    pid = str(r["parent_id"])
                    replies_by_parent.setdefault(pid, []).append(r)

            for p in posts:
                pid = str(p["id"])
                p["replies"] = replies_by_parent.get(pid, [])
                p["reply_count"] = len(p["replies"])
            return posts
    finally:
        conn.close()


@app.post("/api/v2/forum/posts")
async def forum_create_post_v2(
    title: str = Body(""),
    content: str = Body(...),
    parent_id: Optional[str] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    title_clean = (title or "").strip()
    content_clean = (content or "").strip()
    if not content_clean:
        raise HTTPException(status_code=400, detail="content is required")
    if parent_id is None and not title_clean:
        raise HTTPException(status_code=400, detail="title is required for top-level post")
    _moderate_forum_text(title_clean, content_clean)

    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if parent_id:
                cur.execute(
                    """
                    INSERT INTO forum_posts (user_id, parent_id, content, created_at, updated_at)
                    VALUES (%s::uuid, %s::uuid, %s, NOW(), NOW())
                    RETURNING id
                    """,
                    (user_id, parent_id, content_clean),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO forum_posts (user_id, title, content, created_at, updated_at)
                    VALUES (%s::uuid, %s, %s, NOW(), NOW())
                    RETURNING id
                    """,
                    (user_id, title_clean, content_clean),
                )
            new_id = cur.fetchone()["id"]
        conn.commit()
        return {"status": "success", "id": str(new_id)}
    finally:
        conn.close()


@app.post("/api/v2/feedback")
async def submit_feedback_v2(
    content: str = Body(...),
    category: str = Body("general"),
    rating: int = Body(5),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_supabase_user),
):
    content_clean = (content or "").strip()
    if not content_clean:
        raise HTTPException(status_code=400, detail="content is required")
    user_id = current_user["id"] if current_user else None

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedbacks (user_id, category, content, rating, created_at) VALUES (%s::uuid, %s, %s, %s, NOW())",
                (user_id, category, content_clean, int(rating)),
            )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.get("/api/v2/admin/stats")
async def admin_stats_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM words")
            words_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles")
            users_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM feedbacks")
            feedback_count = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM forum_posts WHERE parent_id IS NULL")
            forum_posts_count = int((cur.fetchone() or {}).get("c") or 0)
        return {
            "words_count": words_count,
            "profiles_count": users_count,
            "feedback_count": feedback_count,
            "forum_posts_count": forum_posts_count,
        }
    finally:
        conn.close()


@app.get("/api/v2/admin/users")
async def admin_users_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT p.user_id, p.nickname, p.age, p.initial_level, p.current_level, p.learning_goal, p.is_level_public, p.created_at
                FROM profiles p
                ORDER BY p.created_at DESC
                LIMIT 300
                """
            )
            return cur.fetchall()
    finally:
        conn.close()


@app.get("/api/v2/admin/feedbacks")
async def admin_feedbacks_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT f.id, f.user_id, f.category, f.content, f.rating, f.created_at, p.nickname
                FROM feedbacks f
                LEFT JOIN profiles p ON p.user_id = f.user_id
                ORDER BY f.created_at DESC
                LIMIT 300
                """
            )
            return cur.fetchall()
    finally:
        conn.close()


@app.get("/api/v2/admin/invitation-codes")
async def admin_invitation_codes_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT code, is_used, is_admin, ai_daily_limit,
                       associated_uid, first_used_at, expires_at,
                       updated_at, created_at
                FROM invitation_codes
                ORDER BY code ASC
                LIMIT 500
                """
            )
            return cur.fetchall()
    finally:
        conn.close()


@app.post("/api/v2/plan")
async def set_plan_v2(
    level: str = Body(...),
    plan_date: str = Body(...),
    daily_new_count: int = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_plans (user_id, level, plan_date, daily_new_count, updated_at)
                VALUES (%s::uuid, %s, %s::date, %s, NOW())
                ON CONFLICT (user_id, level, plan_date)
                DO UPDATE SET daily_new_count=EXCLUDED.daily_new_count, updated_at=NOW()
                """,
                (user_id, level, plan_date, int(daily_new_count)),
            )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.get("/api/v2/tasks")
async def get_tasks_v2(
    level: str = Query(...),
    plan_date: str = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM words WHERE level=%s", (level,))
            total_for_level = int((cur.fetchone() or {}).get("c") or 0)
            if total_for_level == 0:
                raise HTTPException(status_code=404, detail=f"词书 {level} 暂无词条，请先导入该等级词库后再学习")

            cur.execute(
                "SELECT daily_new_count FROM user_plans WHERE user_id=%s::uuid AND level=%s AND plan_date=%s::date",
                (user_id, level, plan_date),
            )
            row = cur.fetchone()
            daily_new_count = int((row or {}).get("daily_new_count") or 50)
            if not row:
                cur.execute(
                    """
                    INSERT INTO user_plans (user_id, level, plan_date, daily_new_count, updated_at)
                    VALUES (%s::uuid, %s, %s::date, %s, NOW())
                    ON CONFLICT (user_id, level, plan_date) DO NOTHING
                    """,
                    (user_id, level, plan_date, daily_new_count),
                )
                conn.commit()
    finally:
        conn.close()

    queue_payload = _build_daily_task_queue_pg(user_id=user_id, level=level, daily_new_count=daily_new_count)
    return {"plan_date": plan_date, "daily_new_count": daily_new_count, **queue_payload}


@app.post("/api/v2/study/rate")
async def rate_v2(
    word_id: str = Body(...),
    level: str = Body(...),
    rating: str = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    if rating not in ("know", "fuzzy", "dont_know"):
        raise HTTPException(status_code=400, detail="rating must be one of know/fuzzy/dont_know")

    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT repetition FROM user_progress WHERE user_id=%s::uuid AND word_id=%s::uuid",
                (user_id, word_id),
            )
            row = cur.fetchone()
            repetition = int((row or {}).get("repetition") or 0)
            repetition, interval_days, ease = _compute_next_review(repetition, rating)
            cur.execute(
                """
                INSERT INTO user_progress (
                  user_id, word_id, level, repetition, interval_days, ease_factor, last_result,
                  last_review_at, next_review_at, correct_streak, lapse_count, updated_at
                )
                VALUES (
                  %s::uuid, %s::uuid, %s, %s, %s, %s, %s,
                  NOW(), NOW() + (%s || ' days')::interval,
                  CASE WHEN %s='know' THEN 1 ELSE 0 END,
                  CASE WHEN %s='dont_know' THEN 1 ELSE 0 END,
                  NOW()
                )
                ON CONFLICT (user_id, word_id)
                DO UPDATE SET
                  repetition=EXCLUDED.repetition,
                  interval_days=EXCLUDED.interval_days,
                  ease_factor=EXCLUDED.ease_factor,
                  last_result=EXCLUDED.last_result,
                  last_review_at=NOW(),
                  next_review_at=NOW() + (EXCLUDED.interval_days || ' days')::interval,
                  correct_streak=CASE WHEN EXCLUDED.last_result='know' THEN user_progress.correct_streak + 1 ELSE 0 END,
                  lapse_count=CASE WHEN EXCLUDED.last_result='dont_know' THEN user_progress.lapse_count + 1 ELSE user_progress.lapse_count END,
                  updated_at=NOW()
                """,
                (user_id, word_id, level, repetition, interval_days, ease, rating, interval_days, rating, rating),
            )
        conn.commit()
        return {"status": "success", "repetition": repetition, "interval_days": interval_days}
    finally:
        conn.close()


# --- Stage 2: Use vocab_library as unified study source ---
@app.post("/api/v3/plan")
async def set_plan_v3(
    level: str = Body(...),
    plan_date: str = Body(...),
    daily_new_count: int = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    selector = _normalize_library_selector(level)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO library_plans (user_id, level, plan_date, daily_new_count, updated_at)
                VALUES (%s::uuid, %s, %s::date, %s, NOW())
                ON CONFLICT (user_id, level, plan_date)
                DO UPDATE SET daily_new_count=EXCLUDED.daily_new_count, updated_at=NOW()
                """,
                (user_id, selector, plan_date, int(daily_new_count)),
            )
        conn.commit()
        return {"status": "success"}
    finally:
        conn.close()


@app.get("/api/v3/tasks")
async def get_tasks_v3(
    level: str = Query(...),
    plan_date: str = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    selector = _normalize_library_selector(level)
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if _is_kaoyan_selector(selector):
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE 'kaoyan' = ANY(tags)")
            else:
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE level=%s", (selector,))
            total_for_level = int((cur.fetchone() or {}).get("c") or 0)
            if total_for_level == 0:
                raise HTTPException(status_code=404, detail=f"词书 {selector} 暂无词条，请先导入该等级词库后再学习")

            cur.execute(
                "SELECT daily_new_count FROM library_plans WHERE user_id=%s::uuid AND level=%s AND plan_date=%s::date",
                (user_id, selector, plan_date),
            )
            row = cur.fetchone()
            daily_new_count = int((row or {}).get("daily_new_count") or 50)
            if not row:
                cur.execute(
                    """
                    INSERT INTO library_plans (user_id, level, plan_date, daily_new_count, updated_at)
                    VALUES (%s::uuid, %s, %s::date, %s, NOW())
                    ON CONFLICT (user_id, level, plan_date) DO NOTHING
                    """,
                    (user_id, selector, plan_date, daily_new_count),
                )
                conn.commit()
    finally:
        conn.close()

    queue_payload = _build_daily_task_queue_library_pg(user_id=user_id, level=selector, daily_new_count=daily_new_count)
    return {"plan_date": plan_date, "daily_new_count": daily_new_count, **queue_payload}


@app.get("/api/v3/forecast")
async def get_forecast_v3(
    level: str = Query(...),
    plan_date: str = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    """
    轻量学习计划预测（不生成学习队列，避免前端频繁修改目标时卡顿）：
    - 返回 remaining_new_words / estimated_days_left / daily_new_count
    """
    user_id = current_user["id"]
    selector = _normalize_library_selector(level)
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if _is_kaoyan_selector(selector):
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE 'kaoyan' = ANY(tags)")
            else:
                cur.execute("SELECT COUNT(*) AS c FROM vocab_library WHERE level=%s", (selector,))
            total_words = int((cur.fetchone() or {}).get("c") or 0)
            if total_words == 0:
                raise HTTPException(status_code=404, detail=f"词书 {selector} 暂无词条，请先导入该等级词库后再学习")

            cur.execute(
                "SELECT daily_new_count FROM library_plans WHERE user_id=%s::uuid AND level=%s AND plan_date=%s::date",
                (user_id, selector, plan_date),
            )
            row = cur.fetchone()
            daily_new_count = int((row or {}).get("daily_new_count") or 50)

            if _is_kaoyan_selector(selector):
                cur.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM library_progress lp
                    JOIN vocab_library v ON v.id = lp.entry_id
                    WHERE lp.user_id = %s::uuid AND 'kaoyan' = ANY(v.tags)
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM library_progress lp
                    JOIN vocab_library v ON v.id = lp.entry_id
                    WHERE lp.user_id = %s::uuid AND v.level = %s
                    """,
                    (user_id, selector),
                )
            learned_words = int((cur.fetchone() or {}).get("c") or 0)

        remaining = max(total_words - learned_words, 0)
        days_left = max((remaining + max(1, daily_new_count) - 1) // max(1, daily_new_count), 0)
        return {
            "level": selector,
            "plan_date": plan_date,
            "daily_new_count": daily_new_count,
            "remaining_new_words": remaining,
            "estimated_days_left": days_left,
        }
    finally:
        conn.close()


@app.post("/api/v3/study/rate")
async def rate_v3(
    entry_id: str = Body(...),
    level: str = Body(...),
    rating: str = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    if rating not in ("know", "fuzzy", "dont_know"):
        raise HTTPException(status_code=400, detail="rating must be one of know/fuzzy/dont_know")

    user_id = current_user["id"]
    selector = _normalize_library_selector(level)
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT repetition FROM library_progress WHERE user_id=%s::uuid AND entry_id=%s::uuid",
                (user_id, entry_id),
            )
            row = cur.fetchone()
            repetition = int((row or {}).get("repetition") or 0)
            repetition, interval_days, ease = _compute_next_review(repetition, rating)
            cur.execute(
                """
                INSERT INTO library_progress (
                  user_id, entry_id, level, repetition, interval_days, ease_factor, last_result,
                  last_review_at, next_review_at, correct_streak, lapse_count, updated_at
                )
                VALUES (
                  %s::uuid, %s::uuid, %s, %s, %s, %s, %s,
                  NOW(), NOW() + (%s || ' days')::interval,
                  CASE WHEN %s='know' THEN 1 ELSE 0 END,
                  CASE WHEN %s='dont_know' THEN 1 ELSE 0 END,
                  NOW()
                )
                ON CONFLICT (user_id, entry_id)
                DO UPDATE SET
                  repetition=EXCLUDED.repetition,
                  interval_days=EXCLUDED.interval_days,
                  ease_factor=EXCLUDED.ease_factor,
                  last_result=EXCLUDED.last_result,
                  last_review_at=NOW(),
                  next_review_at=NOW() + (EXCLUDED.interval_days || ' days')::interval,
                  correct_streak=CASE WHEN EXCLUDED.last_result='know' THEN library_progress.correct_streak + 1 ELSE 0 END,
                  lapse_count=CASE WHEN EXCLUDED.last_result='dont_know' THEN library_progress.lapse_count + 1 ELSE library_progress.lapse_count END,
                  updated_at=NOW()
                """,
                (user_id, entry_id, selector, repetition, interval_days, ease, rating, interval_days, rating, rating),
            )
            # 错词本：只要点了“不认识”，就自动加入（可在错词本里删除）
            if rating == "dont_know":
                cur.execute(
                    """
                    INSERT INTO public.library_wrongbook (user_id, entry_id, level)
                    VALUES (%s::uuid, %s::uuid, %s)
                    ON CONFLICT (user_id, entry_id) DO NOTHING
                    """,
                    (user_id, entry_id, selector),
                )
        conn.commit()
        return {"status": "success", "repetition": repetition, "interval_days": interval_days}
    finally:
        conn.close()


@app.post("/api/v3/study/rate_batch")
async def rate_batch_v3(
    payload: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    """
    批量评分：前端背词时会快速连续点击，逐条提交会导致“同步中”很久。
    这里允许一次提交多条，减少网络往返与 pooler 开销。
    payload: { items: [{entry_id, level, rating, date?}, ...] }
    """
    items = (payload or {}).get("items") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list")
    if len(items) > 60:
        raise HTTPException(status_code=400, detail="items too large")

    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            for it in items:
                if not isinstance(it, dict):
                    continue
                entry_id = str(it.get("entry_id") or "").strip()
                level = _normalize_library_selector(str(it.get("level") or "").strip())
                rating = str(it.get("rating") or "").strip()
                if not entry_id or not level:
                    continue
                if rating not in ("know", "fuzzy", "dont_know"):
                    continue
                cur.execute(
                    "SELECT repetition FROM library_progress WHERE user_id=%s::uuid AND entry_id=%s::uuid",
                    (user_id, entry_id),
                )
                row = cur.fetchone()
                repetition = int((row or {}).get("repetition") or 0)
                repetition, interval_days, ease = _compute_next_review(repetition, rating)
                cur.execute(
                    """
                    INSERT INTO library_progress (
                      user_id, entry_id, level, repetition, interval_days, ease_factor, last_result,
                      last_review_at, next_review_at, correct_streak, lapse_count, updated_at
                    )
                    VALUES (
                      %s::uuid, %s::uuid, %s, %s, %s, %s, %s,
                      NOW(), NOW() + (%s || ' days')::interval,
                      CASE WHEN %s='know' THEN 1 ELSE 0 END,
                      CASE WHEN %s='dont_know' THEN 1 ELSE 0 END,
                      NOW()
                    )
                    ON CONFLICT (user_id, entry_id)
                    DO UPDATE SET
                      repetition=EXCLUDED.repetition,
                      interval_days=EXCLUDED.interval_days,
                      ease_factor=EXCLUDED.ease_factor,
                      last_result=EXCLUDED.last_result,
                      last_review_at=NOW(),
                      next_review_at=NOW() + (EXCLUDED.interval_days || ' days')::interval,
                      correct_streak=CASE WHEN EXCLUDED.last_result='know' THEN library_progress.correct_streak + 1 ELSE 0 END,
                      lapse_count=CASE WHEN EXCLUDED.last_result='dont_know' THEN library_progress.lapse_count + 1 ELSE library_progress.lapse_count END,
                      updated_at=NOW()
                    """,
                    (user_id, entry_id, level, repetition, interval_days, ease, rating, interval_days, rating, rating),
                )
                if rating == "dont_know":
                    cur.execute(
                        """
                        INSERT INTO public.library_wrongbook (user_id, entry_id, level)
                        VALUES (%s::uuid, %s::uuid, %s)
                        ON CONFLICT (user_id, entry_id) DO NOTHING
                        """,
                        (user_id, entry_id, level),
                    )
        conn.commit()
        return {"status": "success", "accepted": len(items)}
    finally:
        conn.close()


def _ensure_library_user_lists_tables(conn) -> None:
    """错词本/收藏夹（均为“用户-词条”关系表）。"""
    global _LIBRARY_USER_LISTS_SCHEMA_OK
    if _LIBRARY_USER_LISTS_SCHEMA_OK:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.library_favorites (
                  user_id uuid NOT NULL,
                  entry_id uuid NOT NULL,
                  level text NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY (user_id, entry_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.library_wrongbook (
                  user_id uuid NOT NULL,
                  entry_id uuid NOT NULL,
                  level text NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY (user_id, entry_id)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS library_favorites_user_level_idx ON public.library_favorites(user_id, level);")
            cur.execute("CREATE INDEX IF NOT EXISTS library_wrongbook_user_level_idx ON public.library_wrongbook(user_id, level);")
        conn.commit()
        _LIBRARY_USER_LISTS_SCHEMA_OK = True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


@app.get("/api/v3/favorites")
async def list_favorites_v3(level: str = Query(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT entry_id FROM public.library_favorites WHERE user_id=%s::uuid AND level=%s",
                (user_id, level),
            )
            rows = cur.fetchall() or []
        return [str(r["entry_id"]) for r in rows if r.get("entry_id")]
    finally:
        conn.close()


@app.get("/api/v3/favorites/items")
async def list_favorites_items_v3(
    level: str = Query(...),
    limit: int = Query(200, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    """收藏夹条目（带词条信息，供前端列表展示）。"""
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT v.id, v.level, v.word, v.reading, v.meaning, v.pos, v.frequency, v.image_url
                FROM public.library_favorites f
                JOIN public.vocab_library v ON v.id = f.entry_id
                WHERE f.user_id=%s::uuid AND f.level=%s
                ORDER BY f.created_at DESC
                LIMIT %s
                """,
                (user_id, level, int(limit)),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d.get("id"))
            out.append(d)
        return out
    finally:
        conn.close()


@app.post("/api/v3/favorites")
async def add_favorite_v3(
    entry_id: str = Body(...),
    level: str = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.library_favorites (user_id, entry_id, level)
                VALUES (%s::uuid, %s::uuid, %s)
                ON CONFLICT (user_id, entry_id) DO NOTHING
                """,
                (user_id, entry_id, level),
            )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.delete("/api/v3/favorites/{entry_id}")
async def remove_favorite_v3(entry_id: str, current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.library_favorites WHERE user_id=%s::uuid AND entry_id=%s::uuid",
                (user_id, entry_id),
            )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/api/v3/wrongbook")
async def list_wrongbook_v3(level: str = Query(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT entry_id FROM public.library_wrongbook WHERE user_id=%s::uuid AND level=%s",
                (user_id, level),
            )
            rows = cur.fetchall() or []
        return [str(r["entry_id"]) for r in rows if r.get("entry_id")]
    finally:
        conn.close()


@app.get("/api/v3/wrongbook/items")
async def list_wrongbook_items_v3(
    level: str = Query(...),
    limit: int = Query(200, ge=1, le=500),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    """错词本条目（带词条信息，供前端列表展示）。"""
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT v.id, v.level, v.word, v.reading, v.meaning, v.pos, v.frequency, v.image_url
                FROM public.library_wrongbook w
                JOIN public.vocab_library v ON v.id = w.entry_id
                WHERE w.user_id=%s::uuid AND w.level=%s
                ORDER BY w.created_at DESC
                LIMIT %s
                """,
                (user_id, level, int(limit)),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d.get("id"))
            out.append(d)
        return out
    finally:
        conn.close()


@app.post("/api/v3/wrongbook")
async def add_wrongbook_v3(
    entry_id: str = Body(...),
    level: str = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.library_wrongbook (user_id, entry_id, level)
                VALUES (%s::uuid, %s::uuid, %s)
                ON CONFLICT (user_id, entry_id) DO NOTHING
                """,
                (user_id, entry_id, level),
            )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.delete("/api/v3/wrongbook/{entry_id}")
async def remove_wrongbook_v3(entry_id: str, current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    user_id = current_user["id"]
    conn = _pg_conn()
    try:
        _ensure_library_user_lists_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.library_wrongbook WHERE user_id=%s::uuid AND entry_id=%s::uuid",
                (user_id, entry_id),
            )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.post("/api/v2/system/seed-demo")
async def seed_demo_data_v2():
    conn = _pg_conn()
    demo_rows = [
        ("N5", "お疲れ様です", "おつかれさまです", "辛苦了", None, ["同事", "上司"], 5, 5, ["职场"], 1),
        ("N5", "すみません", "すみません", "不好意思；劳驾", None, ["店员", "陌生人"], 3, 5, ["服务场景"], 2),
        ("N4", "失礼します", "しつれいします", "打扰了；失礼了", None, ["上司", "客户"], 2, 4, ["商务"], 3),
        ("N4", "グッド", "ぐっど", "好；优秀", "good", ["朋友"], 8, 3, ["口语"], 4),
        ("N3", "配慮", "はいりょ", "关照；体谅", None, ["同事", "客户"], 1, 4, ["商务礼仪"], 5),
    ]
    try:
        with conn.cursor() as cur:
            for level, word, kana, meaning_zh, origin, social_targets, offense_risk, usage_frequency, scene_tags, order_no in demo_rows:
                cur.execute(
                    """
                    INSERT INTO words (level, word, kana, meaning_zh, origin, social_targets, offense_risk, usage_frequency, scene_tags, order_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (level, word) DO UPDATE SET
                      kana=EXCLUDED.kana,
                      meaning_zh=EXCLUDED.meaning_zh,
                      origin=EXCLUDED.origin,
                      social_targets=EXCLUDED.social_targets,
                      offense_risk=EXCLUDED.offense_risk,
                      usage_frequency=EXCLUDED.usage_frequency,
                      scene_tags=EXCLUDED.scene_tags
                    """,
                    (level, word, kana, meaning_zh, origin, social_targets, offense_risk, usage_frequency, scene_tags, order_no),
                )
        conn.commit()
        return {"status": "success", "seeded": len(demo_rows)}
    finally:
        conn.close()

# --- Personalized Vocab Routes ---

@app.get("/api/vocab/random")
async def get_random_vocab(level: str = Query("N5"), username: Optional[str] = None):
    # If logged in, prioritize words user is learning but hasn't mastered
    if username:
        conn = sqlite3.connect('cache.db')
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        user_id_row = c.fetchone()
        if user_id_row:
            user_id = user_id_row[0]
            # Try to find a word that needs review or is new from vocab_bank
            c.execute(
                """SELECT word, kana, meaning, register_social, scene_deep_dive, example_ja, example_zh,
                          usage_frequency_note, audio_filename, image_prompt
                   FROM vocab_bank
                   WHERE level=? AND word NOT IN
                         (SELECT word FROM user_progress WHERE user_id=? AND status='mastered')
                   ORDER BY RANDOM() LIMIT 1""",
                (level, user_id),
            )
            row = c.fetchone()
            if row:
                conn.close()
                return {
                    "word": row[0],
                    "kana": row[1],
                    "meaning": row[2],
                    "meaning_zh": row[2],
                    "register_social": row[3] or "",
                    "scene_deep_dive": row[4] or "",
                    "example_ja": row[5] or "",
                    "example_zh": row[6] or "",
                    "usage_frequency_note": row[7] or "",
                    "audio_filename": row[8] or "",
                    "image_prompt": row[9] or "",
                }
        conn.close()
    
    # Fallback to general random from memory service
    return vocab.get_random(level)

@app.get("/api/vocab/list")
async def get_vocab_list(level: str = Query("N5")):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        """SELECT word, kana, meaning, register_social, scene_deep_dive, example_ja, example_zh,
                  usage_frequency_note, audio_filename, image_prompt
           FROM vocab_bank WHERE level=? ORDER BY order_no ASC""",
        (level,),
    )
    rows = c.fetchall()
    conn.close()
    if rows:
        out = []
        for r in rows:
            m = r[2] or ""
            out.append(
                {
                    "word": r[0],
                    "kana": r[1] or "",
                    "meaning": m,
                    "meaning_zh": m,
                    "register_social": r[3] or "",
                    "scene_deep_dive": r[4] or "",
                    "example_ja": r[5] or "",
                    "example_zh": r[6] or "",
                    "usage_frequency_note": r[7] or "",
                    "audio_filename": r[8] or "",
                    "image_prompt": r[9] or "",
                }
            )
        return out
    return vocab.get_list(level)

@app.post("/api/vocab/progress")
async def update_progress(word: str = Body(...), level: str = Body(...), status: str = Body(...), username: str = Depends(get_current_user)):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    user_id = c.fetchone()[0]
    
    now = time.time()
    # Simple SRS-like review schedule
    next_review = now + (86400 * 3) if status == 'mastered' else now + 3600
    
    c.execute("""INSERT OR REPLACE INTO user_progress 
                 (user_id, word, level, status, last_seen, next_review, review_count)
                 VALUES (?, ?, ?, ?, ?, ?, 
                 COALESCE((SELECT review_count FROM user_progress WHERE user_id=? AND word=?)+1, 1))""",
              (user_id, word, level, status, now, next_review, user_id, word))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/user/goal")
async def set_daily_goal(level: str = Body(...), date: str = Body(...), target_count: int = Body(...), username: str = Depends(get_current_user)):
    user_id = _get_user_id_by_username(username)
    if not user_id:
        raise HTTPException(status_code=400, detail="User not found")

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO daily_goals (user_id, level, date, target_count, done_new, done_review) VALUES (?, ?, ?, ?, COALESCE((SELECT done_new FROM daily_goals WHERE user_id=? AND level=? AND date=?), 0), COALESCE((SELECT done_review FROM daily_goals WHERE user_id=? AND level=? AND date=?), 0))",
        (user_id, level, date, int(target_count), user_id, level, date, user_id, level, date),
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/user/goal")
async def get_daily_goal(level: str = Query(...), date: str = Query(...), username: str = Depends(get_current_user)):
    user_id = _get_user_id_by_username(username)
    if not user_id:
        raise HTTPException(status_code=400, detail="User not found")

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "SELECT target_count, done_new, done_review FROM daily_goals WHERE user_id=? AND level=? AND date=?",
        (user_id, level, date),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return {"target_count": 0, "done_new": 0, "done_review": 0}
    return {"target_count": row[0], "done_new": row[1], "done_review": row[2]}

@app.get("/api/study/queue")
async def get_study_queue(level: str = Query("N5"), target_count: int = Query(50), username: str = Depends(get_current_user)):
    user_id = _get_user_id_by_username(username)
    if not user_id:
        raise HTTPException(status_code=400, detail="User not found")

    now = time.time()
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()

    c.execute(
        "SELECT word FROM user_progress WHERE user_id=? AND level=? AND next_review IS NOT NULL AND next_review<=? ORDER BY next_review ASC LIMIT ?",
        (user_id, level, now, int(target_count)),
    )
    due_words = [r[0] for r in c.fetchall()]

    remaining = max(int(target_count) - len(due_words), 0)
    new_words: List[str] = []
    if remaining > 0:
        c.execute(
            "SELECT word FROM vocab_bank WHERE level=? AND word NOT IN (SELECT word FROM user_progress WHERE user_id=? AND level=?) ORDER BY order_no ASC LIMIT ?",
            (level, user_id, level, remaining),
        )
        new_words = [r[0] for r in c.fetchall()]

    if remaining > 0 and len(new_words) < remaining:
        c.execute(
            "SELECT word FROM user_progress WHERE user_id=? AND level=?",
            (user_id, level),
        )
        seen_words = {r[0] for r in c.fetchall()}
        seen_words.update(due_words)
        seen_words.update(new_words)

        builtin = vocab.get_list(level) or []
        for it in builtin:
            w = (it.get("word") or "").strip()
            if not w or w in seen_words:
                continue
            new_words.append(w)
            seen_words.add(w)
            if len(new_words) >= remaining:
                break

    words = [(w, "review") for w in due_words] + [(w, "new") for w in new_words]
    if not words:
        conn.close()
        return {"queue": [], "total": 0, "due": len(due_words), "new": len(new_words)}

    word_values = [w[0] for w in words]
    placeholders = ",".join(["?"] * len(word_values))
    c.execute(
        f"""SELECT word, kana, meaning, register_social, scene_deep_dive, example_ja, example_zh,
                   usage_frequency_note, audio_filename, image_prompt
            FROM vocab_bank WHERE level=? AND word IN ({placeholders})""",
        tuple([level] + word_values),
    )
    bank = {}
    for r in c.fetchall():
        m = r[2] or ""
        bank[r[0]] = {
            "word": r[0],
            "kana": r[1] or "",
            "meaning": m,
            "meaning_zh": m,
            "register_social": r[3] or "",
            "scene_deep_dive": r[4] or "",
            "example_ja": r[5] or "",
            "example_zh": r[6] or "",
            "usage_frequency_note": r[7] or "",
            "audio_filename": r[8] or "",
            "image_prompt": r[9] or "",
        }
    conn.close()

    builtin_map = {}
    try:
        for it in (vocab.get_list(level) or []):
            w = (it.get("word") or "").strip()
            if not w:
                continue
            builtin_map[w] = {"word": w, "kana": (it.get("kana") or ""), "meaning": (it.get("meaning") or "")}
    except Exception:
        builtin_map = {}

    queue = []
    for word, kind in words:
        if word in bank:
            item = bank[word]
        elif word in builtin_map:
            item = builtin_map[word]
        else:
            item = {"word": word, "kana": "", "meaning": ""}
        item["kind"] = kind
        queue.append(item)

    return {"queue": queue, "total": len(queue), "due": len(due_words), "new": len(new_words)}

@app.post("/api/study/rate")
async def rate_word(word: str = Body(...), level: str = Body(...), rating: str = Body(...), kind: str = Body(...), date: str = Body(...), username: str = Depends(get_current_user)):
    if rating not in ("know", "fuzzy", "dont_know"):
        raise HTTPException(status_code=400, detail="Invalid rating")
    if kind not in ("new", "review"):
        raise HTTPException(status_code=400, detail="Invalid kind")

    user_id = _get_user_id_by_username(username)
    if not user_id:
        raise HTTPException(status_code=400, detail="User not found")

    now = time.time()
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "SELECT next_review, review_count FROM user_progress WHERE user_id=? AND word=?",
        (user_id, word),
    )
    row = c.fetchone()
    prev_count = int(row[1]) if row and row[1] is not None else 0
    interval_index = min(prev_count, len(SRS_INTERVALS_SECONDS) - 1)
    interval_index, next_review = _next_review_from_rating(now, interval_index, rating)
    status_value = "learning"
    if interval_index >= len(SRS_INTERVALS_SECONDS) - 1 and rating == "know":
        status_value = "mastered"

    c.execute(
        "INSERT OR REPLACE INTO user_progress (user_id, word, level, status, last_seen, next_review, review_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, word, level, status_value, now, next_review, interval_index),
    )

    c.execute(
        "INSERT OR IGNORE INTO daily_goals (user_id, level, date, target_count, done_new, done_review) VALUES (?, ?, ?, 0, 0, 0)",
        (user_id, level, date),
    )
    if kind == "new":
        c.execute(
            "UPDATE daily_goals SET done_new = done_new + 1 WHERE user_id=? AND level=? AND date=?",
            (user_id, level, date),
        )
    else:
        c.execute(
            "UPDATE daily_goals SET done_review = done_review + 1 WHERE user_id=? AND level=? AND date=?",
            (user_id, level, date),
        )

    conn.commit()
    conn.close()
    return {"status": "success", "next_review": next_review, "interval_index": interval_index, "progress_status": status_value}

@app.get("/api/user/stats")
async def get_user_stats(username: str = Depends(get_current_user)):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    user_id = c.fetchone()[0]
    
    c.execute("SELECT level, COUNT(*) FROM user_progress WHERE user_id=? AND status='mastered' GROUP BY level", (user_id,))
    mastered = dict(c.fetchall())
    
    c.execute("SELECT level, COUNT(*) FROM user_progress WHERE user_id=? AND status='learning' GROUP BY level", (user_id,))
    learning = dict(c.fetchall())
    
    conn.close()
    return {
        "username": username,
        "stats": {
            "mastered": mastered,
            "learning": learning
        }
    }

@app.get("/api/vocab/audio/{filename}")
async def serve_vocab_audio(filename: str):
    filename = unquote((filename or "").strip())
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(ch) < 32 for ch in filename)
    ):
        raise HTTPException(status_code=404, detail="Invalid audio filename")
    if VOCAB_AUDIO_BUCKET and (supabase_admin or supabase_auth):
        storage = (supabase_admin or supabase_auth).storage
        obj_path = f"{VOCAB_AUDIO_PREFIX}/{filename}" if VOCAB_AUDIO_PREFIX else filename
        try:
            signed = None
            try:
                signed = storage.from_(VOCAB_AUDIO_BUCKET).create_signed_url(obj_path, 60 * 60)
            except Exception:
                signed = None
            if isinstance(signed, dict):
                signed_url = (
                    signed.get("signedURL")
                    or signed.get("signed_url")
                    or signed.get("signedUrl")
                )
                if signed_url:
                    return RedirectResponse(
                        signed_url,
                        status_code=302,
                        headers={"Cache-Control": "public, max-age=3600"},
                    )
        except Exception:
            pass
        try:
            public_url = storage.from_(VOCAB_AUDIO_BUCKET).get_public_url(obj_path)
            if public_url:
                return RedirectResponse(
                    public_url,
                    status_code=302,
                    headers={"Cache-Control": "public, max-age=604800, immutable"},
                )
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Audio file not found in Supabase bucket")


@app.get("/api/vocab/tip")
async def get_vocab_tip(word: str = Query(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    _assert_ai_quota_available(current_user["id"])
    try:
        tip = await ai.get_quick_tip(word)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"tip": tip}

@app.get("/api/vocab/meta")
async def get_vocab_meta(word: str = Query(...), kana: str = Query(""), meaning: str = Query("")):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT meaning_zh, origin FROM vocab_meta_cache WHERE word=?", (word,))
    row = c.fetchone()
    if row and (row[0] or ""):
        conn.close()
        return {"meaning_zh": row[0] or "", "origin": row[1]}
    conn.close()

    def _has_cjk(s: str) -> bool:
        for ch in s:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    meaning_clean = (meaning or "").strip()
    meaning_zh = meaning_clean if _has_cjk(meaning_clean) else ""
    origin = None

    if not meaning_zh or all('\u30a0' <= c <= '\u30ff' or c == 'ー' for c in word):
        meta = await ai.get_vocab_meta(word=word, kana=kana, meaning=meaning_clean)
        meaning_zh = (meta.get("meaning_zh") or meaning_zh or meaning_clean).strip()
        origin = meta.get("origin") or None

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO vocab_meta_cache (word, meaning_zh, origin, updated_at) VALUES (?, ?, ?, ?)",
        (word, meaning_zh, origin, time.time()),
    )
    conn.commit()
    conn.close()
    return {"meaning_zh": meaning_zh, "origin": origin}

@app.get("/api/analyze")
async def analyze(word: str = Query(...)):
    cached = get_cached_result(word)
    if cached:
        return cached

    dict_info = dictionary.lookup(word)
    result = await ai.analyze_word(word, dict_info)
    save_to_cache(word, result)
    return result

@app.get("/api/chat")
async def chat(
    q: str = Query(...),
    word: str = Query(""),
    kana: str = Query(""),
    meaning: str = Query(""),
    level: str = Query(""),
    offense_risk: int = Query(0),
    social_targets: str = Query(""),
    register_social: str = Query(""),
    scene_deep_dive: str = Query(""),
    example_ja: str = Query(""),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    _assert_ai_quota_available(current_user["id"])
    context = ""
    if word:
        # 控制上下文长度，减少模型吞吐，从而更快响应
        def _clip(s: str, n: int = 420) -> str:
            t = (s or "").strip()
            return t[:n] + ("…" if len(t) > n else "")
        context = f"当前单词：{_clip(word,80)}（{_clip(kana,80)}）\n释义：{_clip(meaning,120)}\n级别：{_clip(level,20)}\n失礼风险：{offense_risk}/100\n社交对象：{_clip(social_targets,120)}\n"
        if register_social.strip():
            context += f"语体与使用对象（词书备注）：{_clip(register_social, 420)}\n"
        if scene_deep_dive.strip():
            context += f"场景深度解读：{_clip(scene_deep_dive, 420)}\n"
        if example_ja.strip():
            context += f"参考例句：{_clip(example_ja, 220)}\n"
    prompt = f"{context}\n用户问题：{q}\n请结合当前单词给出简洁、可执行的中文回答。"
    try:
        answer = await ai.chat(prompt)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"answer": answer}


@app.post("/api/chat")
async def chat_post(payload: Dict[str, Any] = Body(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    """
    带上下文的聊天（推荐）：前端传入最近多轮 messages，让模型能“接上下文”。
    payload:
      - q: string（本轮用户问题）
      - messages: [{role: "user"|"assistant", content: string}, ...]（可选）
      - 可选：word/kana/meaning/level/offense_risk/social_targets/register_social/scene_deep_dive/example_ja（用于单词页上下文）
    """
    _assert_ai_quota_available(current_user["id"])

    q = str((payload or {}).get("q") or "").strip()
    msgs_in = (payload or {}).get("messages") or []

    # 规范化历史消息：限制长度，避免吞吐过大导致变慢
    msgs: List[Dict[str, str]] = []
    if isinstance(msgs_in, list):
        for m in msgs_in[-12:]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            msgs.append({"role": role, "content": content[:1200]})

    # 附加“当前词条上下文”（若有）
    word = str((payload or {}).get("word") or "").strip()
    kana = str((payload or {}).get("kana") or "").strip()
    meaning = str((payload or {}).get("meaning") or "").strip()
    level = str((payload or {}).get("level") or "").strip()
    offense_risk = int((payload or {}).get("offense_risk") or 0)
    social_targets = str((payload or {}).get("social_targets") or "").strip()
    register_social = str((payload or {}).get("register_social") or "").strip()
    scene_deep_dive = str((payload or {}).get("scene_deep_dive") or "").strip()
    example_ja = str((payload or {}).get("example_ja") or "").strip()

    context = ""
    if word:
        def _clip(s: str, n: int = 420) -> str:
            t = (s or "").strip()
            return t[:n] + ("…" if len(t) > n else "")
        context = (
            f"当前单词：{_clip(word,80)}（{_clip(kana,80)}）\n"
            f"释义：{_clip(meaning,120)}\n"
            f"级别：{_clip(level,20)}\n"
            f"失礼风险：{offense_risk}/100\n"
            f"社交对象：{_clip(social_targets,120)}\n"
        )
        if register_social:
            context += f"语体与使用对象（词书备注）：{_clip(register_social, 420)}\n"
        if scene_deep_dive:
            context += f"场景深度解读：{_clip(scene_deep_dive, 420)}\n"
        if example_ja:
            context += f"参考例句：{_clip(example_ja, 220)}\n"

    # 把“词条上下文”作为一条用户消息塞到历史最前面（减少模型理解成本）
    if context:
        msgs = [{"role": "user", "content": f"【上下文信息】\n{context}".strip()}] + msgs

    if q:
        # 确保最后一条是 user
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": q})
        else:
            # 若前端已经把本轮问题放进 messages，则不重复追加
            if msgs[-1]["content"] != q:
                msgs.append({"role": "user", "content": q})

    try:
        answer = await ai.chat(q, messages=msgs)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"answer": answer}


@app.get("/api/sentence/analyze")
async def analyze_sentence(q: str = Query(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    """
    句子分析：结构/重点单词/是否自然/改写建议（中文输出）。
    注意：不返回敏感信息，不存储用户输入。
    """
    text = (q or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="q is required")

    # 只分析日语句子：必须包含假名（平假名/片假名）。否则按“普通聊天”处理更合理。
    # 这样可以避免出现“分析中文句子结构”的尴尬体验。
    if not re.search(r"[\u3040-\u30FF]", text):
        raise HTTPException(status_code=400, detail="句子分析仅支持日语句子（需要包含假名）。中文问题请直接问小雪梨即可。")

    prompt = f"""
你是面向中文母语者的日语写作教练。请只分析下面这句日语（可能不标准）。
如果输入里夹杂了中文翻译/解释/提问背景，把中文当作参考即可：不要分析任何中文句子结构，也不要输出“中文主谓宾/中文语法结构”之类内容。

句子：
{text}

按以下结构输出（用中文，条目清晰，简洁但有用）：
1) 句子结构（主语/谓语/修饰关系/关键助词）
2) 重点单词（列 3~8 个：词条 + 简短中文释义）
3) 自然度评分（0~100）+ 简短理由
4) 改写建议（给出 1~3 个更自然的日语改写，并附中文解释）
如果句子本身没问题，也给出更地道的替代表达。
"""
    _assert_ai_quota_available(current_user["id"])
    try:
        # 句子分析是“特定任务”，把 prompt 当作本轮用户输入即可
        answer = await ai.chat(prompt)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"answer": answer}

@app.post("/api/feedback")
async def submit_feedback(data: dict = Body(...)):
    content = (data.get("content") or "").strip()
    user_name = (data.get("user_name") or "anonymous").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedbacks_local (user_name, content, created_at) VALUES (?, ?, ?)",
        (user_name, content, time.time()),
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/feedbacks")
async def get_feedbacks():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT id, user_name, content, created_at FROM feedbacks_local ORDER BY id DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "user_name": r[1], "content": r[2], "timestamp": r[3]} for r in rows]


@app.get("/api/forum/posts")
async def get_forum_posts(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        """
        SELECT p.id, p.user_name, p.title, p.content, p.parent_id, p.created_at,
               (SELECT COUNT(*) FROM forum_posts_local r WHERE r.parent_id = p.id) AS reply_count
        FROM forum_posts_local p
        WHERE p.parent_id IS NULL
        ORDER BY p.id DESC
        LIMIT 200
        """
    )
    posts = c.fetchall()
    data = []
    for p in posts:
        c.execute(
            "SELECT id, user_name, content, created_at FROM forum_posts_local WHERE parent_id=? ORDER BY id ASC",
            (p[0],),
        )
        replies = c.fetchall()
        data.append(
            {
                "id": p[0],
                "user_name": p[1],
                "title": p[2],
                "content": p[3],
                "parent_id": p[4],
                "created_at": p[5],
                "reply_count": p[6],
                "replies": [{"id": r[0], "user_name": r[1], "content": r[2], "created_at": r[3]} for r in replies],
            }
        )
    conn.close()
    return data


@app.post("/api/forum/posts")
async def create_forum_post(data: dict = Body(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    user_name = (data.get("user_name") or "").strip() or (current_user.get("email") or "已登录")
    parent_id = data.get("parent_id")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    if parent_id is None and not title:
        raise HTTPException(status_code=400, detail="title is required for top-level post")
    _moderate_forum_text(title, content)
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO forum_posts_local (user_name, title, content, parent_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_name, title if parent_id is None else None, content, parent_id, time.time()),
    )
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return {"status": "success", "id": new_id}


@app.get("/api/admin/stats")
async def get_admin_stats(x_admin_key: str | None = Header(default=None, alias="x-admin-key")):
    # 旧版管理员接口：保持兼容，但必须使用管理员密钥
    _require_admin_key(x_admin_key)
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM ai_cache")
    total_search = int(c.fetchone()[0] or 0)
    c.execute("SELECT COUNT(*) FROM users")
    users_count = int(c.fetchone()[0] or 0)
    c.execute("SELECT COUNT(*) FROM feedbacks_local")
    feedback_count = int(c.fetchone()[0] or 0)
    c.execute("SELECT COUNT(*) FROM forum_posts_local WHERE parent_id IS NULL")
    forum_posts_count = int(c.fetchone()[0] or 0)
    conn.close()
    return {
        "total_search": total_search,
        "active_users": users_count,
        "feedback_count": feedback_count,
        "forum_posts_count": forum_posts_count,
    }


def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except Exception:
        return False


def _library_fetch_by_slug(conn, slug: str) -> Optional[Dict[str, Any]]:
    if not slug:
        return None
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        if _is_uuid(slug):
            cur.execute(
                """
                SELECT id, level, word, reading, meaning, mp3,
                       pos, frequency, examples,
                       social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no
                FROM vocab_library
                WHERE id=%s::uuid
                """,
                (slug,),
            )
        else:
            # 1) 优先精确匹配：日语原形
            cur.execute(
                """
                SELECT id, level, word, reading, meaning, mp3,
                       pos, frequency, examples,
                       social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no
                FROM vocab_library
                WHERE word=%s
                ORDER BY level DESC, order_no ASC
                LIMIT 1
                """,
                (slug,),
            )
            row = cur.fetchone()
            if not row:
                # 2) 精确匹配：假名（reading）
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning, mp3,
                           pos, frequency, examples,
                           social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no
                    FROM vocab_library
                    WHERE reading=%s
                    ORDER BY level DESC, order_no ASC
                    LIMIT 1
                    """,
                    (slug,),
                )
                row = cur.fetchone()
            if not row:
                # 3) 中文/关键词：匹配释义（meaning）模糊搜索
                like = f"%{slug}%"
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning, mp3,
                           pos, frequency, examples,
                           social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no
                    FROM vocab_library
                    WHERE meaning ILIKE %s
                    ORDER BY level DESC, order_no ASC
                    LIMIT 1
                    """,
                    (like,),
                )
                row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)


def _ensure_vocab_reports_table(conn) -> None:
    """用于站内“单词卡报错”功能（若无表则自动创建）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.vocab_library_reports (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              created_at timestamptz NOT NULL DEFAULT now(),
              status text NOT NULL DEFAULT 'open',
              vocab_id uuid NULL,
              level text NULL,
              word text NULL,
              slug text NULL,
              issue_type text NULL,
              message text NULL,
              page text NULL,
              page_url text NULL,
              user_agent text NULL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS vocab_library_reports_created_at_idx ON public.vocab_library_reports(created_at DESC);"
        )
    conn.commit()


def _ensure_announcements_table(conn) -> None:
    """公告表：用于站长发布通知。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.announcements (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              title text NOT NULL,
              content text NOT NULL,
              is_active boolean NOT NULL DEFAULT true,
              pinned boolean NOT NULL DEFAULT false,
              starts_at timestamptz NULL,
              ends_at timestamptz NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS announcements_active_idx ON public.announcements(is_active);")
        cur.execute("CREATE INDEX IF NOT EXISTS announcements_created_at_idx ON public.announcements(created_at DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS announcements_pinned_idx ON public.announcements(pinned DESC);")
    conn.commit()


def _require_admin_key(x_admin_key: str | None) -> None:
    """简单、可控的管理员鉴权：用环境变量 ADMIN_API_KEY。"""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY 未配置，无法进行管理员写操作")
    if not x_admin_key or x_admin_key.strip() != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="无权限：管理员密钥不正确")


@app.get("/api/announcements")
async def list_announcements(limit: int = Query(5, ge=1, le=20)):
    """前端读取：返回当前有效公告（按 pinned + 时间倒序）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    conn = _pg_conn()
    try:
        _ensure_announcements_table(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT id, created_at, updated_at, title, content, pinned
                FROM public.announcements
                WHERE is_active = true
                  AND (starts_at IS NULL OR starts_at <= now())
                  AND (ends_at IS NULL OR ends_at >= now())
                ORDER BY pinned DESC, created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d.get("id"))
            out.append(d)
        return out
    finally:
        conn.close()


@app.get("/api/admin/announcements")
async def admin_list_announcements(
    limit: int = Query(50, ge=1, le=200),
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
):
    """站长查看公告列表（包含已下架）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    _require_admin_key(x_admin_key)
    conn = _pg_conn()
    try:
        _ensure_announcements_table(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT id, created_at, updated_at, title, content, pinned, is_active
                FROM public.announcements
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d.get("id"))
            out.append(d)
        return out
    finally:
        conn.close()


@app.post("/api/admin/announcements")
async def create_announcement(
    payload: Dict[str, Any] = Body(...),
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
):
    """站长创建公告（管理员密钥保护）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    _require_admin_key(x_admin_key)
    title = str(payload.get("title") or "").strip()
    content = str(payload.get("content") or "").strip()
    pinned = bool(payload.get("pinned") or False)
    is_active = bool(payload.get("is_active") if payload.get("is_active") is not None else True)
    if not title or not content:
        raise HTTPException(status_code=400, detail="title/content 不能为空")

    conn = _pg_conn()
    try:
        _ensure_announcements_table(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                INSERT INTO public.announcements (title, content, pinned, is_active)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at, updated_at, title, content, pinned, is_active
                """,
                (title[:120], content[:5000], pinned, is_active),
            )
            row = cur.fetchone()
        conn.commit()
        out = dict(row) if row else {}
        if out.get("id") is not None:
            out["id"] = str(out.get("id"))
        return {"status": "ok", "data": out}
    finally:
        conn.close()


@app.post("/api/admin/announcements/{ann_id}/toggle")
async def toggle_announcement(
    ann_id: str,
    payload: Dict[str, Any] = Body(...),
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
):
    """上架/下架公告（管理员密钥保护）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    _require_admin_key(x_admin_key)
    active = bool(payload.get("is_active"))
    conn = _pg_conn()
    try:
        _ensure_announcements_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.announcements
                SET is_active=%s, updated_at=now()
                WHERE id=%s::uuid
                """,
                (active, ann_id),
            )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()

@app.post("/api/library/report")
async def create_vocab_report(payload: Dict[str, Any] = Body(...)):
    """
    单词卡报错/反馈（无需登录，便于你快速收集问题）
    """
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    conn = _pg_conn()
    try:
        _ensure_vocab_reports_table(conn)
        vocab_id = payload.get("vocab_id") or None
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.vocab_library_reports
                  (vocab_id, level, word, slug, issue_type, message, page, page_url, user_agent)
                VALUES
                  (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    vocab_id,
                    (payload.get("level") or "")[:20],
                    (payload.get("word") or "")[:80],
                    (payload.get("slug") or "")[:200],
                    (payload.get("issue_type") or "")[:40],
                    (payload.get("message") or "")[:2000],
                    (payload.get("page") or "")[:40],
                    (payload.get("page_url") or "")[:500],
                    (payload.get("user_agent") or "")[:500],
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return {"status": "ok", "id": str(row[0]) if row else None}
    finally:
        conn.close()


@app.get("/api/library/reports")
async def list_vocab_reports(limit: int = Query(50, ge=1, le=200), status: str = Query("open")):
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    conn = _pg_conn()
    try:
        _ensure_vocab_reports_table(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT id, created_at, status, vocab_id, level, word, slug, issue_type, message, page, page_url
                FROM public.vocab_library_reports
                WHERE (%s = '' OR status = %s)
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (status or "", status or "", int(limit)),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            d = dict(r)
            if d.get("id") is not None:
                d["id"] = str(d["id"])
            if d.get("vocab_id") is not None:
                d["vocab_id"] = str(d["vocab_id"])
            out.append(d)
        return out
    finally:
        conn.close()


@app.get("/word/{slug}", response_class=HTMLResponse)
async def word_detail_page(slug: str):
    # Serve dedicated detail page (JS fetches data via /api/library/word)
    try:
        with open("word.html", "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    # 避免 Render / 浏览器缓存旧版页面，导致“库里有字段但前端不显示”
                    "Cache-Control": "no-store, max-age=0",
                },
            )
    except Exception:
        raise HTTPException(status_code=404, detail="word.html not found")


@app.get("/api/library/word")
async def get_library_word(slug: str = Query(...)):
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    conn = _pg_conn()
    try:
        row = _library_fetch_by_slug(conn, slug)
        if not row:
            raise HTTPException(status_code=404, detail="Word not found in vocab_library")
        # Ensure JSON serializable
        payload = {
            "id": str(row.get("id")),
            "level": row.get("level") or "",
            "word": row.get("word") or "",
            "reading": row.get("reading") or "",
            "meaning": row.get("meaning") or "",
            "mp3": row.get("mp3") or "",
            "pos": row.get("pos") or "",
            "frequency": row.get("frequency") or None,
            "examples": row.get("examples") or None,
            "social_context": row.get("social_context") or None,
            "heatmap_data": row.get("heatmap_data") or None,
            "insight_text": row.get("insight_text") or "",
            "image_url": row.get("image_url") or "",
            "is_ai_enriched": bool(row.get("is_ai_enriched") or False),
        }
        return payload
    finally:
        conn.close()


@app.get("/api/library/search")
async def search_library(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    """
    词库搜索：
    - 支持日语：word / reading 精确匹配、模糊匹配
    - 支持中文：meaning ILIKE 模糊匹配
    返回多条候选，供前端展示“选择列表”。
    """
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    qq = (q or "").strip()
    if not qq:
        return []

    conn = _pg_conn()
    try:
        # 性能优化：大型词库上 "%xxx%" 的 ILIKE 会很慢。
        # - 日文/假名：优先精确 + 前缀匹配（可走索引）
        # - 仅当输入较长时才启用包含匹配
        # - 中文释义匹配仅在输入包含中文时启用（否则会扫全表）
        like_any = f"%{qq}%"
        like_prefix = f"{qq}%"
        enable_contains = len(qq) >= 3
        # 纯假名输入：不要做中文释义模糊匹配，否则会引入大量无关候选（例如“いる”不该匹配到“僧”）
        is_kana_only = _is_kana_only(qq)
        enable_meaning = (not is_kana_only) and bool(re.search(r"[\u4e00-\u9fff]", qq))
        fetch_limit = max(int(limit) * 8, 80)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # 新版本会多取 insight_text 用于“去重时选内容最丰富的那条”。
            # 为了兼容老库/临时 schema 不一致，这里若查询失败则自动降级到旧查询。
            try:
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning, mp3,
                           pos, frequency, examples,
                           image_url, is_ai_enriched, order_no,
                           insight_text
                    FROM vocab_library
                    WHERE
                      word = %(q)s
                      OR reading = %(q)s
                      OR word ILIKE %(prefix)s
                      OR reading ILIKE %(prefix)s
                      OR (%(enable_contains)s AND word ILIKE %(like_any)s)
                      OR (%(enable_contains)s AND reading ILIKE %(like_any)s)
                      OR (%(enable_meaning)s AND meaning ILIKE %(like_any)s)
                    ORDER BY
                      CASE
                        WHEN word = %(q)s THEN 0
                        WHEN reading = %(q)s THEN 1
                        WHEN word ILIKE %(prefix)s THEN 2
                        WHEN reading ILIKE %(prefix)s THEN 3
                        WHEN (%(enable_contains)s AND word ILIKE %(like_any)s) THEN 4
                        WHEN (%(enable_contains)s AND reading ILIKE %(like_any)s) THEN 5
                        WHEN (%(enable_meaning)s AND meaning ILIKE %(like_any)s) THEN 6
                        ELSE 9
                      END,
                      level DESC,
                      order_no ASC
                    LIMIT %(limit)s
                    """,
                    {
                        "q": qq,
                        "prefix": like_prefix,
                        "like_any": like_any,
                        "enable_contains": enable_contains,
                        "enable_meaning": enable_meaning,
                        "limit": fetch_limit,
                    },
                )
            except Exception:
                # 前一次执行失败会让 transaction 进入 aborted 状态；必须 rollback 后才能继续执行下一条 SQL
                try:
                    conn.rollback()
                except Exception:
                    pass
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning, mp3,
                           pos, frequency, examples,
                           image_url, is_ai_enriched, order_no
                    FROM vocab_library
                    WHERE
                      word = %(q)s
                      OR reading = %(q)s
                      OR word ILIKE %(prefix)s
                      OR reading ILIKE %(prefix)s
                      OR (%(enable_contains)s AND word ILIKE %(like_any)s)
                      OR (%(enable_contains)s AND reading ILIKE %(like_any)s)
                      OR (%(enable_meaning)s AND meaning ILIKE %(like_any)s)
                    ORDER BY
                      CASE
                        WHEN word = %(q)s THEN 0
                        WHEN reading = %(q)s THEN 1
                        WHEN word ILIKE %(prefix)s THEN 2
                        WHEN reading ILIKE %(prefix)s THEN 3
                        WHEN (%(enable_contains)s AND word ILIKE %(like_any)s) THEN 4
                        WHEN (%(enable_contains)s AND reading ILIKE %(like_any)s) THEN 5
                        WHEN (%(enable_meaning)s AND meaning ILIKE %(like_any)s) THEN 6
                        ELSE 9
                      END,
                      level DESC,
                      order_no ASC
                    LIMIT %(limit)s
                    """,
                    {
                        "q": qq,
                        "prefix": like_prefix,
                        "like_any": like_any,
                        "enable_contains": enable_contains,
                        "enable_meaning": enable_meaning,
                        "limit": fetch_limit,
                    },
                )
            rows = cur.fetchall() or []

        def _build_out(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out_local: List[Dict[str, Any]] = []
            for r in raw_rows:
                d = dict(r)
                d["id"] = str(d.get("id"))
                # 前端用于“是否直接跳转/是否需要候选列表”的判断
                w = str(d.get("word") or "")
                rd = str(d.get("reading") or "")
                if w == qq:
                    d["match_kind"] = "word_exact"
                    d["match_rank"] = 0
                elif rd == qq:
                    d["match_kind"] = "reading_exact"
                    d["match_rank"] = 1
                elif qq.lower() in w.lower():
                    d["match_kind"] = "word_partial"
                    d["match_rank"] = 2
                elif qq.lower() in rd.lower():
                    d["match_kind"] = "reading_partial"
                    d["match_rank"] = 3
                else:
                    d["match_kind"] = "meaning_match"
                    d["match_rank"] = 4
                ex = d.get("examples")
                d["examples_count"] = len(ex) if isinstance(ex, list) else 0
                d.pop("examples", None)
                out_local.append(d)
            return out_local

        # 默认先构造“未去重”的列表（永远不应该 500）
        out = _build_out(rows)

        # 去重：同一个 word 只显示一条，优先“场景深度解析更长”的那条
        try:
            def _lv_rank(lv: str) -> int:
                m = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
                return m.get((lv or "").upper(), 0)

            def _score(d: Dict[str, Any]) -> Tuple[int, int, int, int]:
                ins = str(d.get("insight_text") or "")
                return (
                    len(ins),
                    1 if bool(d.get("is_ai_enriched") or False) else 0,
                    _int0(d.get("frequency")),
                    _lv_rank(str(d.get("level") or "")),
                )

            by_word: Dict[str, Dict[str, Any]] = {}
            for d in out:
                wkey = str(d.get("word") or "").strip()
                if not wkey:
                    continue
                prev = by_word.get(wkey)
                if prev is None or _score(d) > _score(prev):
                    by_word[wkey] = d

            out = list(by_word.values())
            if is_kana_only:
                # 纯假名：优先“最常用”（frequency 高），“要る/居る”应排在前面
                out.sort(
                    key=lambda x: (
                        int(x.get("match_rank") or 9),
                        -_int0(x.get("frequency")),
                        -len(str(x.get("insight_text") or "")),
                        -_lv_rank(str(x.get("level") or "")),
                        str(x.get("word") or ""),
                    )
                )
            else:
                # 非纯假名：优先“内容更丰富的解析”（insight 更长）
                out.sort(
                    key=lambda x: (
                        int(x.get("match_rank") or 9),
                        -len(str(x.get("insight_text") or "")),
                        -_int0(x.get("frequency")),
                        -_lv_rank(str(x.get("level") or "")),
                        str(x.get("word") or ""),
                    )
                )
            out = out[: int(limit)]
        except Exception:
            # 若去重逻辑出错，回退为“不去重”列表（保证不影响可用性）
            out = out[: int(limit)]

        # 列表接口不返回大字段，详情页再取
        for d in out:
            d.pop("scene_deep_dive", None)
            d.pop("insight_text", None)
        return out
    finally:
        conn.close()


@app.get("/api/library/suggest")
async def suggest_library(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=30)):
    """
    实时联想（强相关）：
    - 只返回「以用户输入开头」的候选（word 前缀优先，其次 reading 前缀）
    - 用于前端搜索框下方的实时候选，不做跳转
    """
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    qq = (q or "").strip()
    if not qq:
        return []
    # 简体输入 → 日文汉字（用于前缀命中）
    qq_j = _map_s2j(qq)
    # 用于释义匹配（中文关键词）：只在输入长度 >=2 时启用，避免单字造成大量“弱相关”
    # 纯假名输入：不要做中文释义模糊匹配，否则会引入大量无关候选
    is_kana_only = _is_kana_only(qq)
    enable_meaning = (not is_kana_only) and len(qq) >= 2
    like_cn = f"%{qq}%" if enable_meaning else None
    like_j = f"%{qq_j}%" if enable_meaning else None

    conn = _pg_conn()
    try:
        prefix = f"{qq}%"
        prefix_j = f"{qq_j}%"
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            try:
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning,
                           pos, frequency,
                           image_url, order_no,
                           insight_text, is_ai_enriched
                    FROM vocab_library
                    WHERE
                          word ILIKE %(prefix_j)s
                       OR word ILIKE %(prefix)s
                       OR reading ILIKE %(prefix_j)s
                       OR reading ILIKE %(prefix)s
                       OR (%(enable_meaning)s = true AND meaning ILIKE %(like_cn)s)
                       OR (%(enable_meaning)s = true AND meaning ILIKE %(like_j)s)
                    ORDER BY
                      CASE
                        WHEN word = %(q_j)s THEN 0
                        WHEN word = %(q)s THEN 0
                        WHEN reading = %(q_j)s THEN 1
                        WHEN reading = %(q)s THEN 1
                        WHEN word ILIKE %(prefix_j)s THEN 2
                        WHEN word ILIKE %(prefix)s THEN 2
                        WHEN reading ILIKE %(prefix_j)s THEN 3
                        WHEN reading ILIKE %(prefix)s THEN 3
                        WHEN (%(enable_meaning)s = true AND meaning ILIKE %(like_cn)s) THEN 4
                        WHEN (%(enable_meaning)s = true AND meaning ILIKE %(like_j)s) THEN 4
                        ELSE 9
                      END,
                      frequency DESC NULLS LAST,
                      length(word) ASC,
                      word ASC,
                      level DESC,
                      order_no ASC
                    LIMIT %(limit)s
                    """,
                    {
                        "q": qq,
                        "q_j": qq_j,
                        "prefix": prefix,
                        "prefix_j": prefix_j,
                        "like_cn": like_cn,
                        "like_j": like_j,
                        "enable_meaning": enable_meaning,
                        "limit": int(limit),
                    },
                )
            except Exception:
                # 前一次执行失败会让 transaction 进入 aborted 状态；必须 rollback 后才能继续执行下一条 SQL
                try:
                    conn.rollback()
                except Exception:
                    pass
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning,
                           pos, frequency,
                           image_url, order_no
                    FROM vocab_library
                    WHERE
                          word ILIKE %(prefix_j)s
                       OR word ILIKE %(prefix)s
                       OR reading ILIKE %(prefix_j)s
                       OR reading ILIKE %(prefix)s
                       OR (%(enable_meaning)s = true AND meaning ILIKE %(like_cn)s)
                       OR (%(enable_meaning)s = true AND meaning ILIKE %(like_j)s)
                    ORDER BY
                      CASE
                        WHEN word = %(q_j)s THEN 0
                        WHEN word = %(q)s THEN 0
                        WHEN reading = %(q_j)s THEN 1
                        WHEN reading = %(q)s THEN 1
                        WHEN word ILIKE %(prefix_j)s THEN 2
                        WHEN word ILIKE %(prefix)s THEN 2
                        WHEN reading ILIKE %(prefix_j)s THEN 3
                        WHEN reading ILIKE %(prefix)s THEN 3
                        WHEN (%(enable_meaning)s = true AND meaning ILIKE %(like_cn)s) THEN 4
                        WHEN (%(enable_meaning)s = true AND meaning ILIKE %(like_j)s) THEN 4
                        ELSE 9
                      END,
                      frequency DESC NULLS LAST,
                      length(word) ASC,
                      word ASC,
                      level DESC,
                      order_no ASC
                    LIMIT %(limit)s
                    """,
                    {
                        "q": qq,
                        "q_j": qq_j,
                        "prefix": prefix,
                        "prefix_j": prefix_j,
                        "like_cn": like_cn,
                        "like_j": like_j,
                        "enable_meaning": enable_meaning,
                        "limit": int(limit),
                    },
                )
            rows = cur.fetchall() or []

        def _lv_rank(lv: str) -> int:
            m = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
            return m.get((lv or "").upper(), 0)

        def _score(d: Dict[str, Any]) -> Tuple[int, int, int]:
            ins = str(d.get("insight_text") or "")
            return (len(ins), 1 if bool(d.get("is_ai_enriched") or False) else 0, _lv_rank(str(d.get("level") or "")))

        by_word: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = dict(r)
            d["id"] = str(d.get("id"))
            wkey = str(d.get("word") or "").strip()
            if not wkey:
                continue
            prev = by_word.get(wkey)
            try:
                if prev is None or _score(d) > _score(prev):
                    by_word[wkey] = d
            except Exception:
                # score 失败则保留第一条，避免 suggest 500
                if prev is None:
                    by_word[wkey] = d

        out = list(by_word.values())
        # suggest：纯假名优先常用频次；否则优先深度解析长度
        try:
            if is_kana_only:
                out.sort(key=lambda x: (-_int0(x.get("frequency")), -len(str(x.get("insight_text") or "")), -_lv_rank(str(x.get("level") or "")), str(x.get("word") or "")))
            else:
                out.sort(key=lambda x: (-len(str(x.get("insight_text") or "")), -_int0(x.get("frequency")), -_lv_rank(str(x.get("level") or "")), str(x.get("word") or "")))
        except Exception:
            out.sort(key=lambda x: (str(x.get("word") or ""), -_lv_rank(str(x.get("level") or ""))))
        out = out[: int(limit)]
        for d in out:
            d.pop("insight_text", None)
            d.pop("is_ai_enriched", None)
        return out
    finally:
        conn.close()


@app.post("/api/library/enrich")
async def enrich_library_word(slug: str = Body(..., embed=True)):
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")

    conn = _pg_conn()
    try:
        row = _library_fetch_by_slug(conn, slug)
        if not row:
            raise HTTPException(status_code=404, detail="Word not found in vocab_library")
        if bool(row.get("is_ai_enriched") or False):
            return {
                "status": "ok",
                "data": {
                    "id": str(row.get("id")),
                    "level": row.get("level") or "",
                    "word": row.get("word") or "",
                    "reading": row.get("reading") or "",
                    "meaning": row.get("meaning") or "",
                    "mp3": row.get("mp3") or "",
                    "pos": row.get("pos") or "",
                    "frequency": row.get("frequency") or None,
                    "examples": row.get("examples") or None,
                    "social_context": row.get("social_context") or None,
                    "heatmap_data": row.get("heatmap_data") or None,
                    "insight_text": row.get("insight_text") or "",
                    "image_url": row.get("image_url") or "",
                    "is_ai_enriched": True,
                },
                "skipped": True,
            }

        # AI enrich (image_url intentionally left empty)
        enriched = await ai.enrich_library_entry(
            word=str(row.get("word") or ""),
            reading=str(row.get("reading") or ""),
            meaning=str(row.get("meaning") or ""),
            level=str(row.get("level") or ""),
        )
        if not enriched:
            raise HTTPException(status_code=400, detail="AI enrich unavailable (missing API key) or failed")

        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                UPDATE vocab_library SET
                  social_context=%s::jsonb,
                  heatmap_data=%s::jsonb,
                  insight_text=%s,
                  is_ai_enriched=TRUE
                WHERE id=%s::uuid
                RETURNING id, level, word, reading, meaning, mp3,
                          pos, frequency, examples,
                          social_context, heatmap_data, insight_text, image_url, is_ai_enriched
                """,
                (
                    json.dumps(enriched.get("social_context") or {}, ensure_ascii=False),
                    json.dumps(enriched.get("heatmap_data") or {}, ensure_ascii=False),
                    (enriched.get("insight_text") or "").strip(),
                    str(row.get("id")),
                ),
            )
            updated = cur.fetchone()
        conn.commit()
        out = None
        if updated:
            out = dict(updated)
            out["id"] = str(out.get("id"))
        return {"status": "ok", "data": out, "skipped": False}
    finally:
        conn.close()


@app.get("/api/user/progress-forecast")
async def get_progress_forecast(level: str = Query(...), username: str = Depends(get_current_user)):
    user_id = _get_user_id_by_username(username)
    if not user_id:
        raise HTTPException(status_code=400, detail="User not found")
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM vocab_bank WHERE level=?", (level,))
    total = int(c.fetchone()[0] or 0)
    c.execute("SELECT COUNT(*) FROM user_progress WHERE user_id=? AND level=?", (user_id, level))
    learned = int(c.fetchone()[0] or 0)
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute(
        "SELECT target_count FROM daily_goals WHERE user_id=? AND level=? AND date=?",
        (user_id, level, today),
    )
    row = c.fetchone()
    daily_new = int(row[0]) if row and row[0] else 50
    remaining = max(total - learned, 0)
    days_left = (remaining + max(daily_new, 1) - 1) // max(daily_new, 1)
    est_finish_ts = time.time() + days_left * 86400
    est_finish_date = datetime.fromtimestamp(est_finish_ts).strftime("%Y-%m-%d")
    conn.close()
    return {
        "level": level,
        "total": total,
        "learned": learned,
        "remaining": remaining,
        "daily_new": daily_new,
        "estimated_days_left": int(days_left),
        "estimated_finish_date": est_finish_date,
    }

# --- Page Routes ---

@app.get("/api/healthz")
async def healthz():
    """轻量健康检查：不依赖数据库/外部网络。"""
    return {"ok": True}


def _read_local_file(filename: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def root():
    # Render 健康检查常用 HEAD /，若返回 405 会导致被判定不健康进而 502。
    try:
        return HTMLResponse(
            content=_read_local_file("web.html"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    except FileNotFoundError:
        # 若部署时未包含 web.html，避免请求挂住/连接被重置，直接给出可诊断的 500。
        return HTMLResponse(
            content="web.html not found on server (deploy/build issue).",
            status_code=500,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

@app.api_route("/web.html", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def web_html_alias():
    return await root()

@app.api_route("/study-prototype", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def study_prototype_page():
    try:
        return HTMLResponse(
            content=_read_local_file("study-prototype.html"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    except FileNotFoundError:
        return HTMLResponse(
            content="study-prototype.html not found on server.",
            status_code=404,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

@app.api_route("/study-prototype-v2", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def study_prototype_v2_page():
    try:
        return HTMLResponse(
            content=_read_local_file("study-prototype-v2.html"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    except FileNotFoundError:
        return HTMLResponse(
            content="study-prototype-v2.html not found on server.",
            status_code=404,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return _read_local_file("admin.html")

@app.get("/forum", response_class=HTMLResponse)
async def forum_page():
    return _read_local_file("forum.html")

# --- Safe static file serving (allowlist) ---
# IMPORTANT: 不要在生产环境 mount StaticFiles(directory=".")，否则整个仓库文件都可被下载（包括脚本/CSV/密钥等）。
_ALLOWED_STATIC = {
    "styles.css",
    "arigatou_256.png",
    # 小雪梨头像资源
    "xuexueli_avatar_64.png",
    "xuexueli_idle_64.png",
    "xuexueli_typing_64.png",
    "xuexueli_thinking_64.png",
    "xuexueli_done_64.png",
    "xuexueli_love_64.png",
    "xuexueli_angry_64.png",
    "xuexueli_sad_64.png",
    "xuexueli_study_64.png",
    "xuexueli_shy_64.png",
    "xuexueli_tired_64.png",
}


@app.get("/{filename}")
async def serve_public_file(filename: str):
    if filename not in _ALLOWED_STATIC:
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(os.getcwd(), filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    headers = {"Cache-Control": "public, max-age=3600"}
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        headers = {"Cache-Control": "public, max-age=604800, immutable"}
    return FileResponse(path, headers=headers)

if __name__ == "__main__":
    port = int(os.getenv("PORT") or "8000")
    print(f"Starting Japanese Scene Lab Production Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
