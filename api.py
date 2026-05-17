from fastapi import FastAPI, Query, Body, HTTPException, Depends, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
# AIService 延迟导入，避免 Render 启动超时
# DictionaryService 延迟导入，避免 Render 启动超时
# supabase / jose 延迟导入，见 _get_supabase_* / JWT 函数内部
import httpx
import uvicorn
import os
import logging
import secrets
import time
import threading
import asyncio
import sqlite3
import json
from typing import Optional, List, Dict, Any, Tuple
import csv
import re
import mimetypes
from urllib.parse import urlparse, unquote
import psycopg
import psycopg.rows
try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None
import uuid
import string

# 自动加载 .env 文件（本地开发 / Render 均适用）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip("`")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

# 邀请码恢复会话密钥：优先用配置值，否则用进程随机密钥作为保底
# 老用户的 refresh_token 被 Supabase 轮换后，仍需能通过邀请码恢复登录
_RECOVERY_JWT_SECRET = SUPABASE_JWT_SECRET or secrets.token_hex(32)

SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
SUPABASE_DB_ENABLED = bool(SUPABASE_DB_URL)

# 避免每次请求都重复跑 ALTER TABLE
_VOCAB_LIBRARY_SCHEMA_OK = False  # 首次部署需创建 tags 列
# 避免每次评分/收藏都重复跑 CREATE TABLE/INDEX（会导致同步明显变慢）
_LIBRARY_USER_LISTS_SCHEMA_OK = False

# 管理员写操作密钥（仅后端校验，不要硬编码到前端代码里）
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

# 公开分析接口的轻量限流（进程内）
_PUBLIC_ANALYZE_WINDOW_SECONDS = 60
_PUBLIC_ANALYZE_MAX_REQUESTS = 20
_PUBLIC_ANALYZE_HITS: Dict[str, List[float]] = {}

# 搜索缓存（进程内 TTL）
_search_cache: Dict[Tuple[str, int], Tuple[float, list]] = {}
_SEARCH_CACHE_TTL = 120  # seconds
_SEARCH_CACHE_MAX = 200

# ── HTML 文件缓存（进程生命周期内不变，避免每次请求读磁盘）──
_html_file_cache: Dict[str, str] = {}

# ── 本地持久化缓存（SQLite），重启/关机后秒级恢复，不用重新从 Supabase 加载 20 万条 ──
_VOCAB_SNAPSHOT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_snapshot.db")
_VOCAB_SNAPSHOT_VERSION = 6  # v6: tags 改为逗号分隔字符串（非 JSON），支持 SQLite LIKE 精确过滤

# ── Supabase Storage 快照（Render 冷启动时本地文件丢失，从 Storage 下载恢复）──
_VOCAB_SNAPSHOT_BUCKET = (os.getenv("VOCAB_SNAPSHOT_BUCKET") or "vocab-snapshots").strip()
_VOCAB_SNAPSHOT_KEY = (os.getenv("VOCAB_SNAPSHOT_KEY") or "vocab_snapshot.db").strip()
_VOCAB_STORAGE_DOWNLOAD_TIMEOUT = 15  # Storage 下载超时秒数

# ── 内存词库缓存 + 索引（紧凑格式，5 个 int 字段打包为 1 个，省 ~23MB）──
class _VF:
    ID = 0; LEVEL = 1; WORD = 2; READING = 3; POS = 4
    PACKED = 5; MEANING = 6; TAGS = 7
    # PACKED 编码: freq(4b) | examples_count(10b) | insight_len(12b) | is_ai_enriched(1b) | order_no(5b)

def _pack_ints(freq, ec, il, ai, order):
    f = min(int(freq or 0), 15)
    e = min(int(ec or 0), 1023)
    i = min(int(il or 0), 4095)
    a = 1 if ai else 0
    o = min(int(order or 0), 31)
    return (f << 28) | (e << 18) | (i << 6) | (a << 5) | o

def _unpack(p):
    return (
        (p >> 28) & 0xF,
        (p >> 18) & 0x3FF,
        (p >> 6) & 0xFFF,
        bool((p >> 5) & 0x1),
        p & 0x1F,
    )

# 词库快照（SQLite 直查替代内存缓存，OOM 问题彻底解决）
_snapshot_ready: bool = False
_snapshot_lock = threading.Lock()

# psycopg 连接池（避免每次查询都重新建立连接，Render 上 TLS 握手 5-10s）
_db_pool = None
_db_pool_lock = threading.Lock()
_db_pool_create_failed_at: float = 0  # 建池失败时间戳，30s 内不重试

def _check_pool_conn(conn):
    """轻量活性探测 + 事务清理 — 防止池回收已被 Render/Supabase 杀掉的死连接。"""
    try:
        # 回滚残留事务，避免 INTRANS 错误
        if conn.info and conn.info.transaction_status != 0:  # 0=IDLE
            try:
                conn.rollback()
            except Exception:
                pass
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        raise psycopg.OperationalError("connection check failed")

def _get_db_conn():
    global _db_pool, _db_pool_create_failed_at

    def _configure(conn):
        conn.autocommit = True
        return conn

    # 快速路径：池已存在 → 取一次，失败立即回退直连（避免两次 getconn 等 20s+）
    if _db_pool is not None:
        try:
            return _configure(_db_pool.getconn())
        except Exception:
            return _configure(psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                                   connect_timeout=5))

    # 池模块不可用或 DB 未配置 → 直连
    if ConnectionPool is None or not SUPABASE_DB_ENABLED:
        return _configure(psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                               connect_timeout=5))

    # 池尚未创建 → 加锁创建（带退避：失败 30s 内不重试）
    with _db_pool_lock:
        if _db_pool is None:
            now_s = time.time()
            if now_s - _db_pool_create_failed_at < 30:
                return _configure(psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                                       connect_timeout=5))
            try:
                _db_pool = ConnectionPool(
                    SUPABASE_DB_URL, min_size=1, max_size=4, timeout=8,
                    open=True, check=_check_pool_conn,
                )
                _db_pool_create_failed_at = 0
            except Exception:
                _db_pool_create_failed_at = time.time()
                return _configure(psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                                       connect_timeout=5))
    try:
        return _configure(_db_pool.getconn())
    except Exception:
        return _configure(psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None,
                               connect_timeout=5))

def _return_db_conn(conn):
    global _db_pool
    if _db_pool is not None:
        try:
            _db_pool.putconn(conn)
            return
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass

# ── 词库快照（SQLite 直查，无内存缓存）──

_snapshot_conn_local = threading.local()

def _snapshot_conn():
    """线程本地 SQLite 连接（复用，防止并发搜索各开 8MB 页缓存导致 OOM）。"""
    if not hasattr(_snapshot_conn_local, 'db') or _snapshot_conn_local.db is None:
        _snapshot_conn_local.db = sqlite3.connect(_VOCAB_SNAPSHOT_DB)
        _snapshot_conn_local.db.row_factory = sqlite3.Row
    return _snapshot_conn_local.db

def _snapshot_open():
    """打开快照 SQLite，确认表存在后返回连接。"""
    db = sqlite3.connect(_VOCAB_SNAPSHOT_DB)
    db.row_factory = sqlite3.Row
    return db


def _ensure_snapshot_indexes(db):
    """确保快照表有查询索引（幂等，CREATE IF NOT EXISTS）。"""
    db.execute("CREATE INDEX IF NOT EXISTS idx_snap_word ON vocab_snapshot(word)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_snap_reading ON vocab_snapshot(reading)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_snap_level ON vocab_snapshot(level)")
    db.commit()


def _build_audio_url(mp3: str) -> str:
    """从 mp3 文件名构建完整的音频 URL。"""
    if not mp3:
        return ""
    if mp3.startswith("http"):
        return mp3
    if not SUPABASE_URL or not VOCAB_AUDIO_BUCKET:
        return mp3
    obj_path = f"{VOCAB_AUDIO_PREFIX}/{mp3}" if VOCAB_AUDIO_PREFIX else mp3
    return f"{SUPABASE_URL}/storage/v1/object/public/{VOCAB_AUDIO_BUCKET}/{obj_path}"


def _snapshot_row_to_dict(r) -> dict:
    """将快照 SQLite 行转为与旧 _row_to_dict 兼容的 dict。"""
    rid = str(r["id"])
    freq, ec, il, ai, order = _unpack(r["packed"]) if r["packed"] is not None else (0, 0, 0, False, 0)
    detail = _detail_cache.get(rid)
    mp3 = ""
    if detail is not None and time.time() < detail[0]:
        mp3 = detail[1].get("mp3") or ""
    tags_raw = r["tags"] or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    return {
        "id": rid,
        "level": r["level"] or "",
        "word": r["word"] or "",
        "reading": r["reading"] or "",
        "pos": r["pos"] or "",
        "frequency": freq,
        "examples_count": ec,
        "insight_len": il,
        "is_ai_enriched": ai,
        "order_no": order,
        "meaning": r["meaning"] or "",
        "mp3": mp3,
        "audio_url": _build_audio_url(mp3),
        "image_url": "",
        "tags": tags,
    }


def _ensure_snapshot_available():
    """确保本地快照 SQLite 存在并已建立索引。
    优先本地文件 → 不存在则从 Supabase Storage 下载 → 失败则从 DB 重建。
    不加载全量数据到内存，仅建索引。"""
    global _snapshot_ready
    if _snapshot_ready and os.path.exists(_VOCAB_SNAPSHOT_DB):
        return
    with _snapshot_lock:
        if _snapshot_ready and os.path.exists(_VOCAB_SNAPSHOT_DB):
            return
        # Tier 0: 本地文件存在且有效
        if os.path.exists(_VOCAB_SNAPSHOT_DB):
            try:
                db = _snapshot_open()
                cur = db.execute("SELECT value FROM snapshot_meta WHERE key='version'")
                ver_row = cur.fetchone()
                if ver_row and int(ver_row[0]) == _VOCAB_SNAPSHOT_VERSION:
                    _ensure_snapshot_indexes(db)
                    db.close()
                    _snapshot_ready = True
                    return
                db.close()
            except Exception:
                pass

        # Tier 1: 从 Supabase Storage 下载
        _download_snapshot_from_storage()
        if os.path.exists(_VOCAB_SNAPSHOT_DB):
            try:
                db = _snapshot_open()
                cur = db.execute("SELECT value FROM snapshot_meta WHERE key='version'")
                ver_row = cur.fetchone()
                if ver_row and int(ver_row[0]) == _VOCAB_SNAPSHOT_VERSION:
                    _ensure_snapshot_indexes(db)
                    db.close()
                    _snapshot_ready = True
                    return
                db.close()
            except Exception:
                pass

        # Tier 2: 从 Supabase Storage 下载失败 → 走 DB 直查 fallback（不自动重建，防止 OOM）
        # 手动重建请调用 POST /admin/rebuild-snapshot
        _snapshot_ready = True  # 标记就绪，让请求走 DB fallback
        if not os.path.exists(_VOCAB_SNAPSHOT_DB):
            logging.warning("Snapshot unavailable, using DB fallback for search")


def _rebuild_snapshot_from_db():
    """从 Supabase DB 分批拉取词库写入本地快照 + 上传 Storage。后台线程，内存友好。"""
    global _snapshot_ready
    try:
        conn = psycopg.connect(SUPABASE_DB_URL, prepare_threshold=None, connect_timeout=10)
    except Exception:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
            cur.execute(
                """SELECT id::text, level, word, reading, pos, frequency,
                          COALESCE(CASE WHEN jsonb_typeof(examples) = 'array'
                                   THEN jsonb_array_length(examples) ELSE 0 END, 0),
                          COALESCE(length(insight_text), 0),
                          is_ai_enriched, order_no,
                          COALESCE(meaning, ''),
                          COALESCE(tags, '{}'::text[])
                   FROM vocab_library"""
            )
            # 分批获取，每批 5000 行，避免一次性 load 198K 行进内存
            batch_size = 5000
            first_batch = True
            total_rows = 0
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                _save_local_snapshot(rows, first_batch=first_batch)
                first_batch = False
                total_rows += len(rows)
            if total_rows > 0:
                # 更新 row_count
                lite = sqlite3.connect(_VOCAB_SNAPSHOT_DB)
                lite.execute("INSERT OR REPLACE INTO snapshot_meta VALUES ('row_count', ?)", (str(total_rows),))
                lite.commit()
                lite.close()
                _upload_snapshot_to_storage()
                if os.path.exists(_VOCAB_SNAPSHOT_DB):
                    db = _snapshot_open()
                    _ensure_snapshot_indexes(db)
                    db.close()
                    _snapshot_ready = True
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass



# 详情缓存：避免每次搜索都新建 DB 连接查询 mp3
_detail_cache: Dict[str, tuple] = {}  # id -> (expires_at, {mp3})
_DETAIL_CACHE_TTL = 300
_DETAIL_CACHE_MAX = 1000  # 防止无上限增长（512MB 容器限制）

# 单词详情全量缓存（减少 detail 页 DB 查询）
_word_detail_cache: Dict[str, tuple] = {}  # slug -> (expires_at, dict)
_WORD_DETAIL_CACHE_TTL = 600  # 10分钟，词库数据极少变动
_WORD_DETAIL_CACHE_MAX = 300  # 512MB 容器，防止 OOM

def _trim_word_detail_cache():
    """清理过期的详情缓存条目，防止 OOM"""
    if len(_word_detail_cache) <= _WORD_DETAIL_CACHE_MAX:
        return
    now_s = time.time()
    stale = [k for k, (exp, _) in _word_detail_cache.items() if now_s >= exp]
    for k in stale:
        _word_detail_cache.pop(k, None)
    if len(_word_detail_cache) > _WORD_DETAIL_CACHE_MAX * 2:
        sorted_items = sorted(_word_detail_cache.items(), key=lambda x: x[1][0])
        excess = len(sorted_items) - _WORD_DETAIL_CACHE_MAX
        for k, _ in sorted_items[:excess]:
            _word_detail_cache.pop(k, None)


def _fetch_details(ids: List[str]) -> Dict[str, dict]:
    """批量获取 mp3，优先命中进程内缓存。"""
    if not ids:
        return {}
    now_s = time.time()
    result: Dict[str, dict] = {}
    missing: List[str] = []

    # 先查缓存
    for id_ in ids:
        entry = _detail_cache.get(id_)
        if entry is not None and now_s < entry[0]:
            result[id_] = entry[1]
        else:
            missing.append(id_)

    if not missing:
        return result

    # 缓存未命中，批量查 DB
    try:
        conn = _get_db_conn()
        if conn is None:
            return result
    except Exception:
        return result
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT id::text, mp3 FROM vocab_library WHERE id::text = ANY(%s)",
                (missing,)
            )
            for r in cur.fetchall():
                d = {"mp3": r.get("mp3") or ""}
                rid = str(r["id"])
                result[rid] = d
                _detail_cache[rid] = (now_s + _DETAIL_CACHE_TTL, d)
                if len(_detail_cache) > _DETAIL_CACHE_MAX:
                    # 清理过期条目
                    stale = [k for k, (exp, _) in _detail_cache.items() if now_s >= exp]
                    for k in stale:
                        _detail_cache.pop(k, None)
                    # 保底：超过硬上限驱逐最老条目，防止内存无限增长
                    hard_limit = max(5000, _DETAIL_CACHE_MAX * 2)
                    if len(_detail_cache) > hard_limit:
                        sorted_items = sorted(_detail_cache.items(), key=lambda x: x[1][0])
                        excess = len(sorted_items) - _DETAIL_CACHE_MAX
                        for k, _ in sorted_items[:excess]:
                            _detail_cache.pop(k, None)
    except Exception:
        pass
    finally:
        _return_db_conn(conn)
    return result


def _ensure_storage_bucket() -> bool:
    """确保 Supabase Storage bucket 存在，不存在则创建（需要 service_role key）。"""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False
    try:
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{SUPABASE_URL}/storage/v1/bucket/{_VOCAB_SNAPSHOT_BUCKET}",
                headers=headers)
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                resp = client.post(
                    f"{SUPABASE_URL}/storage/v1/bucket",
                    headers=headers,
                    json={"name": _VOCAB_SNAPSHOT_BUCKET, "id": _VOCAB_SNAPSHOT_BUCKET,
                          "public": True, "file_size_limit": 52428800})
                return resp.status_code in (200, 201)
        return False
    except Exception:
        return False


def _download_snapshot_from_storage() -> Optional[str]:
    """从 Supabase Storage 下载 vocab_snapshot.db 到本地。
    Render 冷启动时本地文件丢失，通过 Storage 恢复，避免全量查询 198K 行导致 OOM。
    返回本地路径成功，任何失败返回 None（上层回退到 DB 直查）。
    注意：需先手动创建 public bucket 并上传快照，此函数无需 service_role key。"""
    if not SUPABASE_URL:
        return None
    try:
        # Public bucket: 使用 /public/ 路径，无需认证
        url = f"{SUPABASE_URL}/storage/v1/object/public/{_VOCAB_SNAPSHOT_BUCKET}/{_VOCAB_SNAPSHOT_KEY}"
        with httpx.stream("GET", url,
                          timeout=_VOCAB_STORAGE_DOWNLOAD_TIMEOUT,
                          follow_redirects=True) as resp:
            if resp.status_code != 200:
                return None
            tmp_path = _VOCAB_SNAPSHOT_DB + ".tmp"
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        os.replace(tmp_path, _VOCAB_SNAPSHOT_DB)
        return _VOCAB_SNAPSHOT_DB
    except Exception:
        return None




def _save_local_snapshot(rows: list, first_batch: bool = True):
    """将 8 列紧凑格式写入本地 SQLite 快照。
    first_batch=True 时重建表，False 时追加写入（用于分批重建）。"""
    try:
        lite = sqlite3.connect(_VOCAB_SNAPSHOT_DB)
        cur = lite.cursor()
        if first_batch:
            cur.execute("DROP TABLE IF EXISTS vocab_snapshot")
            cur.execute("DROP TABLE IF EXISTS snapshot_meta")
            cur.execute("""CREATE TABLE vocab_snapshot (
                id TEXT, level TEXT, word TEXT, reading TEXT, pos TEXT,
                packed INTEGER, meaning TEXT, tags TEXT
            )""")
            cur.execute("CREATE TABLE snapshot_meta (key TEXT PRIMARY KEY, value TEXT)")
            cur.execute("INSERT INTO snapshot_meta VALUES ('loaded_at', ?)", (str(time.time()),))
            cur.execute("INSERT INTO snapshot_meta VALUES ('version', ?)", (str(_VOCAB_SNAPSHOT_VERSION),))

        def _gen():
            for r in rows:
                t = tuple(r) if not isinstance(r, tuple) else r
                if len(t) > 8:
                    packed = _pack_ints(t[5], t[6], t[7], t[8], t[9])
                    t = (t[0], t[1], t[2], t[3], t[4], packed, t[10], t[11])
                tags_val = t[7] if len(t) > 7 else []
                if isinstance(tags_val, list):
                    tags_val = ','.join(str(x) for x in tags_val)
                else:
                    tags_val = str(tags_val) if tags_val else ''
                yield (t[0], t[1], t[2], t[3], t[4], t[5], t[6], tags_val)

        cur.executemany(
            "INSERT INTO vocab_snapshot VALUES (?,?,?,?,?,?,?,?)",
            _gen()
        )
        lite.commit()
        lite.close()
    except Exception:
        pass


def _upload_snapshot_to_storage() -> bool:
    """上传本地 vocab_snapshot.db 到 Supabase Storage。后台线程调用。"""
    if not os.path.exists(_VOCAB_SNAPSHOT_DB):
        return False
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False
    try:
        _ensure_storage_bucket()
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "application/octet-stream",
        }
        url = f"{SUPABASE_URL}/storage/v1/object/{_VOCAB_SNAPSHOT_BUCKET}/{_VOCAB_SNAPSHOT_KEY}"
        with open(_VOCAB_SNAPSHOT_DB, "rb") as f:
            with httpx.Client(timeout=60) as client:
                resp = client.put(url, headers=headers, content=f)
                return resp.status_code in (200, 201, 204)
    except Exception:
        return False




# 简体→日文常用汉字映射（可按需要持续扩充；避免引入冷门第三方库）
# 目标：让用户输入简体（如"强/强化"）也能命中日文词条（如"強/強化"）
S2J_KANJI_MAP: Dict[str, str] = {
    "强": "強",
    "来": "来",  # 兼容：日文常用也是"来"（繁体为"來"）
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
    # 中文动词 → 日文常用汉字
    "吃": "食",
    "喝": "飲",
    "说": "話",
    "看": "見",
    "听": "聞",
    "走": "歩",
    "跑": "走",
    "买": "買",
    "卖": "売",
    "读": "読",
    "写": "書",
    "做": "作",
    "睡": "寝",
    "起": "起",
    "坐": "座",
    "开": "開",
    "关": "閉",
    "进": "入",
    "出": "出",
    "回": "戻",
    "来": "来",
    "去": "行",
    "想": "思",
    "爱": "愛",
    "打": "打",
    "笑": "笑",
    "哭": "泣",
    "死": "死",
    "活": "生",
    "给": "与",
    "送": "送",
    "等": "待",
    "找": "探",
    "用": "使",
    "洗": "洗",
    "切": "切",
    "冷": "寒",
    "热": "暑",
    "忙": "忙",
    "累": "疲",
    "饿": "飢",
    "渴": "渇",
    "辣": "辛",
    "甜": "甘",
    "咸": "塩",
    "鸟": "鳥",
    "鱼": "魚",
    "猫": "猫",
    "狗": "犬",
    "花": "花",
    "药": "薬",
    "钱": "金",
    "饭": "飯",
    "茶": "茶",
    "酒": "酒",
    "肉": "肉",
}


def _map_s2j(text: str) -> str:
    return "".join(S2J_KANJI_MAP.get(ch, ch) for ch in (text or ""))


def _client_ip_key(request: Request) -> str:
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_public_analyze_rate_limit(request: Request) -> None:
    now_ts = time.time()
    ip_key = _client_ip_key(request)
    recent = [
        ts for ts in _PUBLIC_ANALYZE_HITS.get(ip_key, [])
        if now_ts - ts <= _PUBLIC_ANALYZE_WINDOW_SECONDS
    ]
    if len(recent) >= _PUBLIC_ANALYZE_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试。")
    recent.append(now_ts)
    _PUBLIC_ANALYZE_HITS[ip_key] = recent
    # 定期清理过期 IP 条目，防止字典无上限增长
    if len(_PUBLIC_ANALYZE_HITS) > 2000:
        stale = [k for k, hits in list(_PUBLIC_ANALYZE_HITS.items())
                 if not [t for t in hits if now_ts - t <= _PUBLIC_ANALYZE_WINDOW_SECONDS]]
        for k in stale:
            _PUBLIC_ANALYZE_HITS.pop(k, None)


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
    
    c.execute('''CREATE TABLE IF NOT EXISTS vocab_meta_cache
                 (word TEXT PRIMARY KEY,
                  meaning_zh TEXT,
                  origin TEXT,
                  updated_at REAL)''')

    conn.commit()
    _return_db_conn(conn)



# --- Cache Helpers ---
def get_cached_result(word):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT result FROM ai_cache WHERE word=?", (word,))
    row = c.fetchone()
    _return_db_conn(conn)
    return json.loads(row[0]) if row else None

def save_to_cache(word, result):
    if "【系统提示】" in result.get("explanation", ""):
        return
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO ai_cache VALUES (?, ?, ?)", 
              (word, json.dumps(result), time.time()))
    conn.commit()
    _return_db_conn(conn)

app = FastAPI()

# 启动后后台预加载快照（不阻塞启动，失败不影响服务）
@app.on_event("startup")
def _preload_snapshot():
    init_db()
    if SUPABASE_DB_ENABLED:
        threading.Thread(target=_ensure_snapshot_available, daemon=True).start()
        # 预热连接池，避免第一个用户请求等待
        def _warmup_pool():
            try:
                conn = _pg_conn()
                _pg_close(conn)
            except Exception:
                pass
        threading.Thread(target=_warmup_pool, daemon=True).start()

# Enable CORS
_cors_origins_raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if _cors_origins_raw:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
else:
    # 安全默认值：仅允许生产域名与常见本地开发地址。
    _cors_origins = [
        "https://jp-scene-lab.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services (AIService 延迟加载，避免冷启动超时)
ai = None
def get_ai():
    global ai
    if ai is None:
        from ai_service import AIService
        ai = AIService()
    return ai
dictionary = None
def get_dictionary():
    global dictionary
    if dictionary is None:
        from dictionary_service import DictionaryService
        dictionary = DictionaryService()
    return dictionary

# Mock In-memory database for feedback
FEEDBACKS = []

_supabase_auth = None
_supabase_admin = None

def _get_supabase_auth():
    global _supabase_auth
    if _supabase_auth is None and SUPABASE_ENABLED:
        from supabase import create_client  # 延迟导入，避免 Render 启动超时
        _supabase_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_auth

def _get_supabase_admin():
    global _supabase_admin
    if _supabase_admin is None and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        from supabase import create_client  # 延迟导入，避免 Render 启动超时
        _supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_admin

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
    t = re.sub(r"[\s\r\n\t\-_，。,.!！?？;；:：/\\|@#￥$%^&*()（）\[\]{}<>《》\"""''']+", "", t)
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
    conn = _get_db_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    global _VOCAB_LIBRARY_SCHEMA_OK
    if not _VOCAB_LIBRARY_SCHEMA_OK:
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS last_active_at timestamptz;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS pos text;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS frequency smallint;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS examples jsonb;")
                cur.execute("ALTER TABLE public.vocab_library ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT '{}'::text[];")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vocab_library_tags_gin ON public.vocab_library USING gin(tags);")
                # pg_trgm 索引：加速同义词 / 替代表达 的 meaning ILIKE 模糊匹配
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vocab_library_meaning_trgm ON public.vocab_library USING gin (meaning gin_trgm_ops);")
            conn.commit()
            _VOCAB_LIBRARY_SCHEMA_OK = True
        except Exception:
            conn.rollback()

    return conn

def _pg_close(conn):
    _return_db_conn(conn)


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
      last_active_at TIMESTAMPTZ,
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
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE INDEX IF NOT EXISTS idx_vocab_library_word_trgm ON vocab_library USING gin (word gin_trgm_ops);
    CREATE INDEX IF NOT EXISTS idx_vocab_library_reading_trgm ON vocab_library USING gin (reading gin_trgm_ops);
    CREATE INDEX IF NOT EXISTS idx_vocab_library_meaning_trgm ON vocab_library USING gin (meaning gin_trgm_ops);
    -- 搜索专用表达式索引：word + reading + meaning 拼接，单索引替代多列 ILIKE 的 OR 条件
    CREATE INDEX IF NOT EXISTS idx_vocab_library_search_trgm ON public.vocab_library
      USING gin ((COALESCE(word, '') || ' ' || COALESCE(reading, '') || ' ' || COALESCE(meaning, '')) gin_trgm_ops);

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
        conn.commit()
    finally:
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


def _decode_supabase_token(token: str) -> Dict[str, Any]:
    from jose import JWTError, jwt  # 延迟导入，避免 Render 启动超时
    secrets_to_try = []
    if SUPABASE_JWT_SECRET:
        secrets_to_try.append(SUPABASE_JWT_SECRET)
    if _RECOVERY_JWT_SECRET and _RECOVERY_JWT_SECRET not in secrets_to_try:
        secrets_to_try.append(_RECOVERY_JWT_SECRET)
    if not secrets_to_try:
        raise HTTPException(status_code=500, detail="No JWT secret configured")
    for s in secrets_to_try:
        try:
            return jwt.decode(token, s, algorithms=["HS256"], options={"verify_aud": False})
        except JWTError:
            continue
    raise HTTPException(status_code=401, detail="Invalid Supabase token")


def _mint_invite_recovery_session(user_id: str, email: str = "") -> Dict[str, Any]:
    from jose import jwt  # 延迟导入，避免 Render 启动超时
    now = datetime.utcnow()
    exp = now + timedelta(days=30)
    payload = {
        "sub": str(user_id),
        "role": "authenticated",
        "email": email or "",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "session_source": "invite_recovery",
    }
    access_token = jwt.encode(payload, _RECOVERY_JWT_SECRET, algorithm="HS256")
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

def _ensure_invitation_codes_extra_columns() -> None:
    """邀请码：支持'首次使用后 7 天内可重复登录，过期失效'"""
    if not SUPABASE_DB_ENABLED:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS first_used_at timestamptz;")
            cur.execute("ALTER TABLE public.invitation_codes ADD COLUMN IF NOT EXISTS expires_at timestamptz;")
        conn.commit()
    finally:
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


def _sync_background_init() -> None:
    """All blocking DB schema init work. Runs in thread pool to keep event loop free."""
    try:
        init_supabase_schema()
        _ensure_pg_words_extra_columns()
        _ensure_invitation_codes_extra_columns()
        _ensure_invitation_codes_limits_columns()
        _ensure_ai_usage_table()
    except Exception:
        logging.exception("Schema init failed (non-fatal).")

    try:
        try:
            conn = _pg_conn()
            try:
                _ensure_library_user_lists_tables(conn)
            finally:
                _return_db_conn(conn)
        except Exception:
            logging.exception("Ensure library user list tables failed (non-fatal).")

        try:
            conn = _pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE public.profiles p SET last_active_at = sub.new_ts
                        FROM (
                            SELECT
                                p2.user_id,
                                COALESCE(
                                    (SELECT MAX(lp.last_review_at) FROM public.library_progress lp WHERE lp.user_id = p2.user_id),
                                    (SELECT MAX(fp.created_at) FROM public.forum_posts fp WHERE fp.user_id = p2.user_id),
                                    p2.updated_at,
                                    p2.created_at
                                ) AS new_ts
                            FROM public.profiles p2
                            WHERE p2.last_active_at IS NULL
                        ) sub
                        WHERE p.user_id = sub.user_id
                        """
                    )
                conn.commit()
            finally:
                _return_db_conn(conn)
        except Exception:
            logging.exception("Backfill last_active_at failed (non-fatal).")
    except Exception:
        pass


def _get_user_ai_limit(user_id: str) -> Optional[int]:
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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


@app.get("/api/config")
async def get_public_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "supabase_enabled": SUPABASE_ENABLED,
    }


@app.get("/api/v2/system/check")
async def system_check_v2(x_admin_key: Optional[str] = Header(default=None, alias="x-admin-key")):
    _require_admin_key(x_admin_key)
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
async def system_bootstrap_v2(x_admin_key: Optional[str] = Header(default=None, alias="x-admin-key")):
    # 初始化数据库属于危险操作：用管理员密钥保护（避免依赖定义顺序问题）
    _require_admin_key(x_admin_key)
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL 未配置，无法初始化 Supabase 数据")
    stats = bootstrap_supabase_data()
    return {"status": "success", "stats": stats}

# --- Supabase Auth ---

def _touch_user_activity(user_id: str) -> None:
    """轻量更新用户最后活跃时间（静默失败不影响主流程）"""
    if not SUPABASE_DB_ENABLED or not user_id:
        return
    try:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.profiles SET last_active_at=NOW(), updated_at=NOW() WHERE user_id=%s::uuid",
                    (user_id,),
                )
            conn.commit()
        finally:
            _return_db_conn(conn)
    except Exception:
        pass


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
        _touch_user_activity(user["id"])
        return {"id": user["id"], "email": user.get("email")}

    payload = _decode_supabase_token(token)
    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid Supabase session")
    _touch_user_activity(user_id)
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
        _return_db_conn(conn)


async def require_admin_user(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    if not _is_admin_uid(current_user["id"]):
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


@app.post("/api/v2/auth/register")
async def register_v2(email: str = Body(...), password: str = Body(...)):
    if not SUPABASE_ENABLED or not _get_supabase_auth():
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    try:
        resp = _get_supabase_auth().auth.sign_up({"email": email, "password": password})
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
    if not SUPABASE_ENABLED or not _get_supabase_auth():
        raise HTTPException(status_code=500, detail="Supabase Auth not configured")
    try:
        resp = _get_supabase_auth().auth.sign_in_with_password({"email": email, "password": password})
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
        _return_db_conn(conn)

    # 走到这里表示"老用户邀请码恢复"
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
            _return_db_conn(conn2)

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
            _return_db_conn(conn3)

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
    仅用于邀请码新用户流程：避免前端依赖 CDN 的 supabase-js 导致"系统初始化中"。
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
        _return_db_conn(conn)
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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


@app.get("/api/v2/auth/me")
async def me_v2(current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    is_admin = _is_admin_uid(current_user["id"])
    return {**current_user, "is_admin": is_admin}


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '1 day'")
            dau = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '7 days'")
            wau = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '30 days'")
            mau = int((cur.fetchone() or {}).get("c") or 0)
        return {
            "words_count": words_count,
            "profiles_count": users_count,
            "feedback_count": feedback_count,
            "forum_posts_count": forum_posts_count,
            "active_1d": dau,
            "active_7d": wau,
            "active_30d": mau,
        }
    finally:
        _return_db_conn(conn)


@app.get("/api/v2/admin/activity")
async def admin_activity_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    """用户活跃度详情：分层统计 + 最近活跃用户 + 流失用户"""
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM profiles")
            total = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '1 day'")
            active_1d = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '7 days' AND last_active_at < NOW() - INTERVAL '1 day'")
            active_2_7d = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at >= NOW() - INTERVAL '30 days' AND last_active_at < NOW() - INTERVAL '7 days'")
            active_8_30d = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at < NOW() - INTERVAL '30 days'")
            churned = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM profiles WHERE last_active_at IS NULL")
            unknown = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM public.invitation_codes WHERE is_used=TRUE AND COALESCE(is_admin, FALSE)=FALSE")
            codes_used = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute("SELECT COUNT(*) AS c FROM public.invitation_codes WHERE COALESCE(is_admin, FALSE)=FALSE")
            codes_total = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute(
                """
                SELECT p.user_id, p.nickname, p.current_level, p.last_active_at, p.created_at,
                       (SELECT COUNT(*) FROM library_progress lp WHERE lp.user_id=p.user_id) AS study_count
                FROM profiles p
                ORDER BY p.last_active_at DESC NULLS LAST
                LIMIT 30
                """
            )
            recent_users = cur.fetchall()
            cur.execute(
                """
                SELECT p.user_id, p.nickname, p.current_level, p.last_active_at, p.created_at
                FROM profiles p
                WHERE p.last_active_at < NOW() - INTERVAL '30 days'
                ORDER BY p.last_active_at ASC
                LIMIT 20
                """
            )
            churned_users = cur.fetchall()
        return {
            "total": total,
            "active_1d": active_1d,
            "active_2_7d": active_2_7d,
            "active_8_30d": active_8_30d,
            "churned": churned,
            "unknown": unknown,
            "codes_used": codes_used,
            "codes_total": codes_total,
            "recent_users": recent_users,
            "churned_users": churned_users,
        }
    finally:
        _return_db_conn(conn)


@app.get("/api/v2/admin/users")
async def admin_users_v2(current_user: Dict[str, Any] = Depends(require_admin_user)):
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT p.user_id, p.nickname, p.age, p.initial_level, p.current_level,
                       p.learning_goal, p.is_level_public, p.created_at, p.last_active_at
                FROM profiles p
                ORDER BY p.last_active_at DESC NULLS LAST
                LIMIT 300
                """
            )
            return cur.fetchall()
    finally:
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        conn.commit()
        return {"status": "success", "repetition": repetition, "interval_days": interval_days}
    finally:
        _return_db_conn(conn)


@app.post("/api/v3/study/rate_batch")
async def rate_batch_v3(
    payload: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_supabase_user),
):
    """
    批量评分：前端背词时会快速连续点击，逐条提交会导致"同步中"很久。
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
        conn.commit()
        return {"status": "success", "accepted": len(items)}
    finally:
        _return_db_conn(conn)


def _ensure_library_user_lists_tables(conn) -> None:
    """收藏夹（用户-词条关系表）。"""
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
            cur.execute("CREATE INDEX IF NOT EXISTS library_favorites_user_level_idx ON public.library_favorites(user_id, level);")
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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


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
        _return_db_conn(conn)




@app.get("/api/vocab/audio/{filename}")
async def serve_vocab_audio(filename: str):
    filename = unquote((filename or "").strip())
    # 如果前端传了完整 URL，提取末尾文件名
    if filename.startswith("http"):
        filename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(ch) < 32 for ch in filename)
    ):
        raise HTTPException(status_code=404, detail="Invalid audio filename")

    # 直接构造 public URL（纯字符串拼接，不查 DB、不调 API，毫秒级）
    if VOCAB_AUDIO_BUCKET and SUPABASE_URL:
        obj_path = f"{VOCAB_AUDIO_PREFIX}/{filename}" if VOCAB_AUDIO_PREFIX else filename
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{VOCAB_AUDIO_BUCKET}/{obj_path}"
        return RedirectResponse(
            public_url,
            status_code=302,
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/api/vocab/tip")
async def get_vocab_tip(word: str = Query(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    _assert_ai_quota_available(current_user["id"])
    try:
        tip = await get_ai().get_quick_tip(word)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"tip": tip}

@app.get("/api/vocab/meta")
async def get_vocab_meta(word: str = Query(...), kana: str = Query(""), meaning: str = Query("")):
    # 1) check SQLite cache first
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute("SELECT meaning_zh, origin FROM vocab_meta_cache WHERE word=?", (word,))
    row = c.fetchone()
    if row and (row[0] or ""):
        _return_db_conn(conn)
        return {"meaning_zh": row[0] or "", "origin": row[1]}
    _return_db_conn(conn)

    origin = None

    # 2) check Supabase vocab_library
    if SUPABASE_DB_ENABLED:
        try:
            pg = _pg_conn()
            with pg.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(
                    "SELECT meaning, reading FROM vocab_library WHERE word=%s ORDER BY level DESC LIMIT 1",
                    (word,),
                )
                lib_row = cur.fetchone()
            _return_db_conn(pg)
            if lib_row:
                meaning_zh = (lib_row.get("meaning") or "").strip()
                # detect loanword origin for katakana words
                if all('゠' <= c <= 'ヿ' or c == 'ー' for c in word) and meaning_zh:
                    origin = _guess_origin_from_meaning(meaning_zh)
                # cache and return
                conn = sqlite3.connect('cache.db')
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO vocab_meta_cache (word, meaning_zh, origin, updated_at) VALUES (?, ?, ?, ?)",
                    (word, meaning_zh, origin, time.time()),
                )
                conn.commit()
                _return_db_conn(conn)
                return {"meaning_zh": meaning_zh, "origin": origin}
        except Exception:
            pass

    # 3) fallback: AI
    def _has_cjk(s: str) -> bool:
        for ch in s:
            if '一' <= ch <= '鿿':
                return True
        return False

    meaning_clean = (meaning or "").strip()
    meaning_zh = meaning_clean if _has_cjk(meaning_clean) else ""

    if not meaning_zh or all('゠' <= c <= 'ヿ' or c == 'ー' for c in word):
        meta = await get_ai().get_vocab_meta(word=word, kana=kana, meaning=meaning_clean)
        meaning_zh = (meta.get("meaning_zh") or meaning_zh or meaning_clean).strip()
        origin = meta.get("origin") or None

    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO vocab_meta_cache (word, meaning_zh, origin, updated_at) VALUES (?, ?, ?, ?)",
        (word, meaning_zh, origin, time.time()),
    )
    conn.commit()
    _return_db_conn(conn)
    return {"meaning_zh": meaning_zh, "origin": origin}


def _extract_meaning_keywords(meaning: str, *, full: bool = False) -> List[str]:
    """
    从中文释义中提取关键词。
    full=False: 优先主释义，若主释义无有效短关键词则兜底用全释义。
    full=True: 取全部释义。
    """
    if not meaning:
        return []
    meaning = re.sub(r"[（(][^)）]*[)）]", " ", meaning)
    meaning = re.sub(r"[぀-ゟ゠-ヿ]+", " ", meaning)

    def _tokenize(text: str) -> List[str]:
        text = re.sub(r"[。；;]", "，", text)
        parts = re.split(r"[，,、/／\s]+", text)
        kws = []
        for p in parts:
            p = p.strip()
            if p and re.search(r"[一-鿿]", p):
                kws.append(p)
        seen = set()
        result = []
        for kw in sorted(kws, key=len, reverse=True):
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
        return result

    if full:
        return _tokenize(meaning)[:8]

    main_part = re.split(r"[。；;]", meaning)[0].strip()
    main_kws = _tokenize(main_part)
    if not main_kws or all(len(k) > 2 for k in main_kws):
        return _tokenize(meaning)[:5]
    return main_kws[:5]


def _guess_origin_from_meaning(meaning: str) -> Optional[str]:
    """从释义中粗略提取外来语来源；仅用于前端展示，不做强依赖。"""
    parts = meaning.replace("；", ";").split(";")
    first = parts[0].strip() if parts else ""
    if first and all(c.isascii() and (c.isalpha() or c.isspace()) for c in first) and len(first) <= 30:
        return first.lower()
    return None

@app.get("/api/analyze")
async def analyze(word: str = Query(..., min_length=1, max_length=80), request: Request = None):
    word = (word or "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="word is required")
    if len(word) > 80:
        raise HTTPException(status_code=400, detail="word is too long")
    if request is not None:
        _enforce_public_analyze_rate_limit(request)
    cached = get_cached_result(word)
    if cached:
        return cached

    dict_info = get_dictionary().lookup(word)
    result = await get_ai().analyze_word(word, dict_info)
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
        answer = await get_ai().chat(prompt)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"answer": answer}


@app.post("/api/chat")
async def chat_post(payload: Dict[str, Any] = Body(...), current_user: Dict[str, Any] = Depends(get_current_supabase_user)):
    """
    带上下文的聊天（推荐）：前端传入最近多轮 messages，让模型能"接上下文"。
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

    # 附加"当前词条上下文"（若有）
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

    # 把"词条上下文"作为一条用户消息塞到历史最前面（减少模型理解成本）
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
        answer = await get_ai().chat(q, messages=msgs)
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

    # 只分析日语句子：必须包含假名（平假名/片假名）。否则按"普通聊天"处理更合理。
    # 这样可以避免出现"分析中文句子结构"的尴尬体验。
    if not re.search(r"[\u3040-\u30FF]", text):
        raise HTTPException(status_code=400, detail="句子分析仅支持日语句子（需要包含假名）。中文问题请直接问小雪梨即可。")

    prompt = f"""
你是面向中文母语者的日语写作教练。请只分析下面这句日语（可能不标准）。
如果输入里夹杂了中文翻译/解释/提问背景，把中文当作参考即可：不要分析任何中文句子结构，也不要输出"中文主谓宾/中文语法结构"之类内容。

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
        # 句子分析是"特定任务"，把 prompt 当作本轮用户输入即可
        answer = await get_ai().chat(prompt)
    except RuntimeError as e:
        if str(e) == "AI_QUOTA":
            raise HTTPException(status_code=503, detail="AI 服务额度暂时不足，请稍后再试。")
        raise HTTPException(status_code=503, detail="AI 服务暂时不可用，请稍后再试。")
    _increment_ai_usage(current_user["id"])
    return {"answer": answer}

def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except Exception:
        return False


def _library_fetch_by_slug(conn, slug: str) -> Optional[Dict[str, Any]]:
    if not slug:
        return None

    # 进程内缓存：避免重复查询同一个词条
    now_s = time.time()
    entry = _word_detail_cache.get(slug)
    if entry is not None and now_s < entry[0]:
        return entry[1]

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        if _is_uuid(slug):
            cur.execute(
                """
                SELECT id, level, word, reading, meaning, mp3,
                       pos, frequency, examples,
                       social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no,
                       COALESCE(tags, '{}'::text[]) AS tags
                FROM vocab_library
                WHERE id=%s::uuid
                """,
                (slug,),
            )
        else:
            # Tier 1: 精确匹配 word/reading（B-tree 索引，< 10ms）
            cur.execute(
                """
                SELECT id, level, word, reading, meaning, mp3,
                       pos, frequency, examples,
                       social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no,
                       COALESCE(tags, '{}'::text[]) AS tags
                FROM vocab_library
                WHERE word = %(slug)s OR reading = %(slug)s
                ORDER BY level DESC, order_no ASC
                LIMIT 1
                """,
                {"slug": slug},
            )
            row = cur.fetchone()
            # Tier 2: ILIKE 回退（仅当精确匹配无结果且 slug >= 3 字符，避免短文本扫全表）
            if not row and len(slug) >= 3:
                like = f"%{slug}%"
                cur.execute(
                    """
                    SELECT id, level, word, reading, meaning, mp3,
                           pos, frequency, examples,
                           social_context, heatmap_data, insight_text, image_url, is_ai_enriched, order_no,
                           COALESCE(tags, '{}'::text[]) AS tags
                    FROM vocab_library
                    WHERE meaning ILIKE %(like)s
                    ORDER BY level DESC, order_no ASC
                    LIMIT 1
                    """,
                    {"like": like},
                )
                row = cur.fetchone()
            if not row:
                return None
            result = dict(row)
            # 同时用 UUID 和查询 slug 缓存
            _word_detail_cache[slug] = (now_s + _WORD_DETAIL_CACHE_TTL, result)
            if str(result.get("id")) != slug:
                _word_detail_cache[str(result["id"])] = (now_s + _WORD_DETAIL_CACHE_TTL, result)
            _trim_word_detail_cache()
            return result
        row = cur.fetchone()
        if not row:
            return None
        result = dict(row)
        _word_detail_cache[slug] = (now_s + _WORD_DETAIL_CACHE_TTL, result)
        _trim_word_detail_cache()
        return result


def _ensure_vocab_reports_table(conn) -> None:
    """用于站内"单词卡报错"功能（若无表则自动创建）。"""
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


def _require_admin_key(x_admin_key: Optional[str]) -> None:
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
        _return_db_conn(conn)


@app.get("/api/admin/announcements")
async def admin_list_announcements(
    limit: int = Query(50, ge=1, le=200),
    x_admin_key: Optional[str] = Header(default=None, alias="x-admin-key"),
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
        _return_db_conn(conn)


@app.post("/api/admin/announcements")
async def create_announcement(
    payload: Dict[str, Any] = Body(...),
    x_admin_key: Optional[str] = Header(default=None, alias="x-admin-key"),
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
        _return_db_conn(conn)


@app.post("/api/admin/announcements/{ann_id}/toggle")
async def toggle_announcement(
    ann_id: str,
    payload: Dict[str, Any] = Body(...),
    x_admin_key: Optional[str] = Header(default=None, alias="x-admin-key"),
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
        _return_db_conn(conn)

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
        _return_db_conn(conn)


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
        _return_db_conn(conn)


@app.get("/word/{slug}", response_class=HTMLResponse)
async def word_detail_page(slug: str):
    try:
        return HTMLResponse(content=_read_local_file("word.html"))
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
        mp3_val = row.get("mp3") or ""
        payload = {
            "id": str(row.get("id")),
            "level": row.get("level") or "",
            "word": row.get("word") or "",
            "reading": row.get("reading") or "",
            "meaning": row.get("meaning") or "",
            "mp3": mp3_val,
            "audio_url": _build_audio_url(mp3_val),
            "pos": row.get("pos") or "",
            "frequency": row.get("frequency") or None,
            "examples": row.get("examples") or None,
            "social_context": row.get("social_context") or None,
            "heatmap_data": row.get("heatmap_data") or None,
            "insight_text": row.get("insight_text") or "",
            "image_url": row.get("image_url") or "",
            "is_ai_enriched": bool(row.get("is_ai_enriched") or False),
            "tags": list(row.get("tags") or []),
        }
        return JSONResponse(content=payload, headers={
            "Cache-Control": "public, max-age=120",
            "CDN-Cache-Control": "public, max-age=300",
        })
    finally:
        _return_db_conn(conn)


def _search_db_direct(qq: str, limit: int) -> list:
    """内存缓存未就绪时的 DB 直查 fallback — 单次查询 < 1s，不阻塞用户。"""
    try:
        conn = _get_db_conn()
        if conn is None:
            return []
    except Exception:
        return []
    try:
        q_j = _map_s2j(qq)
        has_jp = q_j != qq
        patterns = [qq]
        if has_jp:
            patterns.append(q_j)

        where_parts = []
        where_params = []
        rank_parts = []
        rank_params = []

        for p in patterns:
            where_parts.append("word = %s")
            where_params.append(p)
            rank_parts.append("word = %s")
            rank_params.append(p)

            where_parts.append("reading = %s")
            where_params.append(p)
            rank_parts.append("reading = %s")
            rank_params.append(p)

        for p in patterns:
            where_parts.append("word ILIKE %s")
            where_params.append(p + "%")
            where_parts.append("reading ILIKE %s")
            where_params.append(p + "%")

        for p in patterns:
            where_parts.append("word ILIKE %s")
            where_params.append("%" + p + "%")
            where_parts.append("reading ILIKE %s")
            where_params.append("%" + p + "%")

        # 释义搜索：中文查询或较长的非日语查询（低优先级，仅出现在词/读匹配不足时）
        has_kana = bool(re.search(r"[぀-ゟ゠-ヿー]", qq))
        chinese_chars = len(re.findall(r"[一-鿿]", qq))
        need_meaning = chinese_chars >= 1 or (not has_kana and len(qq) >= 2)
        if need_meaning:
            for p in patterns:
                where_parts.append("meaning ILIKE %s")
                where_params.append("%" + p + "%")

        limit_val = int(limit)
        query_sql = f"""
            SELECT id::text, level, word, reading, pos, frequency,
                   CASE WHEN jsonb_typeof(examples) = 'array'
                        THEN jsonb_array_length(examples) ELSE 0 END AS ec,
                   COALESCE(length(insight_text), 0) AS il,
                   is_ai_enriched, order_no,
                   meaning, mp3, image_url,
                   COALESCE(tags, '{{}}'::text[]) AS tags
            FROM vocab_library
            WHERE {" OR ".join(where_parts)}
            ORDER BY
                CASE
                    WHEN {" OR ".join(rank_parts)} THEN 0
                    ELSE 1
                END,
                frequency DESC NULLS LAST
            LIMIT {limit_val}
        """
        params = where_params + rank_params
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        out = []
        seen_words = set()
        for r in rows:
            w = (r.get("word") or "").strip()
            if w in seen_words:
                continue
            seen_words.add(w)
            mp3_val = r.get("mp3") or ""
            out.append({
                "id": str(r["id"]),
                "level": r.get("level") or "",
                "word": w,
                "reading": r.get("reading") or "",
                "pos": r.get("pos") or "",
                "frequency": r.get("frequency") or 0,
                "examples_count": int(r.get("ec") or 0),
                "insight_len": int(r.get("il") or 0),
                "is_ai_enriched": bool(r.get("is_ai_enriched") or False),
                "order_no": r.get("order_no") or 0,
                "meaning": r.get("meaning") or "",
                "mp3": mp3_val,
                "audio_url": _build_audio_url(mp3_val),
                "image_url": r.get("image_url") or "",
                "tags": list(r.get("tags") or []),
            })
        return out
    finally:
        _return_db_conn(conn)


def _suggest_db_direct(qq: str, limit: int) -> list:
    """联想接口的 DB 直查 fallback — 仅前缀匹配，< 1s 返回。"""
    try:
        conn = _get_db_conn()
        if conn is None:
            return []
    except Exception:
        return []
    try:
        q_lower = qq.lower()
        q_j = _map_s2j(qq)
        patterns = [qq]
        if q_j != qq:
            patterns.append(q_j)

        params = []
        exact_parts = []
        for p in patterns:
            exact_parts.append("word = %s")
            params.append(p)
            exact_parts.append("reading = %s")
            params.append(p)

        prefix_parts = []
        for p in patterns:
            prefix_parts.append("word ILIKE %s")
            params.append(p + "%")
            prefix_parts.append("reading ILIKE %s")
            params.append(p + "%")

        rank_parts = exact_parts.copy()
        rank_params = list(params[:len(exact_parts)])
        params.extend(rank_params)

        limit_val = int(limit)
        query_sql = f"""
            SELECT id::text, level, word, reading, pos, frequency,
                   COALESCE(length(insight_text), 0) AS il,
                   is_ai_enriched, meaning, mp3, image_url,
                   COALESCE(tags, '{{}}'::text[]) AS tags
            FROM vocab_library
            WHERE {" OR ".join(exact_parts + prefix_parts)}
            ORDER BY
                CASE WHEN {" OR ".join(rank_parts)} THEN 0 ELSE 1 END,
                frequency DESC NULLS LAST
            LIMIT {limit_val}
        """
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        out = []
        seen = set()
        for r in rows:
            w = (r.get("word") or "").strip()
            if w in seen:
                continue
            seen.add(w)
            mp3_val = r.get("mp3") or ""
            out.append({
                "id": str(r["id"]),
                "level": r.get("level") or "",
                "word": w,
                "reading": r.get("reading") or "",
                "pos": r.get("pos") or "",
                "frequency": r.get("frequency") or 0,
                "meaning": r.get("meaning") or "",
                "mp3": mp3_val,
                "audio_url": _build_audio_url(mp3_val),
                "image_url": r.get("image_url") or "",
                "tags": list(r.get("tags") or []),
            })
        return out
    finally:
        _return_db_conn(conn)


@app.get("/api/library/search")
async def search_library(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50), _debug: int = Query(0, ge=0, le=1)):
    """词库搜索（SQLite 快照直查，零内存缓存，内存友好）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    qq = (q or "").strip()
    if not qq:
        return []

    cache_key = (qq.lower(), int(limit))
    now_s = time.time()
    entry = _search_cache.get(cache_key)
    if entry is not None:
        expires, cached = entry
        if now_s < expires:
            return cached
        del _search_cache[cache_key]

    t0 = time.time()
    _ensure_snapshot_available()
    t_load = time.time() - t0

    qq_lower = qq.lower()
    has_kana = bool(re.search(r"[぀-ゟ゠-ヿー]", qq))
    chinese_chars = len(re.findall(r"[一-鿿]", qq))
    qq_j = _map_s2j(qq)
    has_jp_variant = qq_j != qq

    # 快照未就绪 → DB fallback
    if not os.path.exists(_VOCAB_SNAPSHOT_DB):
        return _search_db_direct(qq, int(limit))

    db = _snapshot_conn()
    seen: set = set()
    results: List[Dict[str, Any]] = []

    def _add_dict(d: dict, rank: int, kind: str):
        wkey = str(d.get("word") or "").strip()
        if not wkey or wkey in seen:
            return
        seen.add(wkey)
        d["match_rank"] = rank
        d["match_kind"] = kind
        results.append(d)

    patterns = [qq_lower]
    if has_jp_variant:
        patterns.append(qq_j.lower())

    # Tier 1: 精确匹配 word/reading
    for p in patterns:
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(word) = ? LIMIT ?", (p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 0, "word_exact")
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(reading) = ? LIMIT ?", (p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 10, "reading_exact")

    # Tier 2: 前缀匹配
    for p in patterns:
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(word) LIKE ? LIMIT ?", (p + '%', int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 2, "word_prefix")
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(reading) LIKE ? LIMIT ?", (p + '%', int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 12, "reading_prefix")

    # Tier 3: 包含匹配 + 释义
    need_meaning = chinese_chars >= 1 or (not has_kana and len(qq) >= 2)
    scan_limit = int(limit) * 5
    if len(results) < int(limit) * 2:
        for p in patterns:
            for row in db.execute(
                "SELECT * FROM vocab_snapshot WHERE LOWER(word) LIKE ? AND LOWER(word) NOT LIKE ? LIMIT ?",
                ('%' + p + '%', p + '%', scan_limit)
            ):
                _add_dict(_snapshot_row_to_dict(row), 5, "word_partial")
            for row in db.execute(
                "SELECT * FROM vocab_snapshot WHERE LOWER(reading) LIKE ? AND LOWER(reading) NOT LIKE ? LIMIT ?",
                ('%' + p + '%', p + '%', scan_limit)
            ):
                _add_dict(_snapshot_row_to_dict(row), 15, "reading_partial")
        if need_meaning and len(results) < scan_limit:
            for p in patterns:
                for row in db.execute(
                    "SELECT * FROM vocab_snapshot WHERE LOWER(meaning) LIKE ? LIMIT ?",
                    ('%' + p + '%', scan_limit)
                ):
                    _add_dict(_snapshot_row_to_dict(row), 30, "meaning_partial")

    t_match = time.time() - t0

    def _lv_rank(lv: str) -> int:
        m = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
        return m.get((lv or "").upper(), 0)

    def _sort_key(d):
        return (
            d["match_rank"],
            -_int0(d.get("frequency")),
            -int(d.get("insight_len") or 0),
            -_lv_rank(str(d.get("level") or "")),
        )

    results.sort(key=_sort_key)
    out = results[:int(limit)]

    for d in out:
        d.pop("insight_len", None)

    if _debug:
        return {
            "items": out,
            "_debug": {
                "t_load": round(t_load, 3),
                "t_match": round(t_match, 3),
                "t_total": round(time.time() - t0, 3),
                "total_matched": len(results),
                "out_count": len(out),
            },
        }

    _search_cache[cache_key] = (now_s + _SEARCH_CACHE_TTL, out)
    if len(_search_cache) > _SEARCH_CACHE_MAX:
        stale = [k for k, (exp, _) in _search_cache.items() if now_s >= exp]
        for k in stale:
            _search_cache.pop(k, None)
    return out

@app.get("/api/library/suggest")
async def suggest_library(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=30)):
    """实时联想（SQLite 快照直查，前缀匹配为主）。"""
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    qq = (q or "").strip()
    if not qq:
        return []

    _ensure_snapshot_available()
    if not os.path.exists(_VOCAB_SNAPSHOT_DB):
        return _suggest_db_direct(qq, int(limit))

    qq_lower = qq.lower()
    qq_j = _map_s2j(qq)
    has_jp_variant = qq_j != qq
    is_kana_only = _is_kana_only(qq)

    db = _snapshot_conn()
    seen: set = set()
    matched: List[Dict[str, Any]] = []

    def _add_dict(d: dict, rank: int):
        wkey = str(d.get("word") or "").strip()
        if not wkey or wkey in seen:
            return
        seen.add(wkey)
        d["_rank"] = rank
        matched.append(d)

    patterns = [qq_lower]
    if has_jp_variant:
        patterns.append(qq_j.lower())

    # Exact match
    for p in patterns:
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(word) = ? LIMIT ?", (p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 0)
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(reading) = ? LIMIT ?", (p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 3)

    # Prefix match
    for p in patterns:
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(word) LIKE ? AND LOWER(word) != ? LIMIT ?", (p + '%', p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 1)
        for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(reading) LIKE ? AND LOWER(reading) != ? LIMIT ?", (p + '%', p, int(limit))):
            _add_dict(_snapshot_row_to_dict(row), 2)

    # Meaning suggest for Chinese queries
    chinese_chars = len(re.findall(r"[一-鿿]", qq))
    if chinese_chars >= 1 and len(matched) < 3:
        for p in patterns:
            for row in db.execute("SELECT * FROM vocab_snapshot WHERE LOWER(meaning) LIKE ? LIMIT 8", ('%' + p + '%',)):
                _add_dict(_snapshot_row_to_dict(row), 10)

    def _lv_rank(lv: str) -> int:
        m = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
        return m.get((lv or "").upper(), 0)

    if is_kana_only:
        matched.sort(key=lambda x: (
            int(x.get("_rank", 9)),
            -_int0(x.get("frequency")),
            -int(x.get("insight_len") or 0),
            -_lv_rank(str(x.get("level") or "")),
            str(x.get("word") or ""),
        ))
    else:
        matched.sort(key=lambda x: (
            int(x.get("_rank", 9)),
            -int(x.get("insight_len") or 0),
            -_int0(x.get("frequency")),
            -_lv_rank(str(x.get("level") or "")),
            str(x.get("word") or ""),
        ))

    out = matched[:int(limit)]
    for d in out:
        d.pop("insight_len", None)
        d.pop("is_ai_enriched", None)
        d.pop("_rank", None)
        d.pop("examples_count", None)
    return out


@app.get("/api/library/tags")
async def list_tags():
    """列出所有标签及对应词数（快照优先，失败回退 DB 直查）。"""
    _ensure_snapshot_available()
    if os.path.exists(_VOCAB_SNAPSHOT_DB):
        db = _snapshot_open()
        try:
            tag_counts: Dict[str, int] = {}
            for row in db.execute("SELECT tags FROM vocab_snapshot"):
                tags_raw = row["tags"] or ""
                for t in tags_raw.split(","):
                    t = t.strip()
                    if t:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
            tags = [{"name": tag, "count": cnt}
                    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1])]
            return {"tags": tags}
        finally:
            db.close()
    # DB fallback
    try:
        conn = _get_db_conn()
        if conn is None:
            return {"tags": []}
    except Exception:
        return {"tags": []}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT unnest(tags) as t, count(*) FROM vocab_library WHERE tags IS NOT NULL GROUP BY t ORDER BY count(*) DESC")
            tags = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]
        return {"tags": tags}
    except Exception:
        return {"tags": []}
    finally:
        _return_db_conn(conn)


@app.get("/api/library/words/by-tag")
async def words_by_tag(tag: str = Query(...), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    """按标签分页取词（快照优先，失败回退 DB 直查）。"""
    _ensure_snapshot_available()
    tag_clean = tag.strip()
    if os.path.exists(_VOCAB_SNAPSHOT_DB):
        db = _snapshot_open()
        try:
            count_row = db.execute(
                "SELECT COUNT(*) as cnt FROM vocab_snapshot WHERE ',' || tags || ',' LIKE ?",
                ('%,' + tag_clean + ',%',)
            ).fetchone()
            total = count_row["cnt"] if count_row else 0
            rows = db.execute(
                "SELECT * FROM vocab_snapshot WHERE ',' || tags || ',' LIKE ? ORDER BY word LIMIT ? OFFSET ?",
                ('%,' + tag_clean + ',%', int(limit), int(offset))
            ).fetchall()
            words = [_snapshot_row_to_dict(r) for r in rows]
            return {"tag": tag_clean, "total": total, "offset": offset, "words": words}
        finally:
            db.close()
    # DB fallback
    try:
        conn = _get_db_conn()
        if conn is None:
            return {"tag": tag_clean, "total": 0, "offset": offset, "words": []}
    except Exception:
        return {"tag": tag_clean, "total": 0, "offset": offset, "words": []}
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT count(*) FROM vocab_library WHERE %s = ANY(tags)", (tag_clean,))
            total = cur.fetchone()["count"]
            cur.execute(
                """SELECT id::text, level, word, reading, pos, frequency,
                   COALESCE(length(insight_text), 0) as insight_len,
                   is_ai_enriched, order_no, COALESCE(meaning, ''),
                   COALESCE(tags, '{}'::text[]) as tags
                FROM vocab_library WHERE %s = ANY(tags)
                ORDER BY word LIMIT %s OFFSET %s""",
                (tag_clean, int(limit), int(offset))
            )
            rows = cur.fetchall()
        words = []
        for r in rows:
            words.append({
                "id": str(r["id"]), "level": r["level"] or "", "word": r["word"] or "",
                "reading": r["reading"] or "", "pos": r["pos"] or "",
                "frequency": r["frequency"] or 0, "examples_count": 0,
                "insight_len": r["insight_len"] or 0, "is_ai_enriched": r["is_ai_enriched"] or False,
                "order_no": r["order_no"] or 0, "meaning": r["meaning"] or "",
                "mp3": "", "audio_url": "", "image_url": "",
                "tags": list(r["tags"] or []),
            })
        return {"tag": tag_clean, "total": total, "offset": offset, "words": words}
    except Exception:
        return {"tag": tag_clean, "total": 0, "offset": offset, "words": []}
    finally:
        _return_db_conn(conn)


@app.get("/api/library/replacements")
async def get_replacements(word: str = Query(...), level: str = Query(""), limit: int = Query(5, ge=1, le=10)):
    """
    替换建议：找出当前词不适合的社交场景，推荐意思相近、但适合在该场景使用的替代表达（零 AI 消耗）。
    例如：当前词不适合对上司说 → 找意思相近且适合对上司说的词。
    """
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    w = (word or "").strip()
    if not w:
        return []

    # 缓存加速（与 synonyms 相同模式，10 分钟 TTL）
    cache_key = ("repl", w, (level or "").strip(), int(limit))
    now_s = time.time()
    entry = _search_cache.get(cache_key)
    if entry is not None:
        expires, cached = entry
        if now_s < expires:
            return cached
        del _search_cache[cache_key]

    _cache_repl = []

    # 社交语境定义
    CTX_MAP = {
        "casual": {"key": "casual", "label_zh": "朋友/同僚", "label": "随意场合"},
        "business": {"key": "business", "label_zh": "上司/客户", "label": "商务场合"},
        "formal": {"key": "formal", "label_zh": "店员/陌生人", "label": "正式场合"},
    }

    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # 1) 获取当前词的 social_context / heatmap_data / meaning
            cur.execute(
                "SELECT social_context, heatmap_data, level, meaning FROM vocab_library WHERE word = %s AND is_ai_enriched = true ORDER BY frequency DESC NULLS LAST LIMIT 1",
                (w,),
            )
            row = cur.fetchone()
            if not row:
                return {"items": [], "denied": []}

            sc = row.get("social_context") or {}
            hm = row.get("heatmap_data") or {}
            lv = (level or row.get("level") or "").strip()
            meaning = (row.get("meaning") or "").strip()

            if isinstance(hm, dict):
                scenes = [k for k in hm.keys() if k and str(k).strip()]
            else:
                scenes = []

            # 找出当前词不适合的社交场景（先算，后面需要返回）
            denied_contexts = []
            if isinstance(sc, dict):
                for ctx_key in ("casual", "business", "formal"):
                    ctx = sc.get(ctx_key)
                    if isinstance(ctx, dict) and not ctx.get("allowed"):
                        info = CTX_MAP.get(ctx_key, {})
                        denied_contexts.append({
                            "key": ctx_key,
                            "label": info.get("label", ctx_key),
                            "label_zh": info.get("label_zh", ""),
                        })

            if not isinstance(sc, dict) or not scenes:
                return {"items": [], "denied": [d["label_zh"] for d in denied_contexts]}

            # 2) 提取意思关键词
            meaning_keywords = _extract_meaning_keywords(meaning)
            if not meaning_keywords:
                return {"items": [], "denied": [d["label_zh"] for d in denied_contexts]}

            if not denied_contexts:
                return {"items": [], "denied": []}

            # 3) 对每个不适合的场景，找意思相近 + 同场景 + 适合的词
            all_results = []
            for dctx in denied_contexts[:2]:
                # 意思筛选：至少匹配一个关键词
                # 单字用前缀匹配（"吃%" 避免匹配到"吃惊""吃不消"），多字用包含匹配
                params = {
                    "word": w,
                    "scenes": scenes,
                    "ctx_key": dctx["key"],
                    "lv": lv,
                    "limit": int(limit),
                }
                clauses = []
                for i, kw in enumerate(meaning_keywords):
                    key = f"mkw{i}"
                    if len(kw) == 1:
                        clauses.append(f"meaning ILIKE %({key})s")
                        params[key] = f"{kw}%"
                    elif len(kw) >= 2:
                        clauses.append(f"meaning ILIKE %({key})s")
                        params[key] = f"%{kw}%"
                meaning_clauses = " OR ".join(clauses)

                # 优先：意思相近 + 同场景 + 社交允许 + 同级
                cur.execute(
                    f"""SELECT word, reading, meaning, level, frequency, heatmap_data, social_context
                     FROM vocab_library
                     WHERE word != %(word)s
                       AND is_ai_enriched = true
                       AND heatmap_data IS NOT NULL
                       AND social_context IS NOT NULL
                       AND heatmap_data ?| %(scenes)s
                       AND social_context -> %(ctx_key)s ->> 'allowed' = 'true'
                       AND ({meaning_clauses})
                       AND (%(lv)s = '' OR level = %(lv)s)
                     ORDER BY frequency DESC NULLS LAST, word ASC
                     LIMIT %(limit)s""",
                    params,
                )
                for r in (cur.fetchall() or []):
                    hm2 = r.get("heatmap_data") or {}
                    if isinstance(hm2, dict):
                        shared = [s for s in scenes if s in hm2]
                    else:
                        shared = []
                    all_results.append({
                        "word": r.get("word"),
                        "reading": r.get("reading"),
                        "meaning": r.get("meaning"),
                        "level": r.get("level"),
                        "frequency": r.get("frequency"),
                        "shared_scenes": shared[:3],
                        "replace_for": dctx["label_zh"] or dctx["label"],
                    })

            # 去重并按频次排序
            seen = set()
            unique = []
            for item in all_results:
                if item["word"] not in seen:
                    seen.add(item["word"])
                    unique.append(item)
            unique.sort(key=lambda x: (-(x.get("frequency") or 0), -len(x.get("shared_scenes") or [])))
            _cache_repl = {
                "items": unique[:int(limit)],
                "denied": [d["label_zh"] for d in denied_contexts],
            }
    finally:
        _return_db_conn(conn)
    if _cache_repl:
        _search_cache[cache_key] = (now_s + 600, _cache_repl)
        if len(_search_cache) > _SEARCH_CACHE_MAX:
            stale = [k for k, (exp, _) in _search_cache.items() if now_s >= exp]
            for k in stale:
                _search_cache.pop(k, None)
    return _cache_repl


@app.get("/api/library/synonyms")
async def get_synonyms(word: str = Query(...), level: str = Query(""), limit: int = Query(8, ge=1, le=15)):
    """
    同义词推荐：找意思相近的词（不看社交场合，纯语义相似）。
    用于拓宽词汇量，了解同一含义的不同表达方式。
    """
    if not SUPABASE_DB_ENABLED:
        raise HTTPException(status_code=500, detail="SUPABASE_DB_URL is not configured")
    w = (word or "").strip()
    if not w:
        return []

    # 缓存加速：同义词结果 TTL 10 分钟
    cache_key = ("syn", w, (level or "").strip(), int(limit))
    now_s = time.time()
    entry = _search_cache.get(cache_key)
    if entry is not None:
        expires, cached = entry
        if now_s < expires:
            return cached
        del _search_cache[cache_key]

    _cache_synonyms = []
    conn = _pg_conn()
    try:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT meaning, level FROM vocab_library WHERE word = %s AND is_ai_enriched = true ORDER BY frequency DESC NULLS LAST LIMIT 1",
                (w,),
            )
            row = cur.fetchone()
            if not row:
                return []

            meaning = (row.get("meaning") or "").strip()
            lv = (level or row.get("level") or "").strip()

            keywords = _extract_meaning_keywords(meaning)
            if not keywords:
                return []

            has_single_char = any(len(k) == 1 for k in keywords)

            def _do_synonym_query(strict: bool):
                """strict=True: 单字用正则前缀（排除假匹配）；strict=False: 全用包含匹配（宽召回）"""
                params = {"word": w, "lv": lv, "limit": int(limit)}
                clauses = []
                score_parts = []
                for i, kw in enumerate(keywords):
                    key = f"mkw{i}"
                    if len(kw) == 1 and strict:
                        pattern = f"^{kw}([，,、。.；;（(（]|$)"
                        clauses.append(f"meaning ~ %({key})s")
                        params[key] = pattern
                    else:
                        clauses.append(f"meaning ILIKE %({key})s")
                        params[key] = f"%{kw}%"
                    score_parts.append(f"CASE WHEN meaning ILIKE %({key}_score)s THEN 1 ELSE 0 END")
                    params[f"{key}_score"] = f"%{kw}%"
                meaning_clauses = " OR ".join(clauses)
                score_expr = " + ".join(score_parts)

                cur.execute(
                    f"""SELECT word, reading, meaning, level, frequency,
                            ({score_expr}) AS kw_score
                     FROM vocab_library
                     WHERE word != %(word)s
                       AND is_ai_enriched = true
                       AND ({meaning_clauses})
                       AND level IN ('N1','N2','N3','N4','N5')
                       AND char_length(word) <= 6
                     ORDER BY kw_score DESC, CASE WHEN level = %(lv)s THEN 0 ELSE 1 END, frequency DESC NULLS LAST
                     LIMIT %(limit)s""",
                    params,
                )
                results = []
                seen_words = set()
                for r in (cur.fetchall() or []):
                    wrd = r.get("word")
                    if wrd in seen_words:
                        continue
                    seen_words.add(wrd)
                    results.append({
                        "word": wrd,
                        "reading": r.get("reading"),
                        "meaning": r.get("meaning"),
                        "level": r.get("level"),
                        "frequency": r.get("frequency"),
                    })
                return results

            # 第一遍：严格匹配（单字正则）
            results = _do_synonym_query(strict=True)
            # 如果严格模式结果太少，用宽松模式兜底
            if len(results) < 2 and has_single_char:
                results = _do_synonym_query(strict=False)

            # 第三遍：如果仍无结果，将长关键词拆为 2 字片段重试
            if len(results) < 2:
                decomposed = []
                seen_d = set()
                for kw in keywords:
                    if len(kw) <= 2:
                        if kw not in seen_d:
                            decomposed.append(kw)
                            seen_d.add(kw)
                    else:
                        for i in range(0, len(kw) - 1, 2):
                            seg = kw[i:i+2]
                            if seg not in seen_d and re.search(r"[一-鿿]", seg):
                                decomposed.append(seg)
                                seen_d.add(seg)
                if decomposed and decomposed != keywords:
                    keywords_backup = keywords
                    keywords = decomposed[:6]
                    has_single_char = any(len(k) == 1 for k in keywords)
                    results = _do_synonym_query(strict=False)

            # 缓存结果（finally 之后统一出口）
            _cache_synonyms = results
    finally:
        _pg_close(conn)
    if _cache_synonyms:
        _search_cache[cache_key] = (now_s + 600, _cache_synonyms)
        if len(_search_cache) > _SEARCH_CACHE_MAX:
            stale = [k for k, (exp, _) in _search_cache.items() if now_s >= exp]
            for k in stale:
                _search_cache.pop(k, None)
    return _cache_synonyms


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
            mp3_val = row.get("mp3") or ""
            return {
                "status": "ok",
                "data": {
                    "id": str(row.get("id")),
                    "level": row.get("level") or "",
                    "word": row.get("word") or "",
                    "reading": row.get("reading") or "",
                    "meaning": row.get("meaning") or "",
                    "mp3": mp3_val,
                    "audio_url": _build_audio_url(mp3_val),
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
        enriched = await get_ai().enrich_library_entry(
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
        _return_db_conn(conn)



@app.get("/api/healthz")
async def healthz():
    """轻量健康检查：不依赖数据库/外部网络。"""
    return {"ok": True}


@app.get("/api/debug/storage-bucket")
async def debug_storage_bucket():
    """调试端点：测试 Storage bucket 创建/上传状态。"""
    bucket_exists = False
    snapshot_uploaded = False
    error_msg = ""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return {"ok": False, "error": "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY"}
    try:
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{SUPABASE_URL}/storage/v1/bucket/{_VOCAB_SNAPSHOT_BUCKET}",
                headers=headers)
            if resp.status_code == 200:
                bucket_exists = True
            elif resp.status_code == 404:
                # 尝试创建
                resp = client.post(
                    f"{SUPABASE_URL}/storage/v1/bucket",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"name": _VOCAB_SNAPSHOT_BUCKET, "id": _VOCAB_SNAPSHOT_BUCKET,
                          "public": True, "file_size_limit": 52428800})
                if resp.status_code in (200, 201):
                    bucket_exists = True
                else:
                    error_msg = f"Create failed: HTTP {resp.status_code} {resp.text[:200]}"
            else:
                error_msg = f"Get failed: HTTP {resp.status_code} {resp.text[:200]}"

        # 检查快照是否已上传
        if bucket_exists:
            resp2 = httpx.head(
                f"{SUPABASE_URL}/storage/v1/object/public/{_VOCAB_SNAPSHOT_BUCKET}/{_VOCAB_SNAPSHOT_KEY}",
                timeout=5)
            snapshot_uploaded = resp2.status_code == 200
    except Exception as e:
        error_msg = str(e)
    return {
        "ok": bucket_exists,
        "bucket": _VOCAB_SNAPSHOT_BUCKET,
        "bucket_exists": bucket_exists,
        "snapshot_uploaded": snapshot_uploaded,
        "error": error_msg
    }


def _read_local_file(filename: str) -> str:
    cached = _html_file_cache.get(filename)
    if cached is not None:
        return cached
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _html_file_cache[filename] = content
    return content


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def root():
    # Render 健康检查常用 HEAD /，若返回 405 会导致被判定不健康进而 502。
    try:
        return HTMLResponse(content=_read_local_file("web.html"))
    except FileNotFoundError:
        # 若部署时未包含 web.html，避免请求挂住/连接被重置，直接给出可诊断的 500。
        return HTMLResponse(
            content="web.html not found on server (deploy/build issue).",
            status_code=500,
        )

@app.api_route("/web.html", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def web_html_alias():
    return await root()

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
    uvicorn.run(app, host="0.0.0.0", port=port,
                limit_concurrency=64, limit_max_requests=10000, timeout_keep_alive=30)
