# Supabase 表结构说明（中文）

这份文档用于把你当前 Supabase（Postgres）里的表按“用途”整理清楚，方便你后续维护、扩展与清理。

> 说明：你的项目同时存在 **本地 SQLite（cache.db）** 与 **Supabase Postgres** 两套数据源。下面重点是 Supabase（Postgres）表；文末附带本地 SQLite 表说明。

---

## A. 核心词库（你的网站搜索/详情页主要依赖）

### 1) `public.vocab_library`（主词库，推荐长期保留）
- **用途**：统一存放 N1~N5 词条，供搜索、详情页展示、背词系统使用。
- **前端是否直接访问**：不直接；前端通过后端 API 访问（更安全可控）。
- **关键字段**：
  - `level`（N1~N5）
  - `word`（日语词条）
  - `reading`（假名）
  - `meaning`（释义，含中文；用于中文检索）
  - `pos`（词性）
  - `frequency`（1~5 频次）
  - `examples`（jsonb，例句数组）
  - `mp3`（发音文件名）
  - `social_context`（jsonb：适用对象）
  - `heatmap_data`（jsonb：高频场景）
  - `insight_text`（深度解析）
  - `image_url`（图片 URL；未来可由 Storage 批量回填）
  - `is_ai_enriched`（是否已 AI 补全）
  - `order_no`（排序）
- **唯一约束**：`UNIQUE(level, word)`（同一等级同一词只保留一条）
- **建议**：这是“唯一主词库”，其它词库表尽量避免再新增（降低复杂度）。

### 2) `public.vocab_library_reports`（单词卡报错/纠错建议，推荐保留）
- **用途**：前端“报错/反馈”提交，站长在后台查看，方便你修正词条。
- **前端是否直接访问**：提交接口可公开，但写入由后端控制；读取建议仅后台使用。
- **建议**：保留。后续可加状态（open/resolved）与处理人字段。

---

## B. 背词学习系统（基于 vocab_library 的学习进度）

### 3) `public.library_progress`（词库学习进度，推荐保留）
- **用途**：记录用户对 `vocab_library` 的复习/记忆曲线（SRS）。
- **关联**：`entry_id -> vocab_library.id`
- **建议**：保留；它是学习功能的核心数据。

### 4) `public.library_plans`（学习计划，推荐保留）
- **用途**：记录用户每天的学习计划（每天新词量等）。
- **建议**：保留。

---

## C. 用户与权限（登录、邀请码、用户资料）

### 5) `public.profiles`（用户资料，推荐保留）
- **用途**：昵称、当前等级、学习目标等。
- **建议**：保留。

### 6) `public.invitation_codes`（邀请码，按你需求保留）
- **用途**：邀请码登录/注册流程的控制。
- **建议**：保留；如果未来不再用邀请码机制，可以整体下线并删除。

---

## D. 社区与反馈

### 7) `public.forum_posts`（论坛帖子，按需保留）
- **用途**：论坛/讨论区的帖子与回复。
- **建议**：如果你确定论坛会长期运营则保留，否则可后续下线简化。

### 8) `public.feedbacks`（用户反馈，推荐保留）
- **用途**：用户反馈/意见收集。

---

## E. 旧系统遗留表（建议“只读或逐步下线”，避免混用）

### 9) `public.words`（旧词库表，遗留）
- **用途**：旧版分析/学习链路使用（与 `vocab_library` 字段不一致）。
- **风险**：容易导致“前端查错表”或“数据重复维护”。
- **建议**：短期不要删（防止旧接口依赖），但应逐步迁移到 `vocab_library` 后再删除。

### 10) `public.user_progress` / `public.user_plans`（旧学习系统，遗留）
- **用途**：旧版与 `words` 绑定的学习进度/计划。
- **建议**：如果新学习链路已全面转到 `library_*`，可标记为遗留并停止写入。

---

## F. 建议新增（你刚提出的需求）

### 11) `public.announcements`（公告/通知）
- **用途**：站长发布公告，通知/联络用户。
- **建议字段**：`title, content, is_active, pinned, starts_at, ends_at, created_at, updated_at`
- **访问策略**：
  - 读取：公开（给前端展示）
  - 写入：仅管理员（通过后端校验管理员密钥/管理员账号）

---

## 本地 SQLite（cache.db）说明（不在 Supabase 里）
你的后端还在本地维护了一套 SQLite 作为缓存/演示/兼容层，主要表包括：
- `ai_cache`（AI 结果缓存）
- `vocab_meta_cache`（词条 meta 缓存）
- `users`（本地用户）
- `feedbacks_local` / `forum_posts_local`（本地反馈/论坛）
- `vocab_bank` / `daily_goals` / `study_sessions`（本地背词）

> 建议：长期来看，最好把“最终真相数据”统一收敛到 Supabase Postgres（尤其是用户/学习进度/词库），SQLite 只做短期缓存（可清空）。

