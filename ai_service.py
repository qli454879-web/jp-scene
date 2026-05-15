import os
import json
import asyncio
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List

# Configure API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
# 默认用更聪明也更快的模型；如需更省钱/更快，可在环境变量覆盖 OPENAI_MODEL
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"

class AIService:
    def __init__(self):
        self.provider = None
        
        # Priority 1: OpenAI
        if OPENAI_API_KEY:
            # 支持 OpenAI 兼容中转（通过环境变量配置，避免硬编码凭证/URL）
            if OPENAI_BASE_URL:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            else:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            print("AI Service: Using OpenAI (ChatGPT).")
        
        # Priority 2: Gemini
        elif GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Use the cutting-edge preview model available for this key
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            self.provider = "gemini"
            print(f"AI Service: Using Google Gemini ({GEMINI_MODEL}).")
        
        else:
            self.provider = "mock"
            print("Warning: No AI API Key set. Running in Mock mode.")

    async def analyze_word(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        if self.provider == "mock":
            return self._get_mock_analysis(word, dict_info)

        prompt = f"""
        Role: Japanese Language Expert.
        Word: "{word}"
        Dict Info: {json.dumps(dict_info) if dict_info else "N/A"}
        
        Output JSON only:
        {{
            "word": "{word}",
            "kana": "假名",
            "romaji": "romaji",
            "pos": "词性",
            "pitch": "语调0-9",
            "origin": "Loanword original or null",
            "meaning": "准确中文",
            "frequency": 1-5,
            "safety": 0-100,
            "tone": "语气描述",
            "contexts": [{{"scene": "场景", "usage": %}}],
            "social_targets": [{{"label": "上司/朋友/店员/初对面", "allowed": bool}}],
            "image": "https://images.unsplash.com/photo-1580828343064-fde4fc206bc6?w=400&h=250&fit=crop",
            "examples": [{{"jp": "地道例句", "cn": "中文"}}],
            "explanation": "深度场景化解释及文化背景。"
        }}
        """
        
        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": "You are a Japanese expert outputting JSON."}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=20
                )
                text = response.choices[0].message.content
            else:
                # Use a more stable model or list of fallback models
                response = self.gemini_model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(text)
        except Exception as e:
            print(f"!!! AI Error ({self.provider}): {type(e).__name__} - {str(e)}")
            # If quota exceeded, return mock but with a better message
            mock_data = self._get_mock_analysis(word, dict_info)
            if "quota" in str(e).lower() or "429" in str(e):
                mock_data["explanation"] = "【系统提示】AI 额度已达今日上限，当前为您展示基础词典解析。请稍后再试或更换 API Key。"
            return mock_data

    def _persona_system_prompt(self) -> str:
        # 更自然：允许追问、允许根据上下文续聊；保留人设但不限制死 3-8 句
        return (
            "你是“小雪梨”，也是“哈基米日语博士”。\n"
            "要求：\n"
            "- 不要自称“日语学习助手”。\n"
            "- 每次回答开头用“【小雪梨】”或“【哈基米日语博士】”二选一（尽量轮换）。\n"
            "- 语气像聪明、靠谱、会聊天的真人老师，不要机械模板。\n"
            "- 能读上下文：如果用户在追问/承接上一句，请直接接着答；必要时可以反问 1 个澄清问题。\n"
            "- 日语问题：优先给结论 + 例句 + 常见坑；动词变形/语法纠错要明确指出哪里不自然并给更自然版本。\n"
            "- 闲聊：也要像正常 AI 一样回应，不要一直把话题拉回背词。\n"
        )

    def _messages_to_transcript(self, messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"小雪梨：{content}")
        return "\n".join(lines).strip()

    async def chat(self, question: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        if self.provider == "mock":
            return f"AI 助手模拟回复: 关于 '{question}'，这是一个非常地道的问题。在真实接入 API (Gemini 或 ChatGPT) 后，我会为您分析更复杂的文化背景。"

        sys_prompt = self._persona_system_prompt()
        hist = messages if isinstance(messages, list) else []
        # 限制长度，避免吞吐过大导致变慢
        hist = hist[-12:]

        try:
            if self.provider == "openai":
                msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
                for m in hist:
                    r = (m.get("role") or "").strip()
                    c = (m.get("content") or "").strip()
                    if not c:
                        continue
                    if r in ("user", "assistant"):
                        msgs.append({"role": r, "content": c[:1200]})
                if question and (not msgs or msgs[-1]["role"] != "user"):
                    msgs.append({"role": "user", "content": question})
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=msgs,
                    max_tokens=800,
                    temperature=0.7,
                    timeout=18,
                )
                return response.choices[0].message.content
            else:
                transcript = self._messages_to_transcript(hist)
                prompt = (
                    f"{sys_prompt}\n"
                    "下面是对话历史（可能为空）：\n"
                    f"{transcript}\n\n"
                    f"用户：{question}\n"
                    "小雪梨："
                )
                # google.generativeai 的参数在不同版本略有差异，尽量兼容
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config={"temperature": 0.7, "max_output_tokens": 800},
                    )
                except Exception:
                    response = self.gemini_model.generate_content(prompt)
                return response.text
        except Exception as e:
            print(f"!!! Chat Error ({self.provider}): {str(e)}")
            msg = str(e)
            low = msg.lower()
            # 上游额度/限流：不要把原始报错回显给用户
            if "quota" in low or "429" in low or "rate" in low and "limit" in low:
                raise RuntimeError("AI_QUOTA") from e
            raise RuntimeError("AI_ERROR") from e

    async def get_quick_tip(self, word: str) -> str:
        if self.provider == "mock":
            return f"把“{word}”先和一个最顺手的场景绑在一起，再顺手把读音一起默念一遍，会更容易记住。"

        prompt = (
            f"你是网站里的 AI 助手“小雪梨”。"
            f"请针对日语单词“{word}”，用中文生成一句非常简短、便于记忆的小技巧（50字以内）。"
            "不要写深度解析、词源分析、场景长文、术语解释。"
            "只给用户一个好记、好背、能立刻拿来记忆这个词的小提示。"
            "语气自然，像在陪用户背单词。"
        )
        
        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "429" in msg:
                raise RuntimeError("AI_QUOTA") from e
            return "这个词挺有意思的，建议结合场景深度分析来看看它的具体用法！"

    async def get_vocab_meta(self, word: str, kana: str = "", meaning: str = "") -> Dict[str, Any]:
        if self.provider == "mock":
            origin = None
            if all('\u30a0' <= c <= '\u30ff' or c == 'ー' for c in word):
                origin = meaning.split(",")[0].strip() if meaning else None
            return {"meaning_zh": meaning, "origin": origin}

        prompt = f"""
你是面向中国用户的日语词汇老师。请为下面词条输出严格 JSON（不要解释，不要 markdown）。

词条: {word}
读音: {kana}
释义(可能是英文或不完整): {meaning}

要求：
- meaning_zh 必须是简短准确的中文释义（尽量 2~12 字）。
- 如果该词是外来语/片假名借词，origin 给出来源原词（通常为英文小写，如 good）。不是外来语则 origin 为 null。

输出 JSON:
{{"meaning_zh": "...", "origin": null}}
"""

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=20
                )
                text = response.choices[0].message.content
            else:
                response = self.gemini_model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()

            data = json.loads(text)
            meaning_zh = (data.get("meaning_zh") or "").strip()
            origin = data.get("origin")
            if origin is not None:
                origin = str(origin).strip()
                if origin.lower() in ["null", "none", ""]:
                    origin = None
            return {"meaning_zh": meaning_zh, "origin": origin}
        except Exception:
            origin = None
            if all('\u30a0' <= c <= '\u30ff' or c == 'ー' for c in word):
                origin = meaning.split(",")[0].strip() if meaning else None
            return {"meaning_zh": meaning, "origin": origin}

    async def enrich_word_profile(
        self,
        word: str,
        kana: str = "",
        meaning_zh: str = "",
        level: str = "N1",
    ) -> Optional[Dict[str, Any]]:
        """
        为词库批量补全：社交语体、频次、场景标签、深度解读、例句等。
        mock 模式返回 None（调用方应中止写库）。
        成功返回与 Postgres words 表扩展列对齐的 dict；解析/校验失败返回 None。
        """
        if self.provider == "mock":
            return None

        prompt = f"""
你是日语词汇专家，面向中文母语者。词条等级: {level}。

词条: {word}
读音: {kana}
已有释义(中文，仅供参考): {meaning_zh}

只输出一个 JSON 对象，不要用 markdown 代码块，不要其它说明文字。字段与类型:
- register_social: string，中文 2~4 句：语体(书面/口语)、适合对谁说、不当使用可能生硬或失礼的情况。
- usage_frequency: integer，1~5（5=日常口语极高频，1=罕用或极书面专语）
- usage_frequency_note: string，中文 4~12 字，如「书面报告常见」
- scene_tags: string 数组，3~6 个极短中文场景标签，如「职场」「邮件」「新闻」
- social_targets: string 数组，2~5 个听话人角色中文词，如「同事」「客户」「朋友」
- offense_risk: integer，0~100，对陌生人/上级用错时的冒犯或失礼风险估计
- scene_deep_dive: string，中文一段 80~200 字：典型场景与文化语感
- example_ja: string，一句自然日语例句（可使用该词的活用形）
- example_zh: string，例句中文翻译
"""

        def _parse_json_text(text: str) -> Dict[str, Any]:
            t = (text or "").strip()
            if "```json" in t:
                t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in t:
                t = t.split("```", 1)[1].split("```", 1)[0].strip()
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
            # Fallback: sanitize control characters inside JSON strings
            def _sanitize_json(s):
                result = []
                in_string = False
                escape_next = False
                for ch in s:
                    if escape_next:
                        result.append(ch)
                        escape_next = False
                        continue
                    if ch == '\\':
                        result.append(ch)
                        escape_next = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        result.append(ch)
                        continue
                    if in_string:
                        if ch == '\n':
                            result.append('\\n')
                        elif ch == '\r':
                            result.append('\\r')
                        elif ch == '\t':
                            result.append('\\t')
                        elif ord(ch) < 32:
                            result.append(' ')
                        else:
                            result.append(ch)
                    else:
                        result.append(ch)
                return ''.join(result)
            try:
                return json.loads(_sanitize_json(t))
            except json.JSONDecodeError:
                pass
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(t)
                return obj
            except json.JSONDecodeError:
                raise

        def _norm_str_list(v: Any, max_n: int) -> List[str]:
            if v is None:
                return []
            if isinstance(v, str):
                parts = [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
                return parts[:max_n]
            if isinstance(v, list):
                out: List[str] = []
                for it in v:
                    if isinstance(it, str) and it.strip():
                        out.append(it.strip())
                    elif isinstance(it, dict):
                        lab = it.get("label") or it.get("name") or it.get("target")
                        if lab and str(lab).strip():
                            out.append(str(lab).strip())
                    if len(out) >= max_n:
                        break
                return out
            return []

        def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
            try:
                x = int(v)
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, x))

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You output JSON only for Japanese CJK learners."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=45,
                )
                text = response.choices[0].message.content or ""
            else:
                def _run_g() -> str:
                    r = self.gemini_model.generate_content(prompt)
                    return (getattr(r, "text", None) or "").strip()

                text = await asyncio.to_thread(_run_g)

            data = _parse_json_text(text)
            register_social = (data.get("register_social") or "").strip()
            scene_deep_dive = (data.get("scene_deep_dive") or "").strip()
            example_ja = (data.get("example_ja") or "").strip()
            example_zh = (data.get("example_zh") or "").strip()
            usage_frequency_note = (data.get("usage_frequency_note") or "").strip()
            if not register_social or not example_ja:
                return None

            return {
                "register_social": register_social,
                "usage_frequency": _clamp_int(data.get("usage_frequency"), 1, 5, 3),
                "usage_frequency_note": usage_frequency_note or "—",
                "scene_tags": _norm_str_list(data.get("scene_tags"), 8) or ["一般"],
                "social_targets": _norm_str_list(data.get("social_targets"), 8) or ["通用"],
                "offense_risk": _clamp_int(data.get("offense_risk"), 0, 100, 0),
                "scene_deep_dive": scene_deep_dive or register_social,
                "example_ja": example_ja,
                "example_zh": example_zh or "",
            }
        except Exception as e:
            print(f"!!! enrich_word_profile ({self.provider}): {type(e).__name__} - {e}")
            return None

    async def enrich_library_entry(
        self,
        word: str,
        reading: str = "",
        meaning: str = "",
        level: str = "N2",
        category: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        For vocab_library: generate social_context (JSON), heatmap_data (JSON), insight_text (TEXT).
        category: None (default), 'slang', 'gaming_lol', 'gaming_valorant' — changes prompt.
        """
        if self.provider == "mock":
            return None

        if category == "slang":
            prompt = self._slang_prompt(word, reading, meaning)
        elif category in ("gaming_lol", "gaming_valorant"):
            game_name = "英雄联盟（League of Legends）" if category == "gaming_lol" else "瓦罗兰特（VALORANT）"
            prompt = self._gaming_prompt(word, reading, meaning, game_name)
        else:
            prompt = self._default_prompt(word, reading, meaning)

        def _parse_json_text(text: str) -> Dict[str, Any]:
            t = (text or "").strip()
            if "```json" in t:
                t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in t:
                t = t.split("```", 1)[1].split("```", 1)[0].strip()
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
            # Fallback: sanitize control characters inside JSON strings
            def _sanitize_json(s):
                result = []
                in_string = False
                escape_next = False
                for ch in s:
                    if escape_next:
                        result.append(ch)
                        escape_next = False
                        continue
                    if ch == '\\':
                        result.append(ch)
                        escape_next = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        result.append(ch)
                        continue
                    if in_string:
                        if ch == '\n':
                            result.append('\\n')
                        elif ch == '\r':
                            result.append('\\r')
                        elif ch == '\t':
                            result.append('\\t')
                        elif ord(ch) < 32:
                            result.append(' ')
                        else:
                            result.append(ch)
                    else:
                        result.append(ch)
                return ''.join(result)
            try:
                return json.loads(_sanitize_json(t))
            except json.JSONDecodeError:
                pass
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(t)
                return obj
            except json.JSONDecodeError:
                raise

        def _clamp_pct(v: Any) -> int:
            try:
                n = int(round(float(v)))
            except Exception:
                return 0
            return max(0, min(100, n))

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=45,
                )
                text = response.choices[0].message.content or ""
            else:
                def _run_g() -> str:
                    r = self.gemini_model.generate_content(prompt)
                    return (getattr(r, "text", None) or "").strip()

                text = await asyncio.to_thread(_run_g)

            data = _parse_json_text(text)
            social_context = data.get("social_context") or {}
            heatmap_data = data.get("heatmap_data") or {}
            insight_text = (data.get("insight_text") or "").strip()
            frequency_stars = data.get("frequency_stars")
            examples = data.get("examples")
            meaning_cn = (data.get("meaning_cn") or "").strip()
            pos_cn = (data.get("pos_cn") or "").strip()
            pitch = data.get("pitch")

            if not isinstance(social_context, dict) or not isinstance(heatmap_data, dict) or not insight_text:
                return None

            # Validate frequency_stars
            try:
                frequency_stars = int(frequency_stars)
                frequency_stars = max(1, min(5, frequency_stars))
            except (TypeError, ValueError):
                frequency_stars = None

            # Validate examples
            if not isinstance(examples, list) or len(examples) < 1:
                examples = None
            else:
                examples = [{"cn": str(e.get("cn","") or e.get("zh","")), "jp": str(e.get("jp","") or e.get("ja",""))} for e in examples[:3] if isinstance(e, dict)]

            # Validate meaning_cn
            if not meaning_cn:
                meaning_cn = None

            # Validate pos_cn
            valid_pos = {"名词","他动词","自动词","自他动词","形容词（い形）","形容动词（な形）","副词","连体词","接续词","感叹词","助词","助动词","接尾词","接头词","惯用表达"}
            if pos_cn not in valid_pos:
                pos_cn = None

            # Validate pitch (0-10 range)
            try:
                pitch = int(pitch)
                pitch = max(0, min(10, pitch))
            except (TypeError, ValueError):
                pitch = None

            # Normalize expected keys for social_context
            for k, ja, zh in (
                ("casual", "カジュアル", "友人・同僚"),
                ("formal", "フォーマル", "店員・陌生人"),
                ("business", "ビジネス", "上司・目上"),
            ):
                v = social_context.get(k) if isinstance(social_context, dict) else None
                if not isinstance(v, dict):
                    v = {}
                allowed = bool(v.get("allowed"))
                reason = str(v.get("reason") or "").strip()
                social_context[k] = {
                    "label_ja": ja,
                    "label_zh": zh,
                    "allowed": allowed,
                    "reason": reason[:60],
                }

            # Normalize heatmap to int percent mapping (top 6)
            hm: Dict[str, int] = {}
            for sk, sv in heatmap_data.items():
                key = str(sk).strip()
                if not key:
                    continue
                hm[key] = _clamp_pct(sv)
            hm = dict(sorted(hm.items(), key=lambda kv: kv[1], reverse=True)[:3])

            return {"social_context": social_context, "heatmap_data": hm, "insight_text": insight_text, "frequency_stars": frequency_stars, "examples": examples, "meaning_cn": meaning_cn, "pos_cn": pos_cn, "pitch": pitch}
        except Exception as e:
            print(f"!!! enrich_library_entry ({self.provider}): {type(e).__name__} - {e}")
            return None

    def _default_prompt(self, word: str, reading: str, meaning: str) -> str:
        return f"""你是东京本地日语老师，专门教中文母语者地道日语。请只输出一个 JSON 对象，不要解释，不要 markdown。

单词: {word}
读音: {reading}
释义: {meaning}

输出字段（严格 JSON）：
1) social_context: 对应 3 个社交场景，allowed 为布尔，reason 为 15~40 字中文解释，要具体到这个词的实际使用语境，不要泛泛而谈：
{{
  "casual": {{"label_ja":"カジュアル","label_zh":"友人・同僚","allowed":true,"reason":"..."}},
  "business": {{"label_ja":"ビジネス","label_zh":"上司・目上","allowed":false,"reason":"..."}},
  "formal": {{"label_ja":"フォーマル","label_zh":"店員・陌生人","allowed":false,"reason":"..."}}
}}
2) heatmap_data: 该词最常出现的 3 个具体场景及真实使用频率（整数 0~100，每个场景独立打分，不要求和为100）。场景名要具体（如 居酒屋、便利店、Line群聊、公司会议、约会、跟邻居唠嗑），不要泛泛的"一般生活"：
{{"具体场景A": 85, "具体场景B": 60, "具体场景C": 30}}
3) insight_text: 中文 100~200字，一段自然的深度解析。像跟朋友聊天一样讲这个词怎么用。只从以下角度挑1-2个展开（别全写，别列清单）：发音是否容易读错、跟近义词的语感差异、背后反映的日本思维习惯、容易闹笑话的坑。不要写"该词""学习者""本词"这类翻译腔，不要在里面写频率和例句（频率和例句已用独立字段输出）。也不用加【】标签，直接写正文段落。

4) frequency_stars: 整数 1~5，代表该词在日常会话中的使用频率（1=几乎不用/冷僻，3=偶尔用到，5=天天挂在嘴边）。

5) examples: 三个地道例句的 JSON 数组 [{{ "cn":"中文翻译1","jp":"日语例句1" }}, ...]。例句必须是日本人真实对话中会说的，不要教科书例句，三个覆盖不同场景。

6) meaning_cn: 将原文释义翻译成地道中文，10~80字。原文即使是英文也必须翻译为中文，禁止返回英文或日文。如果单词是片假名外来语，请在释义末尾用括号标注原词来源（如「蛋糕（cake）」「电脑（computer）」）。

7) pos_cn: 中文词性，从以下选一个最合适的：名词、他动词、自动词、自他动词、形容词（い形）、形容动词（な形）、副词、连体词、接续词、感叹词、助词、助动词、接尾词、接头词、惯用表达。

8) pitch: 日语声调核位置，一个整数。0=平板型（第一个音低，之后全高，无下降），1=头高型（第一个音高，之后全低），2=中高型第2拍后下降，依此类推。参考读音拍数判断。
"""

    def _slang_prompt(self, word: str, reading: str, meaning: str) -> str:
        return f"""你是东京年轻人（20代），深度了解日本网络文化和流行语。请只输出一个 JSON 对象，不要解释，不要 markdown。

流行语: {word}
读音: {reading}
原意: {meaning}

输出字段（严格 JSON）：
1) social_context: 对应 3 个社交场景，allowed 为布尔，reason 为 15~40 字中文解释。流行语通常 casual 为 true，business/formal 为 false：
{{
  "casual": {{"label_ja":"カジュアル","label_zh":"友人・同僚","allowed":true,"reason":"..."}},
  "business": {{"label_ja":"ビジネス","label_zh":"上司・目上","allowed":false,"reason":"..."}},
  "formal": {{"label_ja":"フォーマル","label_zh":"店員・陌生人","allowed":false,"reason":"..."}}
}}
2) heatmap_data: 该词最常出现的 3 个具体场景及真实使用频率（整数 0~100，每个场景独立打分，不要求和为100）。场景名要具体到流行语的使用场景（如 Twitter/X、Instagram、Line群聊、TikTok评论、大学校园、渋谷聚会、居酒屋、バイト先闲聊）：
{{"具体场景A": 85, "具体场景B": 60, "具体场景C": 30}}
3) insight_text: 中文 150~300字，深度解析这个流行语。必须包含两部分：
【原意】30~80字，该词原本在日语中的意思。
【流行语用法】80~200字，作为流行语的具体用法，包括在什么场景/人群中使用、表达什么情绪或态度、近一两年在日本社交网络上的使用趋势。不要写"该词""学习者""本词"这类翻译腔，像跟朋友科普流行梗一样自然地写。

4) frequency_stars: 整数 1~5，代表该词在日本年轻人日常会话中的使用频率。

5) examples: 三个地道例句的 JSON 数组 [{{ "cn":"中文翻译1","jp":"日语例句1" }}, ...]。第一个例句展示传统/原意用法，后两个例句展示流行语用法。例句必须是日本人真实对话中会说的，三个覆盖不同场景。

6) meaning_cn: 将该流行语翻译成地道中文，10~80字。要体现流行语的语感，不要死板直译。

7) pos_cn: 中文词性，从以下选一个最合适的：名词、他动词、自动词、自他动词、形容词（い形）、形容动词（な形）、副词、连体词、接续词、感叹词、助词、助动词、接尾词、接头词、惯用表达。

8) pitch: 日语声调核位置，一个整数。0=平板型（第一个音低，之后全高，无下降），1=头高型（第一个音高，之后全低），2=中高型第2拍后下降，依此类推。参考读音拍数判断。
"""

    def _gaming_prompt(self, word: str, reading: str, meaning: str, game_name: str) -> str:
        return f"""你是日本电竞玩家，深度了解{game_name}的日服术语和中文服术语。请只输出一个 JSON 对象，不要解释，不要 markdown。

游戏术语: {word}
读音: {reading}
英文原词/原意: {meaning}

输出字段（严格 JSON）：
1) social_context: 对应 3 个社交场景，allowed 为布尔，reason 为 15~40 字中文解释。游戏术语 casual/formal 根据实际判断：
{{
  "casual": {{"label_ja":"カジュアル","label_zh":"友人・同僚","allowed":true,"reason":"..."}},
  "business": {{"label_ja":"ビジネス","label_zh":"上司・目上","allowed":false,"reason":"..."}},
  "formal": {{"label_ja":"フォーマル","label_zh":"店員・陌生人","allowed":false,"reason":"..."}}
}}
2) heatmap_data: 该术语最常出现的 3 个具体场景及真实使用频率（整数 0~100，每个场景独立打分，不要求和为100）。场景名要具体（如 游戏语音、Discord群、Twitch直播弹幕、YouTube游戏实况、游戏内聊天、电竞比赛解说）：
{{"具体场景A": 85, "具体场景B": 60, "具体场景C": 30}}
3) insight_text: 中文 150~300字，深度解析这个游戏术语。必须包含以下四部分：
【原意】30~80字，该词在英语/日语中的原本含义。
【游戏中含义】80~150字，在{game_name}中的具体含义，包括在游戏的什么系统/机制中使用、具体指什么。
【游戏中使用方式】50~100字，日本玩家怎么用这个词交流，在什么情境下会说。
【对应中文术语】20~50字，中国玩家对应的叫法是什么（这个很重要，必须准确）。

4) frequency_stars: 整数 1~5，代表该术语在{game_name}玩家中的使用频率。

5) examples: 三个地道例句的 JSON 数组 [{{ "cn":"中文翻译1","jp":"日语例句1" }}, ...]。第一个例句展示日常/普通用法，后两个例句展示游戏对局中的实际用法。例句必须是日本玩家真实对局中会说的话。

6) meaning_cn: 将术语翻译成地道中文，10~80字。必须反映在中文{game_name}中的实际叫法。

7) pos_cn: 中文词性，从以下选一个最合适的：名词、他动词、自动词、自他动词、形容词（い形）、形容动词（な形）、副词、连体词、接续词、感叹词、助词、助动词、接尾词、接头词、惯用表达。

8) pitch: 日语声调核位置，一个整数。0=平板型（第一个音低，之后全高，无下降），1=头高型（第一个音高，之后全低），2=中高型第2拍后下降，依此类推。参考读音拍数判断。
"""

    def _get_mock_analysis(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        kana = dict_info.get("kana", "読み方") if dict_info else "読み方"
        meaning = (
            dict_info.get("meaning", f"词典中关于 '{word}' 的解释")
            if dict_info
            else f"词典中暂无关于 '{word}' 的解释"
        )
        pos = dict_info.get("pos", "名词") if dict_info else "名词"
        
        origin = None
        if all('\u30a0' <= c <= '\u30ff' for c in word):
            origin = "Loanword"

        base = {
            "word": word,
            "kana": kana,
            "romaji": "romaji",
            "pos": pos,
            "pitch": "0",
            "origin": origin,
            "meaning": meaning,
            "frequency": 3,
            "safety": 80,
            "tone": "中性",
            "image": f"https://images.unsplash.com/photo-1528164344705-47542687000d?w=400&h=250&fit=crop&auto=format&q=80",
            "social_targets": [
                {"label": "友人/朋友", "allowed": True},
                {"label": "上司/上司", "allowed": True},
                {"label": "店员/店员", "allowed": True}
            ],
            "contexts": [{"scene": "一般生活", "usage": 100}],
            "examples": [{"jp": f"{word}を勉強しています。", "cn": f"正在学习'{word}'。"}],
            "explanation": "由于未配置 API Key，当前显示的是由本地词典 jamdict 提供的基础信息。为了获得 AI 驱动的深度场景分析、社交建议和动态图片，请在环境变量中配置 GEMINI_API_KEY 或 OPENAI_API_KEY。"
        }
        return base
