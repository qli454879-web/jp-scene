import os
import json
import re
import subprocess
from jamdict import Jamdict
from typing import Optional, Dict, Any

# JMdict_e 文件路径（与 api.py 同目录）
_JMdict_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JMdict_e")

class DictionaryService:
    def __init__(self):
        self._quick_fallback = os.path.exists(_JMdict_PATH)
        if self._quick_fallback:
            print(f"JMdict_e found ({os.path.getsize(_JMdict_PATH)} bytes), quick fallback available.")
        try:
            self.jam = Jamdict()
            # 测试数据库是否真的有数据
            test_result = self.jam.lookup("食べる")
            if test_result.entries:
                self.enabled = True
                print(f"Dictionary service initialized (SQLite), test lookup OK.")
            else:
                raise RuntimeError("Jamdict DB has no entries")
        except Exception as e:
            print(f"Jamdict SQLite unavailable: {e}")
            self.enabled = False
            if self._quick_fallback:
                print("Using XML grep fallback for dictionary lookups.")
            else:
                print("No dictionary available.")

    def _quick_lookup(self, word: str) -> Optional[Dict[str, Any]]:
        """使用 grep 在 JMdict_e XML 文件中快速查找单词释义，不依赖 Jamdict。"""
        try:
            # 搜索 keb(kanji) 或 reb(reading) 精确匹配该单词的 entry
            result = subprocess.run(
                ["grep", "-E", "-m", "1", "-B2", "-A20",
                 f"<keb>{word}</keb>|<reb>{word}</reb>",
                 _JMdict_PATH],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            text = result.stdout

            # 提取 kana/reading
            reb_match = re.search(r"<reb>([^<]+)</reb>", text)
            kana = reb_match.group(1) if reb_match else ""

            # 提取 kanji
            keb_match = re.search(r"<keb>([^<]+)</keb>", text)
            kanji = keb_match.group(1) if keb_match else word

            # 提取英文释义
            glosses = re.findall(r"<gloss[^>]*>([^<]+)</gloss>", text)
            if not glosses:
                return None

            # 提取词性
            pos_tags = re.findall(r"<pos>([^<]+)</pos>", text)

            return {
                "word": word,
                "kanji": kanji,
                "kana": kana,
                "meaning": "; ".join(glosses[:3]),
                "pos": pos_tags[0] if pos_tags else "名词"
            }
        except Exception as e:
            print(f"Quick lookup error: {e}")
            return None

    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            if self._quick_fallback:
                return self._quick_lookup(word)
            return None

        try:
            # First try exact match, then try searching
            result = self.jam.lookup(word)
            if not result.entries:
                return None

            # Find the best match entry
            best_entry = None
            for entry in result.entries:
                # Check if word matches any kanji or kana form
                kanjis = [k.text for k in entry.kanji_forms]
                kanas = [k.text for k in entry.kana_forms]
                if word in kanjis or word in kanas:
                    best_entry = entry
                    break

            if not best_entry:
                best_entry = result.entries[0]

            # Kana and Kanji
            kana = best_entry.kana_forms[0].text if best_entry.kana_forms else ""
            kanji = best_entry.kanji_forms[0].text if best_entry.kanji_forms else ""

            # Senses (meanings)
            senses = []
            for s in best_entry.senses:
                glosses = [g.text for g in s.gloss]
                pos = [p.text for p in s.pos]
                senses.append({
                    "gloss": glosses,
                    "pos": pos
                })

            if not senses:
                return None

            return {
                "word": word,
                "kanji": kanji,
                "kana": kana,
                "meaning": "; ".join(senses[0]["gloss"]),
                "pos": senses[0]["pos"][0] if senses[0]["pos"] else "名词"
            }
        except Exception as e:
            print(f"Dictionary Lookup Error: {e}")
            if self._quick_fallback:
                return self._quick_lookup(word)
            return None
