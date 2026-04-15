import os
import json
from jamdict import Jamdict
from typing import Optional, Dict, Any

class DictionaryService:
    def __init__(self):
        try:
            # Try to initialize and download if missing
            self.jam = Jamdict()
            self.enabled = True
            print("Dictionary service initialized.")
        except Exception as e:
            print(f"Error initializing jamdict: {e}. Trying to download data...")
            try:
                import jamdict.util
                jamdict.util.download()
                self.jam = Jamdict()
                self.enabled = True
                print("Dictionary data downloaded and service initialized.")
            except Exception as e2:
                print(f"Failed to download dictionary data: {e2}")
                self.enabled = False

    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
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
            return None
