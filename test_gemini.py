import google.generativeai as genai
import os

key = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=key)

try:
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
