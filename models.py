import httpx
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(f"GEMINI_API_KEY is missing from the environment")


def list_available_models(api_key: str):
    model_url = "https://generativelanguage.googleapis.com/v1beta/models"
    with httpx.Client(timeout=10.0) as http_client:
        resp = http_client.get(url=model_url, headers={"x-goog-api-key": api_key})
        resp.raise_for_status()
        data = resp.json()

    print("Models supporting generativeContent:")
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(" -", m["name"])
            
    print("\nModels supporting embedContent:")
    for m in data.get("models", []):
        if "embedContent" in m.get("supportedGenerationMethods", []):
            print(" -", m["name"])


list_available_models(GEMINI_API_KEY)