import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import requests
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
META_PATH = BASE_DIR / "dataset" / "metadata.json"
PROMPTS_PATH = BASE_DIR / "prompts" / "multimodal_prompts.json"
OUT_JSONL = BASE_DIR / "results" / "local_check.jsonl"
OUT_CSV = BASE_DIR / "results" / "local_check.csv"
SUMMARY_CSV = BASE_DIR / "results" / "results.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
PER_CATEGORY = 4  # 4 изображения x 5 категорий = 20 контрольных запросов

SYSTEM_PROMPT = (
    "Ты — беспристрастный эксперт-оценщик (LLM-as-a-Judge) в проекте тестирования мультимодальных LLM. "
    "Тебе дают эталонный ответ (ground truth) и ответ локальной модели на тот же визуальный вопрос. "
    "Оцени ответ модели по двум шкалам от 1 до 5 включительно:\n"
    "- accuracy: 1 — факты полностью неверны; 3 — верно лишь частично; 5 — полностью соответствует эталону.\n"
    "- hallucination: 1 — много выдуманных деталей; 3 — есть отдельные выдумки; 5 — выдумок нет.\n"
    "Верни СТРОГО один JSON-объект без markdown и лишнего текста: "
    '{"accuracy": <int 1-5>, "hallucination": <int 1-5>, "comment": "<1-2 предложения на русском>"}'
)


def parse_judge_json(text: str):
    import re
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        raise ValueError(f"Судья вернул не-JSON: {text[:200]}")
    acc, hal = int(obj["accuracy"]), int(obj["hallucination"])
    if not (1 <= acc <= 5 and 1 <= hal <= 5):
        raise ValueError("Оценки вне диапазона 1-5")
    return acc, hal, str(obj.get("comment", ""))[:500]


def call_judge(client, gt: str, ans: str, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Эталонный ответ:\n{gt}\n\nОтвет локальной модели:\n{ans}"},
                ],
            )
            return parse_judge_json(resp.choices[0].message.content or "")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(15 * (attempt + 1))


def main():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    pmap = {p["prompt_id"]: p for p in prompts}

    chosen = []
    for cat in ["code", "charts", "formulas", "slides", "diagrams"]:
        chosen.extend([m for m in meta if m["category"] == cat][:PER_CATEGORY])

    try:
        tags = [m.get("name", "") for m in requests.get(OLLAMA_TAGS, timeout=3).json().get("models", [])]
    except Exception:
        sys.exit("[ОШИБКА] Ollama не запущена. Запусти приложение Ollama и повтори.")
    if not any("moondream" in t for t in tags):
        sys.exit("[ОШИБКА] В Ollama нет moondream. Выполни: ollama pull moondream")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("[ОШИБКА] Нет GEMINI_API_KEY: export GEMINI_API_KEY='...'")
    judge = OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)

    rows = []
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for item in tqdm(chosen, desc="Контроль: реальный Ollama moondream"):
            p = pmap[item["default_prompt_id"]]
            img_b64 = base64.b64encode(Path(item["image_path"]).read_bytes()).decode("utf-8")
            t0 = time.perf_counter()
            resp = requests.post(
                OLLAMA_URL,
                json={"model": "moondream", "prompt": p["prompt_text"], "images": [img_b64],
                      "stream": False, "options": {"seed": 42, "temperature": 0.2}},
                timeout=180,
            )
            latency = round(time.perf_counter() - t0, 2)
            ans = resp.json().get("response", "").strip()
            ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
            acc, hal, comment = call_judge(judge, item["ground_truth"], ans)
            row = {
                "id": item["id"], "category": item["category"], "prompt_id": p["prompt_id"],
                "task_name": p["task_name"], "model": "Moondream2 (real Ollama)",
                "latency_sec": latency, "ram_mb": ram_mb,
                "accuracy": acc, "hallucination": hal,
                "judge_comment": comment, "model_response": ans[:400],
            }
            rows.append(row)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(60.0 / 9)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n===== КОНТРОЛЬНЫЙ ЗАМЕР (реальный moondream на твоём железе) =====")
    print(df.groupby("category").agg(
        n=("id", "count"), acc=("accuracy", "mean"), hal=("hallucination", "mean"),
        lat=("latency_sec", "mean")).round(2).to_string())
    print(f"\n[РЕАЛЬНО] acc={df.accuracy.mean():.2f}, hal={df.hallucination.mean():.2f}, lat={df.latency_sec.mean():.2f} s")
    if SUMMARY_CSV.exists():
        s = pd.read_csv(SUMMARY_CSV)
        r = s[s.model == "Moondream2"]
        if not r.empty:
            p = r.iloc[0]
            print(f"[ПРОФИЛЬ ] acc={p.mean_accuracy}, hal={p.mean_hallucination}, lat={p.mean_latency_sec} s (Safe Mode)")
    print(f"\n[OK] Сохранено: {OUT_CSV}, {OUT_JSONL}")


if __name__ == "__main__":
    main()