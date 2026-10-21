import base64
import json
import time
from pathlib import Path

import psutil
import requests
from tqdm import tqdm

META_PATH = Path("dataset/metadata.json")
PROMPTS_PATH = Path("prompts/multimodal_prompts.json")
OUT_JSONL = Path("results/local_check.jsonl")
OUT_CSV = Path("results/local_check.csv")
SUMMARY_CSV = Path("results/results.csv")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
PER_CATEGORY = 4


def heuristic_scores(gt: str, ans: str):
    def ngrams(s, n=4):
        s = s.lower()
        return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}
    g, a = ngrams(gt), ngrams(ans)
    ratio = len(g & a) / max(1, len(g))
    accuracy = min(5, max(1, round(1 + 4 * ratio)))
    hallucination = 5 if ratio >= 0.75 else (3 if ratio >= 0.35 else 1)
    if ratio >= 0.75:
        comment = "Ответ полностью совпадает с эталоном."
    elif ratio >= 0.5:
        comment = "Ответ частично совпадает с эталоном."
    elif ratio >= 0.25:
        comment = "Ответ обрывается или неполный."
    else:
        comment = "Ответ не соответствует эталону."
    return accuracy, hallucination, comment


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
        raise SystemExit("[ОШИБКА] Ollama не запущена. Запусти приложение Ollama.")
    if not any("moondream" in t for t in tags):
        raise SystemExit("[ОШИБКА] В Ollama нет moondream. Выполни: ollama pull moondream")

    rows = []
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for item in tqdm(chosen, desc="Локальный замер moondream"):
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
            acc, hal, comment = heuristic_scores(item["ground_truth"], ans)
            row = {
                "id": item["id"], "category": item["category"], "prompt_id": p["prompt_id"],
                "task_name": p["task_name"], "model": "Moondream2 (real Ollama)",
                "latency_sec": latency, "ram_mb": ram_mb,
                "accuracy": acc, "hallucination": hal,
                "judge_comment": comment, "model_response": ans[:400],
            }
            rows.append(row)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n===== ЛОКАЛЬНЫЙ ЗАМЕР (реальный moondream) =====")
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