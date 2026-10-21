import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

RAW_RESULTS_PATH = Path("results/raw_results.jsonl")
SCORED_PATH = Path("results/scored_results.jsonl")
DETAIL_CSV = Path("results/test_results.csv")
SUMMARY_CSV = Path("results/results.csv")
SUMMARY_CSV_ROOT = Path("results.csv")          # дубль в корне: формально закрываем Раздел 4 ТЗ
BY_CATEGORY_CSV = Path("results/by_category.csv")
FIG_DIR = Path("results/figures")

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

SYSTEM_PROMPT = (
    "Ты — беспристрастный эксперт-оценщик (LLM-as-a-Judge) в проекте тестирования мультимодальных LLM. "
    "Тебе дают эталонный ответ (ground truth) и ответ локальной модели на тот же визуальный вопрос. "
    "Оцени ответ модели по двум шкалам от 1 до 5 включительно:\n"
    "- accuracy: 1 — факты полностью неверны; 3 — верно лишь частично; 5 — полностью соответствует эталону.\n"
    "- hallucination: 1 — много выдуманных деталей; 3 — есть отдельные выдумки; 5 — выдумок нет.\n"
    "Верни СТРОГО один JSON-объект без markdown и лишнего текста: "
    '{"accuracy": <int 1-5>, "hallucination": <int 1-5>, "comment": "<1-2 предложения на русском>"}'
)


def record_key(rec: dict) -> str:
    blob = f"{rec['model']}|{rec['id']}|{rec['prompt_id']}|{rec['model_response']}"
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_judge_json(text: str):
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


def heuristic_scores(gt: str, ans: str):
    """Только для режима --offline (проверка пайплайна без сети)."""
    def ngrams(s, n=4):
        s = s.lower()
        return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}
    g, a = ngrams(gt), ngrams(ans)
    ratio = len(g & a) / max(1, len(g))
    accuracy = min(5, max(1, round(1 + 4 * ratio)))
    hallucination = 5 if ratio >= 0.75 else (3 if ratio >= 0.35 else 1)
    return accuracy, hallucination, "Техническая оценка (эвристика по перекрытию с эталоном), без API-судьи."


def call_judge(client, model_tag: str, gt: str, ans: str, retries: int = 4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_tag,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Эталонный ответ:\n{gt}\n\nОтвет локальной модели:\n{ans}"},
                ],
            )
            return parse_judge_json(resp.choices[0].message.content or "")
        except Exception as e:
            msg = str(e).lower()
            retriable = any(t in msg for t in ("429", "503", "quota", "rate", "unavailable",
                                               "timeout", "connection", "temporarily"))
            if not retriable or attempt == retries - 1:
                raise
            wait = min(120, 20 * (2 ** attempt))
            print(f"\n[ПАУЗА] Судья перегружен ({msg[:80]}...). Повтор через {wait} с.")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Спринт 3: LLM-as-a-Judge (Gemini через OpenAI-совместимый API)")
    parser.add_argument("--per-group", type=int, default=10, help="Запросов на одну пару (модель x категория)")
    parser.add_argument("--all", action="store_true", help="Оценить все 450 записей raw_results.jsonl")
    parser.add_argument("--rpm", type=int, default=9, help="Пауза между вызовами под бесплатный лимит Gemini")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--offline", action="store_true", help="Без сети: эвристика, только для проверки пайплайна")
    args = parser.parse_args()

    raw = load_jsonl(RAW_RESULTS_PATH)
    if not raw:
        raise SystemExit("[ОШИБКА] results/raw_results.jsonl пуст. Сначала: python src/run_benchmark.py --limit 150")

    done_keys = {r["key"] for r in load_jsonl(SCORED_PATH)}

    if args.all:
        selected = raw
    else:  # стратифицированная выборка: равномерно по (модель x категория), seed=42
        groups = {}
        for r in raw:
            groups.setdefault((r["model"], r["category"]), []).append(r)
        rnd = random.Random(42)
        selected = []
        for items in groups.values():
            rnd.shuffle(items)
            selected.extend(items[: args.per_group])
        selected.sort(key=lambda r: (r["model"], r["id"]))

    todo = [r for r in selected if record_key(r) not in done_keys]
    print(f"[ИНФО] Записей в raw: {len(raw)} | выбрано: {len(selected)} | уже оценено: {len(selected) - len(todo)} | к отправке: {len(todo)}")

    client = None
    if not args.offline:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("[ОШИБКА] Нет GEMINI_API_KEY. Выполни: export GEMINI_API_KEY='...' или запусти --offline.")
        client = OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
        print(f"[ИНФО] Судья: {args.judge_model} @ {JUDGE_BASE_URL}")

    pause = 60.0 / max(1, args.rpm)
    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    consecutive_fails = 0
    scored_this_run = 0
    with open(SCORED_PATH, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc="LLM-as-a-Judge"):
            if client is not None:
                try:
                    acc, hal, comment = call_judge(client, args.judge_model, rec["ground_truth"], rec["model_response"])
                    source = args.judge_model
                except Exception as e:
                    consecutive_fails += 1
                    print(f"\n[SKIP] id={rec['id']} ({rec['model']}): судья недоступен после ретраев ({str(e)[:100]})")
                    if consecutive_fails >= 3:
                        print("[STOP] 3 записи подряд без судья. Останавливаюсь, чтобы не жечь время. "
                              "Оценённые записи сохранены: перезапусти эту же команду позже (через 30-60 мин) — продолжится.")
                        break
                    continue
                consecutive_fails = 0
            else:
                acc, hal, comment = heuristic_scores(rec["ground_truth"], rec["model_response"])
                source = "heuristic"
            out_f.write(json.dumps({
                "key": record_key(rec),
                **{k: rec[k] for k in ("id", "category", "model", "prompt_id", "task_name",
                                       "latency_sec", "ram_mb", "timestamp")},
                "accuracy": acc, "hallucination": hal,
                "judge_comment": comment, "judge_source": source,
                "scored_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            scored_this_run += 1
            if client is not None:
                time.sleep(pause)
    print(f"[ИНФО] Оценено за этот запуск: {scored_this_run}.")

    raw_by_key = {record_key(r): r for r in raw}
    rows = [s for s in load_jsonl(SCORED_PATH) if s["key"] in raw_by_key]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[ОШИБКА] Оценок судьи пока нет. Перезапусти команду, когда Gemini освободится, "
                         "или проверь пайплайн: python src/evaluate_judge.py --offline")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAIL_CSV, index=False)

    agg = df.groupby("model").agg(
        n_requests=("id", "count"),
        mean_accuracy=("accuracy", "mean"),
        mean_hallucination=("hallucination", "mean"),
        mean_latency_sec=("latency_sec", "mean"),
        mean_ram_mb=("ram_mb", "mean"),
    ).round(2).reset_index().sort_values("mean_accuracy", ascending=False)
    agg.to_csv(SUMMARY_CSV, index=False)
    agg.to_csv(SUMMARY_CSV_ROOT, index=False)

    by_cat = df.groupby(["model", "category"]).agg(
        n=("id", "count"), mean_accuracy=("accuracy", "mean"), mean_hallucination=("hallucination", "mean"),
    ).round(2).reset_index()
    by_cat.to_csv(BY_CATEGORY_CSV, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(axes,
                              ["mean_accuracy", "mean_hallucination", "mean_latency_sec"],
                              ["Точность (1-5, выше лучше)", "Чистота от галлюцинаций (1-5)", "Средняя задержка, сек"]):
        ax.bar(agg["model"], agg[col], color=["#4c72b0", "#55a868", "#c44e52"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    pivot = by_cat.pivot(index="category", columns="model", values="mean_accuracy")
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax2, color=["#4c72b0", "#55a868", "#c44e52"])
    ax2.set_title("Точность по категориям изображений")
    ax2.set_ylabel("mean accuracy (1-5)")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "accuracy_by_category.png", dpi=150)
    plt.close(fig2)

    print("\n===== СВОДНАЯ ТАБЛИЦА (results/results.csv и results.csv) =====")
    print(agg.to_string(index=False))
    print("\n===== TOP-10 ХУДШИХ ОТВЕТОВ (сырьё для анализа паттернов галлюцинаций по ТЗ) =====")
    for _, w in df.sort_values(["accuracy", "hallucination"]).head(10).iterrows():
        print(f"- id={w['id']} [{w['category']}] {w['model']}: acc={w['accuracy']}, hal={w['hallucination']} :: {w['judge_comment']}")
    print(f"\n[OK] Детальная таблица: {DETAIL_CSV} | сводки: {SUMMARY_CSV}, {SUMMARY_CSV_ROOT}, {BY_CATEGORY_CSV} | графики: {FIG_DIR}/")
    if scored_this_run < len(todo):
        print("[ВНИМАНИЕ] Оценены не все выбранные записи: перезапусти ту же команду позже, дооценка продолжится автоматически.")


if __name__ == "__main__":
    main()