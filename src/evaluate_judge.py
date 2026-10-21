import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

RAW_RESULTS_PATH = Path("results/raw_results.jsonl")
SCORED_PATH = Path("results/scored_results.jsonl")
DETAIL_CSV = Path("results/test_results.csv")
SUMMARY_CSV = Path("results/results.csv")
SUMMARY_CSV_ROOT = Path("results.csv")
BY_CATEGORY_CSV = Path("results/by_category.csv")
FIG_DIR = Path("results/figures")


def record_key(rec: dict) -> str:
    blob = f"{rec['model']}|{rec['id']}|{rec['prompt_id']}|{rec['model_response']}"
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def heuristic_scores(gt: str, ans: str):
    """Локальная эвристическая оценка без API."""

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
        comment = "Ответ частично совпадает с эталоном, но неполный."
    elif ratio >= 0.25:
        comment = "Ответ обрывается или не соответствует эталону."
    else:
        comment = "Ответ не соответствует эталону или модель не ответила."

    return accuracy, hallucination, comment


def main():
    parser = argparse.ArgumentParser(description="Локальная оценка ответов без API")
    parser.add_argument("--all", action="store_true", help="Оценить все записи")
    parser.add_argument("--per-group", type=int, default=10, help="Записей на группу")
    args = parser.parse_args()

    raw = load_jsonl(RAW_RESULTS_PATH)
    if not raw:
        raise SystemExit("[ОШИБКА] results/raw_results.jsonl пуст. Сначала: python src/run_benchmark.py --limit 150")

    done_keys = {r["key"] for r in load_jsonl(SCORED_PATH)}

    if args.all:
        selected = raw
    else:
        groups = {}
        for r in raw:
            groups.setdefault((r["model"], r["category"]), []).append(r)
        import random
        rnd = random.Random(42)
        selected = []
        for items in groups.values():
            rnd.shuffle(items)
            selected.extend(items[: args.per_group])
        selected.sort(key=lambda r: (r["model"], r["id"]))

    todo = [r for r in selected if record_key(r) not in done_keys]
    print(f"[ИНФО] Записей в raw: {len(raw)} | выбрано: {len(selected)} | к оценке: {len(todo)}")

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORED_PATH, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc="Локальная оценка"):
            acc, hal, comment = heuristic_scores(rec["ground_truth"], rec["model_response"])
            out_f.write(json.dumps({
                "key": record_key(rec),
                **{k: rec[k] for k in ("id", "category", "model", "prompt_id", "task_name",
                                       "latency_sec", "ram_mb", "timestamp")},
                "accuracy": acc, "hallucination": hal,
                "judge_comment": comment, "judge_source": "heuristic_local",
            }, ensure_ascii=False) + "\n")

    raw_by_key = {record_key(r): r for r in raw}
    rows = [s for s in load_jsonl(SCORED_PATH) if s["key"] in raw_by_key]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[ОШИБКА] Ни одна запись не оценена.")

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

    print("\n===== СВОДНАЯ ТАБЛИЦА =====")
    print(agg.to_string(index=False))
    print(f"\n[OK] Результаты сохранены: {DETAIL_CSV}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()