import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "metadata.json"

def create_chart_image(idx: int, path: Path) -> str:
    years = [2020, 2021, 2022, 2023, 2024]
    base = 10 + (idx % 15)
    step = 5 + (idx % 7)
    values = [base, base + step, base + step * 2, base + step, base + step * 3]
    max_val = max(values)
    min_val = min(values)

    plt.figure(figsize=(6, 4), dpi=120)
    if idx % 2 == 0:
        plt.plot(years, values, marker="o", color="#1f77b4", linewidth=2)
        chart_type = "линейный график"
    else:
        plt.bar([str(y) for y in years], values, color="#2ca02c")
        chart_type = "столбчатая диаграмма"

    plt.title(f"Динамика показателя №{idx} по годам")
    plt.xlabel("Год")
    plt.ylabel("Значение показателя (ед.)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return (
        f"На изображении представлен {chart_type} 'Динамика показателя №{idx} по годам'. "
        f"По оси X отложены годы с 2020 по 2024, по оси Y — значения показателя. "
        f"Минимальное значение составляет {min_val} (в 2020 году), максимальное значение — {max_val} (в 2024 году). "
        f"Значения по годам: 2020={values[0]}, 2021={values[1]}, 2022={values[2]}, 2023={values[3]}, 2024={values[4]}. "
        f"Общий тренд — восходящий с небольшой коррекцией в 2023 году."
    )

def create_code_image(idx: int, path: Path) -> str:
    bugs = [
        (
            f"def calc_average_{idx}(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / 0  # ZeroDivisionError",
            f"Функция calc_average_{idx} содержит ошибку деления на ноль (ZeroDivisionError) в строке 'return total / 0'. Для исправления нужно делить на длину списка: 'return total / len(nums)' при условии, что список не пуст."
        ),
        (
            f"def get_last_item_{idx}(arr):\n    # Возвращает последний элемент\n    return arr[len(arr)]  # IndexError",
            f"В функции get_last_item_{idx} допущена ошибка выхода за границы массива (IndexError): обращение к индексу len(arr). В Python индексация начинается с 0, поэтому нужно использовать 'arr[len(arr) - 1]' или 'arr[-1]'."
        ),
        (
            f"def check_value_{idx}(x):\n    if x = {idx}:\n        return True\n    return False",
            f"В функции check_value_{idx} допущена синтаксическая ошибка (SyntaxError) в условии 'if x = {idx}:'. Использован оператор присваивания '=' вместо оператора сравнения '=='."
        )
    ]
    code_text, gt = bugs[idx % len(bugs)]

    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=120)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.text(0.05, 0.5, code_text, color="#d4d4d4", fontsize=12, family="monospace", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, facecolor=fig.get_facecolor())
    plt.close()
    return gt

def create_formula_image(idx: int, path: Path) -> str:
    formulas = [
        (r"$E = m \cdot c^2 + " + str(idx) + r"$", f"Формула энергии: E = m * c^2 + {idx}, где E — энергия, m — масса, c — скорость света, константа {idx}."),
        (r"$f(x) = \int_{0}^{" + str(idx) + r"} x^2 \, dx$", f"Определенный интеграл функции x^2 на отрезке от 0 до {idx}: \\int_{{0}}^{{{idx}}} x^2 dx."),
        (r"$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$", "Формула дисперсии случайной величины: среднее квадратов отклонений значений x_i от среднего значения mu по выборке размера N.")
    ]
    latex_str, gt = formulas[idx % len(formulas)]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=120)
    ax.text(0.5, 0.5, latex_str, fontsize=20, ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return gt

def create_slide_image(idx: int, path: Path) -> str:
    topics = [
        (
            f"Лекция №{idx}: Локальные LLM в образовании",
            "1. Конфиденциальность: данные не покидают контур вуза.\n2. Автономность: работа без доступа к сети Интернет.\n3. Оптимизация: использование квантования Q4/Q8 снижает VRAM.",
            f"Учебный слайд 'Лекция №{idx}: Локальные LLM в образовании'. Ключевые тезисы: 1) Конфиденциальность данных внутри вуза; 2) Автономная работа без интернета; 3) Снижение потребления видеопамяти (VRAM) за счет квантования Q4/Q8."
        ),
        (
            f"Слайд №{idx}: Метрики оценки качества ИИ",
            "1. Latency — время генерации ответа в секундах.\n2. Точность (Accuracy) — соответствие эталону (Ground Truth).\n3. Галлюцинации — вымышленные факты в ответе модели.",
            f"Слайд №{idx} посвящен метрикам оценки качества ИИ: Latency (скорость генерации ответа), Accuracy (точность по сравнению с эталоном Ground Truth) и уровень галлюцинаций (наличие вымышленных фактов)."
        )
    ]
    title, body, gt = topics[idx % len(topics)]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.text(0.05, 0.82, title, fontsize=14, weight="bold", color="#003366")
    ax.axhline(0.74, color="#003366", linewidth=1.5)
    ax.text(0.05, 0.40, body, fontsize=11, color="#222222", va="center", linespacing=1.6)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return gt

def create_diagram_image(idx: int, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 3), dpi=120)
    steps = ["1. Входные данные\n(Датасет)", f"2. Модуль обработки\n(Пайплайн #{idx})", "3. Оценка качества\n(LLM-Судья)", "4. Итоговый отчет\n(results.csv)"]
    x_positions = [0.12, 0.37, 0.62, 0.87]

    for i, (x, label) in enumerate(zip(x_positions, steps)):
        box = patches.FancyBboxPatch((x - 0.1, 0.35), 0.2, 0.3, boxstyle="round,pad=0.03", facecolor="#e6f2ff", edgecolor="#0055a5", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9, weight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_positions[i+1] - 0.11, 0.5), xytext=(x + 0.11, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#0055a5", lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return (
        f"На блок-схеме изображен последовательный процесс из 4 шагов: "
        f"Шаг 1: Входные данные (Датасет) -> Шаг 2: Модуль обработки (Пайплайн #{idx}) -> "
        f"Шаг 3: Оценка качества (LLM-Судья) -> Шаг 4: Итоговый отчет (results.csv)."
    )

def main():
    categories = [
        ("code", create_code_image, ["code_01", "code_02", "code_03"]),
        ("charts", create_chart_image, ["chart_01", "chart_02", "chart_03"]),
        ("formulas", create_formula_image, ["formula_01", "formula_02", "formula_03"]),
        ("slides", create_slide_image, ["slide_01", "slide_02", "slide_03"]),
        ("diagrams", create_diagram_image, ["diagram_01", "diagram_02", "diagram_03"]),
    ]

    metadata = []
    item_id = 1

    for cat_name, generator_fn, prompt_ids in categories:
        cat_dir = IMAGES_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 31):  # 30 изображений на каждую из 5 категорий = 150 изображений
            file_name = f"{cat_name}_{i:03d}.png"
            file_path = cat_dir / file_name
            gt_text = generator_fn(i, file_path)
            assigned_prompt = prompt_ids[(i - 1) % len(prompt_ids)]

            metadata.append({
                "id": f"{item_id:03d}",
                "category": cat_name,
                "image_path": str(file_path).replace("\\", "/"),
                "default_prompt_id": assigned_prompt,
                "ground_truth": gt_text
            })
            item_id += 1

    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Сгенерировано {len(metadata)} изображений и записан файл {METADATA_FILE}!")

if __name__ == "__main__":
    main()