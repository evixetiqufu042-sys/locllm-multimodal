import base64
import json
import time
from pathlib import Path

import psutil
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
PROMPTS_PATH = Path("prompts/multimodal_prompts.json")
METADATA_PATH = Path("dataset/metadata.json")
DATASET_IMAGES_DIR = Path("dataset/images")

st.set_page_config(page_title="LocLLM: Мультимодальный ассистент", page_icon="🧠", layout="wide")

# --- Hero-секция: объясняем, что это за проект ---
st.title(" LocLLM — Локальный мультимодальный ассистент для учебных материалов")
st.markdown("""
**Проект №2444** | Автор: Эсен Амирхан | Группа БПМИ241

Этот инструмент демонстрирует результаты исследования локальных мультимодальных LLM 
(Moondream2, Mistral NeMo Vision, Llama-3-Vision) на 150+ учебных изображениях. 
Все вычисления выполняются **локально** — данные не покидают устройство.
""")

st.divider()


# --- Проверка Ollama ---
def get_available_models():
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=3)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


models = get_available_models()
if not models:
    st.error("⚠️ Ollama не запущена или недоступна. Запусти приложение Ollama или выполни `ollama serve` в терминале.")
    st.stop()

# --- Загрузка данных ---
prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else []

# Создаём маппинг: путь к картинке -> ground truth
gt_map = {item["image_path"]: item["ground_truth"] for item in metadata}

# --- Боковая панель ---
with st.sidebar:
    st.header("⚙️ Настройки")

    selected_model = st.selectbox("🤖 Локальная модель", options=models, index=0 if models else None)

    # Выбор задачи
    task_choices = {f"[{p['category'].upper()}] {p['task_name']}": p for p in prompts}
    selected_task = st.selectbox(" Учебная задача", options=list(task_choices.keys()))

    temperature = st.slider(" Температура (креативность)", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### 📊 Статистика проекта")
    st.markdown(f"- **150** изображений в датасете")
    st.markdown(f"- **15** учебных задач")
    st.markdown(f"- **3** модели протестированы")
    st.markdown(f"- **LLM-as-a-Judge**: Gemini 3.5 Flash Lite")

    st.markdown("---")
    st.markdown("###  Ключевые результаты")
    st.markdown("""
    | Модель | Accuracy | Latency |
    |--------|----------|---------|
    | Llama-3-Vision | 5.00 | 4.83 сек |
    | Mistral NeMo | 4.72 | 4.33 сек |
    | Moondream2 | 2.84 | 1.62 сек |
    """)

# --- Основная область ---
prompt_obj = task_choices[selected_task]

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### 📋 Выбранная задача: {prompt_obj['task_name']}")
    st.info(f"**Промпт:** {prompt_obj['prompt_text']}")

    # Выбор источника изображения
    source_choice = st.radio("🖼️ Источник изображения", ["Загрузить свой файл", "Выбрать из датасета"], horizontal=True)

    uploaded_file = None
    selected_dataset_image = None

    if source_choice == "Загрузить свой файл":
        uploaded_file = st.file_uploader("📤 Загрузите изображение", type=["png", "jpg", "jpeg"])
    else:
        # Dropdown с картинками из датасета
        image_paths = [str(p) for p in DATASET_IMAGES_DIR.rglob("*.png")]
        if image_paths:
            selected_dataset_image = st.selectbox("📁 Выберите изображение из датасета", image_paths)

with col2:
    st.markdown("### 💡 Подсказка")
    st.markdown("""
    **Как использовать:**
    1. Выберите модель (moondream — быстрая, но слабая)
    2. Выберите задачу (код, график, формула)
    3. Загрузите картинку или выберите из датасета
    4. Нажмите "Проанализировать"

    **Внимание:** Moondream2 на CPU может выдавать бессмысленные ответы — это нормально и демонстрирует ограничения edge-моделей.
    """)

# --- Обработка изображения ---
img_bytes = None
img_path_for_gt = None

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    st.image(uploaded_file, caption="Загруженное изображение", width="stretch")
elif selected_dataset_image is not None:
    img_path = Path(selected_dataset_image)
    if img_path.exists():
        img_bytes = img_path.read_bytes()
        img_path_for_gt = selected_dataset_image
        st.image(img_bytes, caption=f"Из датасета: {img_path.name}", width="stretch")
        # Показываем ground truth
        if img_path_for_gt in gt_map:
            st.markdown("### 🎯 Эталонный ответ (Ground Truth)")
            st.success(gt_map[img_path_for_gt])

# --- Кнопка анализа ---
if st.button("🚀 Проанализировать", type="primary", width="stretch", disabled=(img_bytes is None)):
    if img_bytes is None:
        st.warning("⚠️ Сначала загрузите изображение или выберите из датасета.")
    else:
        with st.spinner("🤔 Модель думает..."):
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            proc = psutil.Process()
            ram_before = proc.memory_info().rss / (1024 * 1024)

            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": selected_model,
                        "prompt": prompt_obj["prompt_text"],
                        "images": [img_b64],
                        "stream": False,
                        "options": {"temperature": temperature, "seed": 42},
                    },
                    timeout=300,
                )
                latency = round(time.perf_counter() - t0, 2)
                ram_after = proc.memory_info().rss / (1024 * 1024)
                answer = resp.json().get("response", "Пустой ответ от модели.").strip()

                # --- Интерпретация качества ответа ---
                answer_length = len(answer)
                is_garbage = answer_length < 50 or answer.count(" ") < 5
                is_repetition = any(word in answer.lower() for word in
                                    ["количество количество", "формулу на изображении", "как изображение"])

                st.divider()
                st.markdown("###  Ответ модели")

                if is_garbage or is_repetition:
                    st.error(f"**⚠️ Низкое качество ответа** (длина: {answer_length} симв.)")
                    st.markdown(f"> {answer}")
                    st.warning("""
                    **Интерпретация:** Модель выдала бессмысленный или зацикленный текст. 
                    Это типичное поведение edge-моделей (1-2B параметров) на CPU без GPU. 
                    В исследовании это классифицируется как **repetition loop** или **prompt echo**.
                    """)
                elif answer_length < 200:
                    st.warning(f"** Краткий ответ** (длина: {answer_length} симв.)")
                    st.markdown(f"> {answer}")
                    st.info("Модель дала короткий ответ. Возможно, она не смогла полностью распознать изображение.")
                else:
                    st.success(f"**✅ Развёрнутый ответ** (длина: {answer_length} симв.)")
                    st.markdown(f"> {answer}")

                # --- Метрики ---
                st.markdown("###  Метрики производительности")
                col1, col2, col3 = st.columns(3)
                col1.metric("⏱ Время ответа", f"{latency} сек")
                col2.metric("💾 RAM приложения", f"{ram_after:.1f} МБ")
                col3.metric("📏 Длина ответа", f"{answer_length} симв.")

                # --- Сравнение с эталоном (если есть) ---
                if img_path_for_gt and img_path_for_gt in gt_map:
                    st.markdown("### 🔍 Сравнение с эталоном")
                    gt = gt_map[img_path_for_gt]
                    col_gt, col_ans = st.columns(2)
                    col_gt.markdown("**🎯 Эталонный ответ:**")
                    col_gt.success(gt)
                    col_ans.markdown("** Ответ модели:**")
                    if is_garbage or is_repetition:
                        col_ans.error(answer)
                    else:
                        col_ans.info(answer)

                    # Простая оценка совпадения
                    gt_lower = gt.lower()
                    ans_lower = answer.lower()
                    overlap = sum(1 for word in ans_lower.split() if word in gt_lower.split())
                    overlap_pct = round(overlap / max(1, len(ans_lower.split())) * 100)
                    st.markdown(f"**Совпадение ключевых слов:** {overlap_pct}%")

            except requests.exceptions.Timeout:
                st.error("⏰ Таймаут: модель не ответила за 5 минут. Возможно, она перегружена.")
            except Exception as e:
                st.error(f" Ошибка при обращении к Ollama: {e}")

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>🧠 LocLLM Project | Исследование локальных мультимодальных LLM для учебных материалов</p>
    <p>Все вычисления выполняются локально через Ollama API | Оценка качества: Gemini 3.5 Flash Lite (LLM-as-a-Judge)</p>
</div>
""", unsafe_allow_html=True)