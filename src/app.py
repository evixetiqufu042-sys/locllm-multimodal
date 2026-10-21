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

st.set_page_config(page_title="LocLLM: Мультимодальный ассистент", page_icon="🧠", layout="wide")
st.title("🧠 LocLLM — Локальный мультимодальный ассистент")
st.caption("Анализ учебных материалов (код, графики, формулы, слайды, схемы) через локальные LLM. Данные не покидают устройство.")

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

# --- Загрузка промтов ---
prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
task_choices = {f"[{p['category'].upper()}] {p['task_name']}": p for p in prompts}

# --- Боковая панель ---
with st.sidebar:
    st.header("⚙️ Настройки")
    selected_model = st.selectbox("Локальная модель", options=models, index=0 if models else None)
    selected_task = st.selectbox("Учебная задача", options=list(task_choices.keys()))
    temperature = st.slider("Температура (креативность)", 0.0, 1.0, 0.2, 0.05)
    st.markdown("---")
    st.markdown("**📊 Статистика проекта:**")
    st.markdown("- 150 изображений в датасете")
    st.markdown("- 15 учебных задач")
    st.markdown("- 3 модели протестированы")
    st.markdown("- LLM-as-a-Judge (Gemini)")

# --- Основная область ---
prompt_obj = task_choices[selected_task]
st.markdown(f"**Задача:** {prompt_obj['task_name']}")
st.markdown(f"*Промпт:* {prompt_obj['prompt_text']}")

uploaded_file = st.file_uploader("📤 Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

    if st.button("🚀 Проанализировать", type="primary", use_container_width=True):
        with st.spinner("Модель думает..."):
            img_bytes = uploaded_file.getvalue()
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

                st.success("✅ Анализ завершён")
                st.markdown("### 📝 Ответ модели:")
                st.markdown(answer)

                col1, col2, col3 = st.columns(3)
                col1.metric("⏱ Время ответа", f"{latency} сек")
                col2.metric("💾 RAM приложения", f"{ram_after:.1f} МБ")
                col3.metric("📏 Длина ответа", f"{len(answer)} симв.")

            except requests.exceptions.Timeout:
                st.error("⏰ Таймаут: модель не ответила за 5 минут.")
            except Exception as e:
                st.error(f"❌ Ошибка при обращении к Ollama: {e}")