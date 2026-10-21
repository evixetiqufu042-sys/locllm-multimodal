# 🧠 LocLLM — Локальный мультимодальный ассистент

Программный инструмент для автоматизированного анализа визуальных учебных материалов (код, графики, формулы, слайды, схемы) с помощью локальных мультимодальных LLM и системой автоматической оценки качества (LLM-as-a-Judge).

**Проект №2444** | Автор: Эсен Амирхан (группа БПМИ241)

## 🎯 Цель

Разработать пайплайн тестирования локальных мультимодальных LLM (Moondream2, Mistral NeMo Vision, Llama-3-Vision) на 150+ учебных изображениях с автоматической оценкой качества через Gemini API и веб-интерфейсом для демонстрации.

## 📦 Структура репозитория
```
├── dataset/
│ ├── images/ # 150 изображений (code, charts, formulas, slides, diagrams)
│ └── metadata.json # Эталонные ответы (ground truth)
├── prompts/
│ └── multimodal_prompts.json # 15 учебных задач
├── src/
│ ├── run_benchmark.py # Спринт 2: пакетный прогон моделей
│ ├── evaluate_judge.py # Спринт 3: LLM-as-a-Judge (Gemini)
│ ├── local_check.py # Контрольный замер на реальном Ollama
│ └── app.py # Спринт 4: Streamlit веб-интерфейс
├── results/
│ ├── test_results.csv # Детальная таблица (150 записей)
│ ├── results.csv # Сводная таблица по моделям
│ ├── by_category.csv # Разбивка по категориям
│ └── figures/ # Графики сравнения
├── requirements.txt
└── README.md
```


## 🚀 Quick Start

### 1. Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt