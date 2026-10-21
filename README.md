# 🧠 LocLLM — Локальный мультимодальный ассистент

Программный инструмент для автоматизированного анализа визуальных учебных материалов (код, графики, формулы, слайды, схемы) с помощью локальных мультимодальных LLM и системой автоматической оценки качества (LLM-as-a-Judge).

**Проект №2444**

## 🎯 Цель

Разработать пайплайн тестирования локальных мультимодальных LLM (Moondream2, Mistral NeMo Vision, Llama-3-Vision) на 150+ учебных изображениях с автоматической оценкой качества через Gemini API и веб-интерфейсом для демонстрации.

## 📦 Структура репозитория
## 📦 Структура репозитория

```
├── dataset/
│   ├── images/           # 150 изображений
│   └── metadata.json     # Эталонные ответы
├── prompts/
│   └── multimodal_prompts.json   # 15 учебных задач
├── src/
│   ├── run_benchmark.py
│   ├── evaluate_judge.py
│   ├── local_check.py
│   └── app.py
── results/
│   ├── test_results.csv
│   ├── results.csv
│   └── figures/
├── screenshots/
│   ├── app_main.png
│   └── app_response.png
├── REPORT.md
├── requirements.txt
└── README.md
```


## 🚀 Quick Start

### 1. Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

### Для Mac/Linux:
```bash
chmod +x run.sh
./run.sh