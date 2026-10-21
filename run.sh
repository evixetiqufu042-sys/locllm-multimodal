#!/bin/bash

# Устанавливаем UTF-8 локаль
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "🚀 Запускаем LocLLM..."

# Проверяем, установлен ли virtual environment
if [ ! -d "venv" ]; then
    echo " Создаём виртуальное окружение..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Проверяем, запущена ли Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama не запущена!"
    echo "Запусти приложение Ollama или выполни: ollama serve"
    exit 1
fi

# Запускаем Streamlit
echo "✅ Запускаем веб-интерфейс..."
streamlit run src/app.py
