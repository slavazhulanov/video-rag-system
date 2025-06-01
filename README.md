# Video RAG System

Система для поиска и анализа видео с использованием RAG (Retrieval-Augmented Generation).

## Возможности

- Загрузка и обработка видео
- Разбиение на клипы
- Извлечение мультимодальных признаков (видео + аудио)
- Семантический поиск по видео
- Генерация ответов на вопросы о содержимом видео

## Требования

- Python 3.10+
- FFmpeg
- Ollama

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/video-rag-system.git
cd video-rag-system
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv .venv
source .venv/bin/activate  # для Linux/Mac
# или
.venv\Scripts\activate  # для Windows
pip install -r requirements.txt
```

3. Установите FFmpeg:
- MacOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: скачайте с [официального сайта](https://ffmpeg.org/download.html)

4. Установите Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

5. Загрузите модель для генерации ответов:
```bash
ollama pull qwen3:0.6b
```

## Использование

1. Запустите приложение:
```bash
python -m src.interface.app
```

2. Откройте веб-интерфейс по адресу: http://localhost:7860

3. Загрузите видео через веб-интерфейс

4. Дождитесь обработки видео

5. Задавайте вопросы о содержимом видео

## Структура проекта

```
video-rag-system/
├── src/
│   ├── data_preparation/    # Подготовка данных
│   ├── knowledge_extraction/ # Извлечение признаков
│   ├── retrieval/           # Поиск по видео
│   ├── generation/          # Генерация ответов
│   └── interface/           # Веб-интерфейс
├── processed_data/          # Обработанные данные
├── requirements.txt         # Зависимости
└── README.md               # Документация
```

## Технологии

- ImageBind для мультимодальных эмбеддингов
- FAISS для векторного поиска
- Ollama для генерации ответов
- Gradio для веб-интерфейса

## Лицензия

MIT
