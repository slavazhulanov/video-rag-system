import gradio as gr
import logging
import numpy as np
from pathlib import Path
from src import VideoProcessor, VectorStore, Retriever, MultimodalExtractor, LLMGenerator, GifGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoRAGApp:
    '''
    Главный класс VideoRAGApp объединяет все компоненты системы и предоставляет веб-интерфейс Gradio.
    '''
    def __init__(self, base_dir: Path):
        # Инициализация всех компонентов системы:
        # - VideoProcessor: обработка видео
        # - MultimodalExtractor: извлечение признаков
        # - VectorStore: векторное хранилище
        # - Retriever: поиск
        # - LLMGenerator: генерация ответов
        # - GifGenerator: создание GIF
        self.base_dir = base_dir
        self.video_dir = base_dir / "video"
        self.audio_dir = base_dir / "audio"
        self.gif_dir = base_dir / "gifs"
        self.index_path = base_dir / "video_index"
        
        logger.info("Инициализация компонентов приложения")
        
        try:
            self.video_processor = VideoProcessor(base_dir)
            self.extractor = MultimodalExtractor()
            self.vector_store = VectorStore(self.extractor.model)
            self.gif_generator = GifGenerator(base_dir)
            
            if self.index_path.with_suffix('.index').exists():
                logger.info(f"Загрузка существующего индекса: {self.index_path}")
                try:
                    self.vector_store.load(self.index_path)
                    logger.info("Индекс успешно загружен")
                except Exception as e:
                    logger.error(f"Ошибка загрузки индекса: {e}")
                    logger.info("Создание нового индекса")
            else:
                logger.info("Существующий индекс не найден. Создание нового индекса")
            
            self.retriever = Retriever(self.vector_store)
            self.generator = LLMGenerator(self.retriever)
            self.processed_videos = set()
            
            self.gif_generator.cleanup_old_gifs(max_age_hours=24)
            
            logger.info("Приложение успешно инициализировано")
            
        except Exception as e:
            logger.exception("Ошибка инициализации приложения")
            raise

    def process_video_only(self, video_path: str) -> str:
        # Полная обработка видео:
        # 1. Валидация формата
        # 2. Разделение на клипы
        # 3. Извлечение признаков из каждого клипа
        # 4. Индексирование в векторном хранилище
        # 5. Сохранение индекса
        try:
            if not video_path or not Path(video_path).exists():
                logger.warning("Недопустимый видео файл")
                return "❌ Пожалуйста, загрузите корректный видео файл"
            
            video_name = Path(video_path).name
            if video_name in self.processed_videos:
                logger.info(f"Видео '{video_name}' уже обработано")
                return f"✅ Видео '{video_name}' уже обработано"
            
            logger.info(f"Обработка видео: {video_name}")
            
            supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
            if not any(video_name.lower().endswith(fmt) for fmt in supported_formats):
                return f"❌ Неподдерживаемый формат видео. Поддерживаемые: {', '.join(supported_formats)}"
            
            success = self._process_video_internal(video_path)
            
            if success:
                self.processed_videos.add(video_name)
                logger.info(f"Видео '{video_name}' успешно обработано")
                return f"✅ Видео '{video_name}' успешно обработано! Теперь можно выполнять поиск."
            else:
                return f"❌ Ошибка обработки видео '{video_name}'. Проверьте логи для деталей."
                
        except Exception as e:
            logger.exception(f"Ошибка обработки для {video_path}: {str(e)}")
            return f"❌ Ошибка обработки видео: {str(e)}"

    def _process_video_internal(self, video_path: str) -> bool:
        try:
            clips_info = self.video_processor.process_video(video_path)
            
            if not clips_info:
                logger.error("Не создано ни одного клипа из видео")
                return False
                
            logger.info(f"Обработка {len(clips_info)} клипов")
            
            processed_count = 0
            
            for clip_video_path, clip_audio_path in clips_info:
                try:
                    logger.debug(f"Извлечение признаков из клипа: {clip_video_path}")
                    
                    if not Path(clip_video_path).exists():
                        logger.error(f"Файл клипа не найден: {clip_video_path}")
                        continue
                    
                    features = self.extractor.extract_features(clip_video_path)
                    
                    if features['embeddings'] is None or np.all(features['embeddings'] == 0):
                        logger.warning(f"Пустые эмбеддинги для клипа: {clip_video_path}")
                        continue
                    
                    metadata = {
                        'source_video': Path(video_path).name,
                        'clip_path': clip_video_path,
                        'audio_path': clip_audio_path,
                        'start_time': features['start_time'],
                        'end_time': features['end_time'],
                        'transcript': features.get('transcript', ''),
                        'visual_description': features.get('visual_description', '')
                    }
                    
                    self.vector_store.add_video(
                        clip_video_path,
                        metadata,
                        features['embeddings']
                    )
                    
                    processed_count += 1
                    logger.debug(f"Клип добавлен в векторное хранилище: {clip_video_path}")
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки клипа {clip_video_path}: {e}")
                    continue
            
            if processed_count == 0:
                logger.error("Ни один клип не был успешно обработан")
                return False
            
            try:
                self.vector_store.save(self.index_path)
                logger.info(f"Индекс векторного хранилища сохранен: {self.index_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения индекса: {e}")
                return False
            
            logger.info(f"Успешно обработано клипов: {processed_count}/{len(clips_info)}")
            return True
            
        except Exception as e:
            logger.exception(f"Внутренняя ошибка обработки: {e}")
            return False

    def search_video(self, video_path: str, query: str) -> tuple[str, str, list]:
        # Поиск и генерация ответа:
        # 1. Поиск релевантных клипов
        # 2. Создание GIF-превью
        # 3. Генерация текстового ответа
        # 4. Форматирование результатов для UI
        try:
            if not video_path or not Path(video_path).exists():
                logger.warning("Поиск вызван без видео")
                return "❌ Пожалуйста, загрузите и обработайте видео сначала", "", []
            
            if not query.strip():
                logger.warning("Поиск вызван без запроса")
                return "❌ Пожалуйста, введите вопрос", "", []
            
            video_name = Path(video_path).name
            if video_name not in self.processed_videos:
                logger.warning(f"Попытка поиска по необработанному видео: {video_name}")
                return f"❌ Видео '{video_name}' не обработано. Пожалуйста, обработайте его сначала.", "", []
            
            logger.info(f"Поиск: '{query}' в видео: {video_name}")
            
            results = self.retriever.search(query, top_k=3)
            
            if not results:
                return "❌ Релевантные клипы не найдены. Попробуйте перефразировать вопрос.", "", []
            
            gif_info = self.gif_generator.create_gifs_from_results(results, max_gifs=3)
            
            try:
                answer = self.generator.generate_response(query, results)
            except Exception as e:
                logger.error(f"Ошибка генерации ответа: {e}")
                answer = "✅ Найдены релевантные клипы, но генерация ответа не удалась. Смотрите клипы ниже."
            
            fragments_info = []
            for i, result in enumerate(results):
                try:
                    meta = result['metadata']
                    clip_name = Path(meta['clip_path']).name if meta.get('clip_path') else f"Клип {i+1}"
                    
                    fragment = (
                        f"🎬 **{clip_name}**\n"
                        f"⏱️ Время: {meta.get('start_time', 0):.1f}-{meta.get('end_time', 0):.1f}с\n"
                        f"🎯 Релевантность: {result.get('score', 0):.3f}\n"
                        f"👁️ Визуальное: {meta.get('visual_description', 'N/A')[:100]}...\n"
                    )
                    
                    if meta.get('transcript'):
                        fragment += f"🗣️ Аудио: {meta['transcript'][:100]}...\n"
                    
                    fragments_info.append(fragment)
                    
                except Exception as e:
                    logger.error(f"Ошибка форматирования результата {i}: {e}")
                    fragments_info.append(f"🎬 Клип {i+1}: Ошибка форматирования")
            
            fragments_text = "\n" + "="*50 + "\n".join(fragments_info)
            
            gif_paths = [info['gif_path'] for info in gif_info if Path(info['gif_path']).exists()]
            
            logger.info(f"Поиск завершен: найдено {len(results)} результатов, создано {len(gif_paths)} GIF")
            return answer, fragments_text, gif_paths
            
        except Exception as e:
            logger.exception(f"Ошибка поиска: {str(e)}")
            return f"❌ Ошибка поиска: {str(e)}", "", []

    def get_status(self) -> str:
        try:
            processed_count = len(self.processed_videos)
            index_size = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            
            gif_count = len(list(self.gif_dir.glob("*.gif"))) if self.gif_dir.exists() else 0
            
            return (
                f"📊 **Статус системы**\n"
                f"🎥 Обработано видео: {processed_count}\n"
                f"📁 Проиндексировано клипов: {index_size}\n"
                f"🎬 Создано GIF: {gif_count}\n"
                f"💾 Путь к индексу: {self.index_path}\n"
            )
        except Exception as e:
            return f"❌ Ошибка статуса: {str(e)}"

def launch_app():
    base_dir = Path("processed_data")
    base_dir.mkdir(exist_ok=True)
    (base_dir / "video").mkdir(parents=True, exist_ok=True)
    (base_dir / "audio").mkdir(parents=True, exist_ok=True)
    (base_dir / "gifs").mkdir(parents=True, exist_ok=True)
    
    logger.info("Запуск Gradio приложения")
    
    try:
        app = VideoRAGApp(base_dir)
    except Exception as e:
        logger.exception("Ошибка инициализации приложения")
        raise
    
    with gr.Blocks(
        title="Video-RAG с предпросмотром GIF",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px;
            margin: auto;
        }
        .gif-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .gif-item {
            max-width: 300px;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 5px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎥 Video-RAG с предпросмотром GIF
        
        Загрузите видео, обработайте его и задавайте вопросы о его содержании! 
        **С анимированными GIF превью** наиболее релевантных клипов!
        
        **Поддерживаемые форматы:** MP4, AVI, MOV, MKV
        """)
        
        with gr.Row():
            status_display = gr.Textbox(
                label="📊 Статус системы",
                value=app.get_status(),
                interactive=False,
                lines=5
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.File(
                    label="📁 Загрузить видео",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    file_count="single"
                )
            with gr.Column(scale=1):
                process_btn = gr.Button("⚡ Обработать видео", variant="primary", size="lg")
        
        process_status = gr.Textbox(
            label="🔄 Статус обработки",
            interactive=False,
            lines=2
        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="❓ Задайте вопрос",
                    placeholder="Что происходит в видео? Кто появляется? Что они говорят?",
                    lines=2
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Поиск", variant="secondary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=2):
                answer = gr.Textbox(
                    label="💬 Ответ",
                    interactive=False,
                    lines=6
                )
            with gr.Column(scale=1):
                fragments = gr.Textbox(
                    label="📝 Детали клипа",
                    interactive=False,
                    lines=6
                )
        
        gr.Markdown("## 🎬 Визуальные результаты")
        gif_gallery = gr.Gallery(
            label="Найденные видео клипы (в формате GIF)",
            show_label=True,
            elem_id="gif_gallery",
            columns=3,
            rows=1,
            height="300px",
            object_fit="contain"
        )
        
        process_btn.click(
            fn=app.process_video_only,
            inputs=[video_input],
            outputs=[process_status]
        )
        
        search_btn.click(
            fn=app.search_video,
            inputs=[video_input, query],
            outputs=[answer, fragments, gif_gallery]
        )
        
        process_btn.click(
            fn=app.get_status,
            outputs=[status_display]
        )
        
        gr.Markdown("""
        ### 💡 Примеры запросов:
        - "Какие объекты видны в сцене?"
        - "Есть ли люди в видео?"
        - "Какие действия или движения происходят?"
        - "Какая обстановка или место действия?"
        - "Какие цвета преобладают в видео?"
        """)
    
    try:
        demo.launch(
            server_name="localhost",
            server_port=7860,
            inbrowser=True,
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.exception("Ошибка запуска Gradio приложения")
        raise

if __name__ == "__main__":
    launch_app()