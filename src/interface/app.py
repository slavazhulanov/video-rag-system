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
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.video_dir = base_dir / "video"
        self.audio_dir = base_dir / "audio"
        self.gif_dir = base_dir / "gifs"  # Добавляем директорию для GIF
        self.index_path = base_dir / "video_index"
        
        logger.info("Initializing application components")
        
        try:
            self.video_processor = VideoProcessor(base_dir)
            self.extractor = MultimodalExtractor()
            self.vector_store = VectorStore(self.extractor.model)
            self.gif_generator = GifGenerator(base_dir)  # Инициализируем генератор GIF
            
            # Проверяем существование индекса
            if self.index_path.with_suffix('.index').exists():
                logger.info(f"Loading existing index: {self.index_path}")
                try:
                    self.vector_store.load(self.index_path)
                    logger.info("Index loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}")
                    logger.info("Creating new index")
            else:
                logger.info("No existing index found. Creating new index.")
            
            self.retriever = Retriever(self.vector_store)
            self.generator = LLMGenerator(self.retriever)
            self.processed_videos = set()
            
            # Очищаем старые GIF при запуске
            self.gif_generator.cleanup_old_gifs(max_age_hours=24)
            
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.exception("Failed to initialize application")
            raise

    def process_video_only(self, video_path: str) -> str:
        """Обрабатывает видео и добавляет в индекс"""
        try:
            if not video_path or not Path(video_path).exists():
                logger.warning("Invalid video file provided")
                return "❌ Please upload a valid video file first"
            
            video_name = Path(video_path).name
            if video_name in self.processed_videos:
                logger.info(f"Video '{video_name}' already processed")
                return f"✅ Video '{video_name}' already processed"
            
            logger.info(f"Processing video: {video_name}")
            
            # Проверяем поддержку формата
            supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
            if not any(video_name.lower().endswith(fmt) for fmt in supported_formats):
                return f"❌ Unsupported video format. Supported: {', '.join(supported_formats)}"
            
            success = self._process_video_internal(video_path)
            
            if success:
                self.processed_videos.add(video_name)
                logger.info(f"Video '{video_name}' processed successfully")
                return f"✅ Video '{video_name}' processed successfully! You can now search through it."
            else:
                return f"❌ Failed to process video '{video_name}'. Check logs for details."
                
        except Exception as e:
            logger.exception(f"Processing error for {video_path}: {str(e)}")
            return f"❌ Error processing video: {str(e)}"

    def _process_video_internal(self, video_path: str) -> bool:
        """Внутренняя обработка видео"""
        try:
            # Обрабатываем видео на клипы
            clips_info = self.video_processor.process_video(video_path)
            
            if not clips_info:
                logger.error("No clips created from video")
                return False
                
            logger.info(f"Processing {len(clips_info)} clips")
            
            processed_count = 0
            
            for clip_video_path, clip_audio_path in clips_info:
                try:
                    logger.debug(f"Extracting features from clip: {clip_video_path}")
                    
                    # Проверяем существование файла
                    if not Path(clip_video_path).exists():
                        logger.error(f"Clip file not found: {clip_video_path}")
                        continue
                    
                    features = self.extractor.extract_features(clip_video_path)
                    
                    # Проверяем качество извлеченных признаков
                    if features['embeddings'] is None or np.all(features['embeddings'] == 0):
                        logger.warning(f"Empty embeddings for clip: {clip_video_path}")
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
                    logger.debug(f"Added clip to vector store: {clip_video_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing clip {clip_video_path}: {e}")
                    continue
            
            if processed_count == 0:
                logger.error("No clips were processed successfully")
                return False
            
            # Сохраняем индекс
            try:
                self.vector_store.save(self.index_path)
                logger.info(f"Saved vector store index: {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                return False
            
            logger.info(f"Successfully processed {processed_count}/{len(clips_info)} clips")
            return True
            
        except Exception as e:
            logger.exception(f"Internal processing failed: {e}")
            return False

    def search_video(self, video_path: str, query: str) -> tuple[str, str, list]:
        """Поиск по видео с созданием GIF"""
        try:
            if not video_path or not Path(video_path).exists():
                logger.warning("Search called without valid video")
                return "❌ Please upload and process a video first", "", []
            
            if not query.strip():
                logger.warning("Search called without query")
                return "❌ Please enter a question", "", []
            
            video_name = Path(video_path).name
            if video_name not in self.processed_videos:
                logger.warning(f"Search attempted on unprocessed video: {video_name}")
                return f"❌ Video '{video_name}' not processed. Please process it first.", "", []
            
            logger.info(f"Searching for: '{query}' in video: {video_name}")
            
            # Выполняем поиск
            results = self.retriever.search(query, top_k=3)
            
            if not results:
                return "❌ No relevant clips found. Try rephrasing your question.", "", []
            
            # Создаем GIF для найденных результатов
            gif_info = self.gif_generator.create_gifs_from_results(results, max_gifs=3)
            
            # Генерируем ответ
            try:
                answer = self.generator.generate_response(query, results)
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                answer = "✅ Found relevant clips, but answer generation failed. See clips below."
            
            # Формируем информацию о найденных клипах
            fragments_info = []
            for i, result in enumerate(results):
                try:
                    meta = result['metadata']
                    clip_name = Path(meta['clip_path']).name if meta.get('clip_path') else f"Clip {i+1}"
                    
                    fragment = (
                        f"🎬 **{clip_name}**\n"
                        f"⏱️ Time: {meta.get('start_time', 0):.1f}-{meta.get('end_time', 0):.1f}s\n"
                        f"🎯 Relevance: {result.get('score', 0):.3f}\n"
                        f"👁️ Visual: {meta.get('visual_description', 'N/A')[:100]}...\n"
                    )
                    
                    if meta.get('transcript'):
                        fragment += f"🗣️ Audio: {meta['transcript'][:100]}...\n"
                    
                    fragments_info.append(fragment)
                    
                except Exception as e:
                    logger.error(f"Error formatting result {i}: {e}")
                    fragments_info.append(f"🎬 Clip {i+1}: Error formatting result")
            
            fragments_text = "\n" + "="*50 + "\n".join(fragments_info)
            
            # Подготавливаем пути к GIF для отображения
            gif_paths = [info['gif_path'] for info in gif_info if Path(info['gif_path']).exists()]
            
            logger.info(f"Search completed: found {len(results)} results, created {len(gif_paths)} GIFs")
            return answer, fragments_text, gif_paths
            
        except Exception as e:
            logger.exception(f"Search error: {str(e)}")
            return f"❌ Search failed: {str(e)}", "", []

    def get_status(self) -> str:
        """Возвращает статус системы"""
        try:
            processed_count = len(self.processed_videos)
            index_size = self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0
            
            # Считаем количество GIF файлов
            gif_count = len(list(self.gif_dir.glob("*.gif"))) if self.gif_dir.exists() else 0
            
            return (
                f"📊 **System Status**\n"
                f"🎥 Processed videos: {processed_count}\n"
                f"📁 Indexed clips: {index_size}\n"
                f"🎬 Generated GIFs: {gif_count}\n"
                f"💾 Index path: {self.index_path}\n"
            )
        except Exception as e:
            return f"❌ Status error: {str(e)}"

def launch_app():
    """Запуск приложения Gradio"""
    base_dir = Path("processed_data")
    base_dir.mkdir(exist_ok=True)
    (base_dir / "video").mkdir(parents=True, exist_ok=True)
    (base_dir / "audio").mkdir(parents=True, exist_ok=True)
    (base_dir / "gifs").mkdir(parents=True, exist_ok=True)  # Создаем директорию для GIF
    
    logger.info("Launching Gradio application")
    
    try:
        app = VideoRAGApp(base_dir)
    except Exception as e:
        logger.exception("Failed to initialize app")
        raise
    
    # Создаем интерфейс Gradio
    with gr.Blocks(
        title="Video-RAG with GIF Preview",
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
        # 🎥 Video-RAG with GIF Preview
        
        Upload a video, process it, and then ask questions about its content! 
        **Now with animated GIF previews** of the most relevant clips!
        
        **Supported formats:** MP4, AVI, MOV, MKV
        """)
        
        # Статус системы
        with gr.Row():
            status_display = gr.Textbox(
                label="📊 System Status",
                value=app.get_status(),
                interactive=False,
                lines=5
            )
        
        # Раздел загрузки и обработки видео
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.File(
                    label="📁 Upload Video",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    file_count="single"
                )
            with gr.Column(scale=1):
                process_btn = gr.Button("⚡ Process Video", variant="primary", size="lg")
        
        process_status = gr.Textbox(
            label="🔄 Processing Status",
            interactive=False,
            lines=2
        )
        
        gr.Markdown("---")
        
        # Раздел поиска
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="❓ Ask a Question",
                    placeholder="What happens in the video? Who appears? What do they say?",
                    lines=2
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Search", variant="secondary", size="lg")
        
        # Результаты поиска
        with gr.Row():
            with gr.Column(scale=2):
                answer = gr.Textbox(
                    label="💬 Answer",
                    interactive=False,
                    lines=6
                )
            with gr.Column(scale=1):
                fragments = gr.Textbox(
                    label="📝 Clip Details",
                    interactive=False,
                    lines=6
                )
        
        # Галерея GIF
        gr.Markdown("## 🎬 Visual Results")
        gif_gallery = gr.Gallery(
            label="Found Video Clips (as GIFs)",
            show_label=True,
            elem_id="gif_gallery",
            columns=3,
            rows=1,
            height="300px",
            object_fit="contain"
        )
        
        # Привязка событий
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
        
        # Обновление статуса после обработки
        process_btn.click(
            fn=app.get_status,
            outputs=[status_display]
        )
        
        # Добавляем примеры запросов
        gr.Markdown("""
        ### 💡 Example Queries:
        - "What objects are visible in the scene?"
        - "Are there any people in the video?"
        - "What actions or movements happen?"
        - "What is the setting or location?"
        - "What colors are dominant in the video?"
        """)
    
    # Запуск с обработкой ошибок
    try:
        demo.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.exception("Failed to launch Gradio app")
        raise

if __name__ == "__main__":
    launch_app()