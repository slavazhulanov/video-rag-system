from typing import Dict, Any
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from pathlib import Path
import logging
import av
import tempfile
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.info(f"Initializing MultimodalExtractor on device: {device}")
        
        # Инициализация модели ImageBind
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval().to(self.device)
        # Добавляем свойство device к модели для использования в retriever
        self.model.device = self.device
        logger.info("Loaded ImageBind model")

    def extract_features(self, video_path: str) -> Dict[str, Any]:
        try:
            logger.info(f"Extracting features from: {video_path}")
            
            # Получаем длительность видео
            duration = self._get_video_duration(video_path)
            logger.debug(f"Video duration: {duration:.2f} seconds")
            
            # Готовим данные для ImageBind
            inputs = {}
            
            # Загружаем видео
            try:
                video_input = data.load_and_transform_video_data([video_path], self.device)
                inputs[ModalityType.VISION] = video_input
                logger.debug("Successfully loaded video data")
            except Exception as e:
                logger.error(f"Failed to load video data: {e}")
                return self._create_empty_features(duration)
            
            # Загружаем аудио, если оно есть
            audio_path = None
            if self._has_audio(video_path):
                try:
                    audio_path = self._extract_audio(video_path)
                    if audio_path:
                        audio_input = data.load_and_transform_audio_data([audio_path], self.device)
                        inputs[ModalityType.AUDIO] = audio_input
                        logger.debug("Successfully loaded audio data")
                except Exception as e:
                    logger.warning(f"Failed to load audio data: {e}")
            
            # Обрабатываем через ImageBind
            with torch.no_grad():
                embeddings = self.model(inputs)
                
                # Безопасная обработка визуальных эмбеддингов
                visual_emb = self._extract_safe_embedding(embeddings, ModalityType.VISION)
                
                # Безопасная обработка аудио эмбеддингов
                audio_emb = self._extract_safe_embedding(embeddings, ModalityType.AUDIO)
                
                # Комбинируем эмбеддинги
                if audio_emb is not None:
                    combined_emb = (visual_emb + audio_emb) / 2
                    logger.debug("Combined visual and audio embeddings")
                else:
                    combined_emb = visual_emb
                    logger.debug("Using only visual embeddings")
            
            # Очищаем временные файлы
            self._cleanup_temp_file(audio_path)
            
            return {
                "embeddings": combined_emb,
                "start_time": 0.0,
                "end_time": duration,
                "transcript": "",
                "visual_description": self._generate_visual_description(combined_emb)
            }
            
        except Exception as e:
            logger.exception(f"Feature extraction failed for {video_path}: {str(e)}")
            return self._create_empty_features(0.0)
    
    def _extract_safe_embedding(self, embeddings: Dict, modality) -> np.ndarray:
        """Безопасно извлекает эмбеддинг из результата модели"""
        try:
            if modality not in embeddings:
                logger.debug(f"No {modality} embeddings found")
                return None
            
            emb = embeddings[modality]
            
            # Обработка различных форматов
            if isinstance(emb, torch.Tensor):
                emb_np = emb.cpu().numpy()
            elif isinstance(emb, list):
                if len(emb) > 0 and isinstance(emb[0], torch.Tensor):
                    emb_np = emb[0].cpu().numpy()
                else:
                    emb_np = np.array(emb)
            elif isinstance(emb, np.ndarray):
                emb_np = emb
            else:
                logger.error(f"Unexpected embedding type for {modality}: {type(emb)}")
                return None
            
            # Приводим к правильной размерности
            if emb_np.ndim > 1:
                emb_np = emb_np.flatten()
            
            # Нормализуем размер до 1024 (стандартный размер ImageBind)
            if len(emb_np) != 1024:
                if len(emb_np) > 1024:
                    emb_np = emb_np[:1024]
                else:
                    # Дополняем нулями до 1024
                    padded = np.zeros(1024)
                    padded[:len(emb_np)] = emb_np
                    emb_np = padded
            
            logger.debug(f"{modality} embedding shape: {emb_np.shape}")
            return emb_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract {modality} embedding: {e}")
            return None
    
    def _get_video_duration(self, video_path: str) -> float:
        """Получает длительность видео"""
        try:
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == 'video')
                return float(video_stream.duration * video_stream.time_base)
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0
    
    def _has_audio(self, video_path: str) -> bool:
        """Проверяет наличие аудио в видео"""
        try:
            with av.open(video_path) as container:
                return any(s.type == 'audio' for s in container.streams)
        except Exception:
            return False
    
    def _extract_audio(self, video_path: str) -> str:
        """Извлекает аудио во временный файл"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                audio_path = tmpfile.name
            
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-ac', '1', '-ar', '16000',
                '-loglevel', 'error', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                logger.debug(f"Extracted audio to: {audio_path}")
                return audio_path
            else:
                logger.error(f"FFmpeg failed: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _cleanup_temp_file(self, file_path: str):
        """Удаляет временный файл"""
        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
                logger.debug("Deleted temporary audio file")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
    
    def _create_empty_features(self, duration: float) -> Dict[str, Any]:
        """Создает пустые признаки при ошибке"""
        return {
            "embeddings": np.zeros(1024, dtype=np.float32),
            "start_time": 0.0,
            "end_time": duration,
            "transcript": "",
            "visual_description": "Processing failed"
        }
    
    def _generate_visual_description(self, visual_emb: np.ndarray) -> str:
        """Генерирует простое описание на основе эмбеддинга"""
        try:
            if visual_emb is None or len(visual_emb) == 0:
                return "No visual features"
            
            # Простая эвристика на основе доминантных признаков
            top_indices = np.argsort(np.abs(visual_emb))[-5:]
            dominant_feature = top_indices[-1]
            
            # Простая категоризация
            if dominant_feature < 256:
                category = "Scene/Background"
            elif dominant_feature < 512:
                category = "Objects/People"
            elif dominant_feature < 768:
                category = "Motion/Action"
            else:
                category = "Visual Effects"
            
            return f"{category} (feature #{dominant_feature})"
        except Exception as e:
            logger.error(f"Failed to generate visual description: {e}")
            return "Description unavailable"