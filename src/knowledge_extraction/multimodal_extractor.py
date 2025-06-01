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
    '''
    Класс MultimodalExtractor извлекает мультимодальные признаки (визуальные + аудио) из видеоклипов с помощью модели ImageBind.
    '''
    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.info(f"Инициализация MultimodalExtractor на устройстве: {device}")
        
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval().to(self.device)
        self.model.device = self.device
        logger.info("Модель ImageBind загружена")

    def extract_features(self, video_path: str) -> Dict[str, Any]:
        # 1. Загружает и преобразует видеоданные для модели ImageBind
        # 2. Извлекает аудио во временный файл при наличии
        # 3. Получает эмбеддинги для визуальной и аудио модальностей
        # 4. Комбинирует эмбеддинги (среднее арифметическое)
        # 5. Возвращает 1024-мерный вектор признаков
        try:
            logger.info(f"Извлечение признаков из: {video_path}")
            
            duration = self._get_video_duration(video_path)
            logger.debug(f"Длительность видео: {duration:.2f} секунд")
            
            inputs = {}
            
            try:
                video_input = data.load_and_transform_video_data([video_path], self.device)
                inputs[ModalityType.VISION] = video_input
                logger.debug("Видео данные успешно загружены")
            except Exception as e:
                logger.error(f"Ошибка загрузки видео данных: {e}")
                return self._create_empty_features(duration)
            
            audio_path = None
            if self._has_audio(video_path):
                try:
                    audio_path = self._extract_audio(video_path)
                    if audio_path:
                        audio_input = data.load_and_transform_audio_data([audio_path], self.device)
                        inputs[ModalityType.AUDIO] = audio_input
                        logger.debug("Аудио данные успешно загружены")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки аудио данных: {e}")
            
            with torch.no_grad():
                embeddings = self.model(inputs)
                
                visual_emb = self._extract_safe_embedding(embeddings, ModalityType.VISION)
                audio_emb = self._extract_safe_embedding(embeddings, ModalityType.AUDIO)
                
                if audio_emb is not None:
                    combined_emb = (visual_emb + audio_emb) / 2
                    logger.debug("Комбинирование визуальных и аудио эмбеддингов")
                else:
                    combined_emb = visual_emb
                    logger.debug("Использование только визуальных эмбеддингов")
            
            self._cleanup_temp_file(audio_path)
            
            return {
                "embeddings": combined_emb,
                "start_time": 0.0,
                "end_time": duration,
                "transcript": "",
                "visual_description": self._generate_visual_description(combined_emb)
            }
            
        except Exception as e:
            logger.exception(f"Ошибка извлечения признаков для {video_path}: {str(e)}")
            return self._create_empty_features(0.0)
    
    def _extract_safe_embedding(self, embeddings: Dict, modality) -> np.ndarray:
        # Безопасное извлечение эмбеддингов с проверкой типов
        # Нормализация размерности до 1024 (обрезка или дополнение нулями)
        # Преобразование в float32 для совместимости с FAISS
        try:
            if modality not in embeddings:
                logger.debug(f"Эмбеддинги для {modality} не найдены")
                return None
            
            emb = embeddings[modality]
            
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
                logger.error(f"Неподдерживаемый тип эмбеддинга для {modality}: {type(emb)}")
                return None
            
            if emb_np.ndim > 1:
                emb_np = emb_np.flatten()
            
            if len(emb_np) != 1024:
                if len(emb_np) > 1024:
                    emb_np = emb_np[:1024]
                else:
                    padded = np.zeros(1024)
                    padded[:len(emb_np)] = emb_np
                    emb_np = padded
            
            logger.debug(f"Форма эмбеддинга {modality}: {emb_np.shape}")
            return emb_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддинга {modality}: {e}")
            return None
    
    def _get_video_duration(self, video_path: str) -> float:
        try:
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == 'video')
                return float(video_stream.duration * video_stream.time_base)
        except Exception as e:
            logger.error(f"Ошибка получения длительности видео: {e}")
            return 0.0
    
    def _has_audio(self, video_path: str) -> bool:
        try:
            with av.open(video_path) as container:
                return any(s.type == 'audio' for s in container.streams)
        except Exception:
            return False
    
    def _extract_audio(self, video_path: str) -> str:
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
                logger.debug(f"Аудио извлечено в: {audio_path}")
                return audio_path
            else:
                logger.error(f"Ошибка FFmpeg: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка извлечения аудио: {e}")
            return None
    
    def _cleanup_temp_file(self, file_path: str):
        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
                logger.debug("Временный аудио файл удален")
            except Exception as e:
                logger.warning(f"Ошибка удаления временного файла: {e}")
    
    def _create_empty_features(self, duration: float) -> Dict[str, Any]:
        return {
            "embeddings": np.zeros(1024, dtype=np.float32),
            "start_time": 0.0,
            "end_time": duration,
            "transcript": "",
            "visual_description": "Ошибка обработки"
        }
    
    def _generate_visual_description(self, visual_emb: np.ndarray) -> str:
        try:
            if visual_emb is None or len(visual_emb) == 0:
                return "Визуальные признаки отсутствуют"
            
            top_indices = np.argsort(np.abs(visual_emb))[-5:]
            dominant_feature = top_indices[-1]
            
            if dominant_feature < 256:
                category = "Сцена/Фон"
            elif dominant_feature < 512:
                category = "Объекты/Люди"
            elif dominant_feature < 768:
                category = "Движение/Действие"
            else:
                category = "Визуальные эффекты"
            
            return f"{category} (признак #{dominant_feature})"
        except Exception as e:
            logger.error(f"Ошибка генерации описания: {e}")
            return "Описание недоступно"