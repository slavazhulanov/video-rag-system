import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GifGenerator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.gif_dir = base_dir / "gifs"
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки по умолчанию для GIF
        self.default_settings = {
            'fps': 10,          # Кадров в секунду
            'width': 320,       # Ширина GIF
            'quality': 'medium', # Качество: 'low', 'medium', 'high'
            'max_duration': 10   # Максимальная длительность GIF в секундах
        }
        
        logger.info(f"GifGenerator initialized at {base_dir}")
    
    def create_gif_from_clip(self, clip_path: str, start_time: float, end_time: float, 
                           output_name: str = None, **kwargs) -> str:
        """
        Создает GIF из видеоклипа
        
        Args:
            clip_path: Путь к видеоклипу
            start_time: Время начала (в секундах от начала клипа)
            end_time: Время окончания
            output_name: Имя выходного файла (без расширения)
            **kwargs: Дополнительные настройки (fps, width, quality)
        
        Returns:
            Путь к созданному GIF файлу
        """
        try:
            if not Path(clip_path).exists():
                logger.error(f"Clip file not found: {clip_path}")
                return None
            
            # Настройки для конвертации
            settings = {**self.default_settings, **kwargs}
            
            # Ограничиваем длительность
            duration = min(end_time - start_time, settings['max_duration'])
            if duration <= 0:
                logger.warning(f"Invalid duration: {duration}")
                return None
            
            # Генерируем имя файла
            if not output_name:
                clip_name = Path(clip_path).stem
                output_name = f"{clip_name}_{start_time:.1f}_{end_time:.1f}"
            
            gif_path = self.gif_dir / f"{output_name}.gif"
            
            # Создаем GIF с помощью ffmpeg
            success = self._create_gif_ffmpeg(
                clip_path, gif_path, start_time, duration, settings
            )
            
            if success and gif_path.exists():
                logger.info(f"Created GIF: {gif_path}")
                return str(gif_path)
            else:
                logger.error(f"Failed to create GIF: {gif_path}")
                return None
                
        except Exception as e:
            logger.exception(f"GIF creation failed: {e}")
            return None
    
    def create_gifs_from_results(self, search_results: List[Dict[str, Any]], 
                               max_gifs: int = 3) -> List[Dict[str, str]]:
        """
        Создает GIF для результатов поиска
        
        Args:
            search_results: Результаты поиска из retriever
            max_gifs: Максимальное количество GIF для создания
        
        Returns:
            List со словарями содержащими информацию о созданных GIF
        """
        try:
            gif_info = []
            
            for i, result in enumerate(search_results[:max_gifs]):
                try:
                    metadata = result['metadata']
                    clip_path = metadata.get('clip_path')
                    
                    if not clip_path or not Path(clip_path).exists():
                        logger.warning(f"Clip path invalid for result {i}: {clip_path}")
                        continue
                    
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', start_time + 5)
                    score = result.get('score', 0)
                    
                    # Создаем GIF для всего клипа (клипы уже короткие - 30 сек)
                    output_name = f"result_{i+1}_score_{score:.3f}"
                    
                    gif_path = self.create_gif_from_clip(
                        clip_path, 0, min(end_time - start_time, 8), 
                        output_name, fps=8, width=280
                    )
                    
                    if gif_path:
                        gif_info.append({
                            'gif_path': gif_path,
                            'original_clip': clip_path,
                            'start_time': start_time,
                            'end_time': end_time,
                            'score': score,
                            'visual_description': metadata.get('visual_description', ''),
                            'transcript': metadata.get('transcript', ''),
                            'source_video': metadata.get('source_video', '')
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to create GIF for result {i}: {e}")
                    continue
            
            logger.info(f"Created {len(gif_info)} GIFs from {len(search_results)} results")
            return gif_info
            
        except Exception as e:
            logger.exception(f"Batch GIF creation failed: {e}")
            return []
    
    def _create_gif_ffmpeg(self, input_path: str, output_path: str, 
                          start_time: float, duration: float, settings: dict) -> bool:
        """
        Создает GIF с помощью ffmpeg с оптимизацией качества
        """
        try:
            # Определяем параметры качества
            if settings['quality'] == 'high':
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3"
            elif settings['quality'] == 'low':
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=fast_bilinear"
            else:  # medium
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
            
            # Команда ffmpeg для создания оптимизированного GIF
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_path,
                '-vf', vf_options,
                '-loop', '0',  # Бесконечный цикл
                str(output_path)
            ]
            
            logger.debug(f"Creating GIF with command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60  # Таймаут 60 секунд
            )
            
            if result.returncode == 0:
                file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
                logger.debug(f"GIF created successfully. Size: {file_size:.2f} MB")
                return True
            else:
                logger.error(f"ffmpeg failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("GIF creation timed out")
            return False
        except Exception as e:
            logger.error(f"GIF creation error: {e}")
            return False
    
    def cleanup_old_gifs(self, max_age_hours: int = 24):
        """Удаляет старые GIF файлы"""
        try:
            import time
            current_time = time.time()
            deleted_count = 0
            
            for gif_file in self.gif_dir.glob("*.gif"):
                try:
                    file_age = current_time - gif_file.stat().st_mtime
                    if file_age > (max_age_hours * 3600):
                        gif_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {gif_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old GIF files")
                
        except Exception as e:
            logger.error(f"GIF cleanup failed: {e}")
    
    def get_gif_info(self, gif_path: str) -> Dict[str, Any]:
        """Получает информацию о GIF файле"""
        try:
            path = Path(gif_path)
            if not path.exists():
                return None
            
            # Получаем размер файла
            file_size = path.stat().st_size
            
            # Пробуем получить информацию через ffprobe
            try:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', str(path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    
                    # Извлекаем полезную информацию
                    duration = float(info.get('format', {}).get('duration', 0))
                    streams = info.get('streams', [])
                    
                    video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
                    width = video_stream.get('width', 0)
                    height = video_stream.get('height', 0)
                    
                    return {
                        'file_size': file_size,
                        'file_size_mb': file_size / (1024 * 1024),
                        'duration': duration,
                        'width': width,
                        'height': height,
                        'path': str(path)
                    }
            except Exception as e:
                logger.warning(f"Could not get detailed GIF info: {e}")
            
            return {
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'path': str(path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get GIF info: {e}")
            return None