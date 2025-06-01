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
        
        self.default_settings = {
            'fps': 10,
            'width': 320,
            'quality': 'medium',
            'max_duration': 10
        }
        
        logger.info(f"Инициализация GifGenerator в {base_dir}")
    
    def create_gif_from_clip(self, clip_path: str, start_time: float, end_time: float, 
                           output_name: str = None, **kwargs) -> str:
        try:
            if not Path(clip_path).exists():
                logger.error(f"Файл клипа не найден: {clip_path}")
                return None
            
            settings = {**self.default_settings, **kwargs}
            
            duration = min(end_time - start_time, settings['max_duration'])
            if duration <= 0:
                logger.warning(f"Недопустимая длительность: {duration}")
                return None
            
            if not output_name:
                clip_name = Path(clip_path).stem
                output_name = f"{clip_name}_{start_time:.1f}_{end_time:.1f}"
            
            gif_path = self.gif_dir / f"{output_name}.gif"
            
            success = self._create_gif_ffmpeg(
                clip_path, gif_path, start_time, duration, settings
            )
            
            if success and gif_path.exists():
                logger.info(f"GIF создан: {gif_path}")
                return str(gif_path)
            else:
                logger.error(f"Ошибка создания GIF: {gif_path}")
                return None
                
        except Exception as e:
            logger.exception(f"Ошибка создания GIF: {e}")
            return None
    
    def create_gifs_from_results(self, search_results: List[Dict[str, Any]], 
                               max_gifs: int = 3) -> List[Dict[str, str]]:
        try:
            gif_info = []
            
            for i, result in enumerate(search_results[:max_gifs]):
                try:
                    metadata = result['metadata']
                    clip_path = metadata.get('clip_path')
                    
                    if not clip_path or not Path(clip_path).exists():
                        logger.warning(f"Недопустимый путь к клипу для результата {i}: {clip_path}")
                        continue
                    
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', start_time + 5)
                    score = result.get('score', 0)
                    
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
                    logger.error(f"Ошибка создания GIF для результата {i}: {e}")
                    continue
            
            logger.info(f"Создано GIF: {len(gif_info)} из {len(search_results)} результатов")
            return gif_info
            
        except Exception as e:
            logger.exception(f"Ошибка пакетного создания GIF: {e}")
            return []
    
    def _create_gif_ffmpeg(self, input_path: str, output_path: str, 
                          start_time: float, duration: float, settings: dict) -> bool:
        try:
            if settings['quality'] == 'high':
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3"
            elif settings['quality'] == 'low':
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=fast_bilinear"
            else:
                vf_options = f"fps={settings['fps']},scale={settings['width']}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_path,
                '-vf', vf_options,
                '-loop', '0',
                str(output_path)
            ]
            
            logger.debug(f"Создание GIF командой: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                file_size = Path(output_path).stat().st_size / (1024 * 1024)
                logger.debug(f"GIF успешно создан. Размер: {file_size:.2f} МБ")
                return True
            else:
                logger.error(f"Ошибка ffmpeg: код {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Таймаут создания GIF")
            return False
        except Exception as e:
            logger.error(f"Ошибка создания GIF: {e}")
            return False
    
    def cleanup_old_gifs(self, max_age_hours: int = 24):
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
                    logger.warning(f"Ошибка удаления {gif_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Удалено старых GIF: {deleted_count}")
                
        except Exception as e:
            logger.error(f"Ошибка очистки GIF: {e}")
    
    def get_gif_info(self, gif_path: str) -> Dict[str, Any]:
        try:
            path = Path(gif_path)
            if not path.exists():
                return None
            
            file_size = path.stat().st_size
            
            try:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', str(path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    
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
                logger.warning(f"Ошибка получения детальной информации о GIF: {e}")
            
            return {
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'path': str(path)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о GIF: {e}")
            return None