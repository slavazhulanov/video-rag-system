import os
os.environ['MallocStackLogging'] = '1'
import av
import numpy as np
import subprocess
from typing import List, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    '''
    Класс VideoProcessor отвечает за обработку видеофайлов: разделение на клипы, 
    извлечение аудио и подготовку данных для дальнейшего анализа.
    '''
    def __init__(self, base_dir: Path, max_workers: int = None):
        # Инициализация базовых директорий для хранения обработанных данных
        # max_workers ограничивает количество параллельных процессов
        self.base_dir = base_dir
        self.video_dir = base_dir / "video"
        self.audio_dir = base_dir / "audio"
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_resolution = (640, 360)
        # Используем количество CPU ядер, но не больше 8 для контроля ресурсов
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        logger.info(f"Инициализация VideoProcessor в {base_dir}, workers: {self.max_workers}")

    def process_video(self, video_path: str) -> List[Tuple[str, str]]:
        # Основной метод обработки видео
        # 1. Получает информацию о видео (длительность, наличие аудио)
        # 2. Разделяет на клипы по 30 секунд
        # 3. Обрабатывает клипы параллельно для ускорения
        # Возвращает список кортежей (путь_к_видео_клипу, путь_к_аудио_клипу)
        try:
            logger.info(f"Обработка видео: {video_path}")
            
            # Получаем информацию о видео один раз
            video_info = self._get_video_info(video_path)
            duration = video_info['duration']
            has_audio = video_info['has_audio']
            
            clip_duration = 30.0
            logger.info(f"Длительность видео: {duration:.2f}с, разделение на {duration/clip_duration:.1f} клипов")
            
            # Создаем список задач для параллельной обработки
            clip_tasks = []
            for start in range(0, int(duration), int(clip_duration)):
                end = min(start + clip_duration, duration)
                clip_tasks.append((video_path, start, end, has_audio))
            
            # Обрабатываем клипы параллельно
            clips_info = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = executor.map(self._create_clip_parallel, clip_tasks)
                clips_info = list(results)
            
            # Фильтруем неуспешные результаты
            clips_info = [clip for clip in clips_info if clip[0]]
            
            logger.info(f"Создано клипов: {len(clips_info)}")
            return clips_info
            
        except Exception as e:
            logger.exception(f"Ошибка обработки видео для {video_path}: {str(e)}")
            return []

    def _get_video_info(self, video_path: str) -> dict:
        """Получаем информацию о видео один раз для всех клипов"""
        try:
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == 'video')
                duration = float(video_stream.duration * video_stream.time_base)
                
            has_audio = self._has_audio(video_path)
            
            return {
                'duration': duration,
                'has_audio': has_audio
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о видео {video_path}: {str(e)}")
            raise

    def _create_clip_parallel(self, task_data: Tuple[str, float, float, bool]) -> Tuple[str, str]:
        """Версия _create_clip для параллельного выполнения"""
        video_path, start, end, has_audio = task_data
        return self._create_clip_optimized(video_path, start, end, has_audio)
        
    def _create_clip_optimized(self, video_path: str, start: float, end: float, has_audio: bool) -> Tuple[str, str]:
        # Оптимизированное создание клипов с помощью FFmpeg
        # Использует быстрые пресеты для ускорения обработки
        # Параллельно извлекает видео и аудио если есть звук
        base_filename = self._generate_clip_filename(video_path, start, end)
        clip_path = self.video_dir / f"{base_filename}.mp4"
        audio_path = self.audio_dir / f"{base_filename}.wav"
        
        logger.debug(f"Создание клипа: {clip_path}")
        
        try:
            # Оптимизированная команда FFmpeg для видео
            video_cmd = [
                'ffmpeg', '-y', 
                '-ss', str(start), 
                '-to', str(end),
                '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Изменено с veryfast на ultrafast
                '-crf', '30',  # Увеличено с 28 для быстрой обработки
                '-tune', 'fastdecode',
                '-movflags', '+faststart',
                '-vf', f'scale={self.target_resolution[0]}:{self.target_resolution[1]}:flags=fast_bilinear,format=yuv420p',
                '-threads', '2',  # Ограничиваем потоки для каждого процесса
                '-an',  # Без аудио в видео
                str(clip_path)
            ]
            
            # Обрабатываем видео и аудио одновременно, если есть аудио
            if has_audio:
                audio_cmd = [
                    'ffmpeg', '-y', 
                    '-ss', str(start), 
                    '-to', str(end),
                    '-i', video_path, 
                    '-ac', '1', 
                    '-ar', '16000',
                    '-c:a', 'pcm_s16le',
                    '-threads', '1',
                    str(audio_path)
                ]
                
                # Запускаем обе команды параллельно
                video_process = subprocess.Popen(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Ждем завершения обеих команд
                video_result = video_process.communicate()
                audio_result = audio_process.communicate()
                
                if video_process.returncode != 0:
                    raise subprocess.CalledProcessError(video_process.returncode, video_cmd, video_result[1])
                if audio_process.returncode != 0:
                    raise subprocess.CalledProcessError(audio_process.returncode, audio_cmd, audio_result[1])
                    
                logger.debug(f"Видео и аудио клипы созданы: {clip_path}, {audio_path}")
                return str(clip_path), str(audio_path)
            else:
                # Только видео
                result = subprocess.run(video_cmd, capture_output=True, check=True)
                logger.debug(f"Видео клип создан: {clip_path}")
                return str(clip_path), ""
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка команды FFmpeg: {e.cmd}")
            if e.stderr:
                logger.error(f"Ошибка FFmpeg: {e.stderr.decode().strip()}")
            raise
        except Exception as e:
            logger.exception(f"Ошибка создания клипа для {video_path} [{start}-{end}с]: {str(e)}")
            raise

    def _has_audio(self, video_path: str) -> bool:
        try:
            # Используем более быстрый способ проверки
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=codec_type', 
                   '-select_streams', 'a:0', '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            has_audio = result.stdout.strip() == 'audio'
            logger.debug(f"Проверка аудио для {video_path}: {has_audio}")
            return has_audio
        except Exception as e:
            logger.warning(f"Ошибка проверки аудио для {video_path}: {str(e)}")
            return False

    def _generate_clip_filename(self, video_path: str, start: float, end: float) -> str:
        base_name = Path(video_path).stem
        return f"{base_name}_{start:.1f}_{end:.1f}"

    def process_multiple_videos(self, video_paths: List[str]) -> dict:
        """Обработка нескольких видео с общим пулом потоков"""
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self.process_video, video_path): video_path 
                for video_path in video_paths
            }
            
            for future in future_to_video:
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    all_results[video_path] = result
                except Exception as e:
                    logger.error(f"Ошибка обработки {video_path}: {str(e)}")
                    all_results[video_path] = []
        
        return all_results