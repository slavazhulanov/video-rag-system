import os
os.environ['MallocStackLogging'] = '1'
import av
import numpy as np
import subprocess
from typing import List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.video_dir = base_dir / "video"
        self.audio_dir = base_dir / "audio"
        
        # Создаем директории при инициализации
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_resolution = (640, 360)
        logger.info(f"VideoProcessor initialized at {base_dir}")

    def process_video(self, video_path: str) -> List[Tuple[str, str]]:
        try:
            logger.info(f"Processing video: {video_path}")
            clips_info = []
            
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == 'video')
                duration = float(video_stream.duration * video_stream.time_base)
                clip_duration = 30.0
                logger.info(f"Video duration: {duration:.2f}s, splitting into {duration/clip_duration:.1f} clips")
                
                for start in range(0, int(duration), int(clip_duration)):
                    end = min(start + clip_duration, duration)
                    logger.debug(f"Creating clip: {start:.1f}-{end:.1f}s")
                    clip_path, audio_path = self._create_clip(video_path, start, end)
                    clips_info.append((clip_path, audio_path))
            
            logger.info(f"Created {len(clips_info)} clips")
            return clips_info
        except Exception as e:
            logger.exception(f"Video processing failed for {video_path}: {str(e)}")
            return []
        
    def _create_clip(self, video_path: str, start: float, end: float) -> Tuple[str, str]:
        base_filename = self._generate_clip_filename(video_path, start, end)
        clip_path = self.video_dir / f"{base_filename}.mp4"
        audio_path = self.audio_dir / f"{base_filename}.wav"
        
        logger.debug(f"Creating clip: {clip_path}")
        
        try:
            # Create video clip
            cmd = [
                'ffmpeg', '-y', 
                '-ss', str(start), 
                '-to', str(end),
                '-i', video_path, 
                '-c:v', 'libx264', 
                '-preset', 'veryfast',
                '-crf', '28',
                '-tune', 'fastdecode',
                '-movflags', '+faststart',
                '-vf', f'scale={self.target_resolution[0]}:{self.target_resolution[1]},format=yuv420p',
                '-an', 
                str(clip_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Created video clip: {clip_path}")
            
            # Extract audio if available
            if self._has_audio(video_path):
                cmd = [
                    'ffmpeg', '-y', 
                    '-ss', str(start), 
                    '-to', str(end),
                    '-i', video_path, 
                    '-ac', '1', 
                    '-ar', '16000',
                    '-c:a', 'pcm_s16le',
                    '-sample_fmt', 's16',
                    str(audio_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.debug(f"Created audio clip: {audio_path}")
                return str(clip_path), str(audio_path)
            
            return str(clip_path), ""
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.cmd}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode().strip()}")
            raise
        except Exception as e:
            logger.exception(f"Clip creation failed for {video_path} [{start}-{end}s]: {str(e)}")
            raise

    def _has_audio(self, video_path: str) -> bool:
        try:
            cmd = ['ffprobe', '-i', video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
            result = subprocess.run(cmd, capture_output=True, text=True)
            has_audio = 'codec_type=audio' in result.stdout
            logger.debug(f"Audio check for {video_path}: {has_audio}")
            return has_audio
        except Exception as e:
            logger.warning(f"Audio check failed for {video_path}: {str(e)}")
            return False

    def _generate_clip_filename(self, video_path: str, start: float, end: float) -> str:
        base_name = Path(video_path).stem
        return f"{base_name}_{start:.1f}_{end:.1f}"