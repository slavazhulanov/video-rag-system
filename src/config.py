from pathlib import Path

class Config:
    BASE_DIR = Path("processed_data")
    VIDEO_DIR = BASE_DIR / "video"
    AUDIO_DIR = BASE_DIR / "audio"
    GIF_DIR = BASE_DIR / "gifs"
    INDEX_PATH = BASE_DIR / "video_index"
    MAX_WORKERS = 4
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']