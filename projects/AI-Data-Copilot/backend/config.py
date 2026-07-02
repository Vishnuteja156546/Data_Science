from pathlib import Path
from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

load_dotenv(BASE_DIR / ".env")


class Settings:
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_upload_rows: int = int(os.getenv("MAX_UPLOAD_ROWS", "100000"))
    max_preview_rows: int = int(os.getenv("MAX_PREVIEW_ROWS", "20"))
    max_sample_rows: int = int(os.getenv("MAX_SAMPLE_ROWS", "8"))
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_api_url: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    max_chat_history: int = int(os.getenv("MAX_CHAT_HISTORY", "8"))
    default_forecast_periods: int = int(os.getenv("DEFAULT_FORECAST_PERIODS", "12"))
    default_automl_mode: str = os.getenv("DEFAULT_AUTOML_MODE", "auto")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    max_plot_points: int = int(os.getenv("MAX_PLOT_POINTS", "200"))
    histogram_bins: int = int(os.getenv("HISTOGRAM_BINS", "20"))


settings = Settings()
