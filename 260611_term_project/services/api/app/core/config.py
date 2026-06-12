from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    COMFYUI_URL: str = "http://host.docker.internal:8188"
    WORKFLOW_PATH: str = "app/workflow.json"
    OLLAMA_URL: str = "http://host.docker.internal:11434"

config = Settings()
