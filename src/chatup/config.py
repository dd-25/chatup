from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    OPENAI_API_KEY: str
    COHERE_API_KEY: str = ""  # Optional for reranking
    
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX_NAME: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
