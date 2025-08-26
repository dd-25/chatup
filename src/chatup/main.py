import logging
from fastapi import FastAPI
from chatup.config import settings
from chatup.routes import health, upload, query
from chatup.constants import APP_SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_SETTINGS.APP_NAME,
        version=APP_SETTINGS.VERSION,
        description=APP_SETTINGS.DESCRIPTION
    )
    
    @app.get("/")
    async def root():
        return {"message": "Welcome to Chatup"}
    
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(upload.router, prefix="/upload", tags=["Upload"])
    app.include_router(query.router, prefix="/query", tags=["Query"])

    return app


app = create_app()

def main():
    import uvicorn

    uvicorn.run(
        "chatup.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=True
    )
    
if __name__ == "__main__":
    main()