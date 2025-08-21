from fastapi import FastAPI
from chatup.config import settings
from chatup.routes import health, upload

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="RAG-powered agentic chatbot"
    )
    
    @app.get("/")
    async def root():
        return {"message": "Welcome to Chatup"}
    
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(upload.router, prefix="/upload", tags=["Upload"])

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