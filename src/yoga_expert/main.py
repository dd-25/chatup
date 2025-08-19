from fastapi import FastAPI
from yoga_expert.config import settings
from yoga_expert.routes import health

def create_app() -> FastAPI:
    app = FastAPI(
        title="Yoga-Expert",
        version="0.1.0",
        description="RAG-powered agentic chatbot"
    )
    
    @app.get("/")
    async def root():
        return {"message": "Welcome to Yoga-Expert"}
    
    app.include_router(health.router, prefix="/health", tags=["Health"])

    return app


app = create_app()

def main():
    import uvicorn

    uvicorn.run(
        "yoga_expert.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=True
    )
    
if __name__ == "__main__":
    main()