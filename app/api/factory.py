from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ROUTERS


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in ROUTERS:
        app.include_router(router)

    return app
