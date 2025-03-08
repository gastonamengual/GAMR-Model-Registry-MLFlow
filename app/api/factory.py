from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from app.api.routes import ERROR_TO_HANDLER_MAPPING
from app.api.routes.base import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # for error_exception, error_handler in ERROR_TO_HANDLER_MAPPING:
    #     app.add_exception_handler(error_exception, error_handler)  # type: ignore

    return app
