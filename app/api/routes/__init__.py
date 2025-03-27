from .base import router as home_router
from .model import router as model_router

ROUTERS = [home_router, model_router]

__all__ = ["ROUTERS"]
