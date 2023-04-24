"""
file name: service.py

create time: 2023-03-21 10:19
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faceswap.main import router
import uvicorn
from server.app import api_router


def create_app():
    app = FastAPI()

    include_router(app)
    add_cors_middleware(app)


def include_router(app: FastAPI):
    app.include_router(api_router)
    app.include_router(router)


def add_cors_middleware(app: FastAPI):
    origins = ["http://localhost"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = create_app()

if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
