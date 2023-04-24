"""
file name: __init__.py

create time: 2023-03-29 14:46
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
from fastapi import APIRouter

from server.app import service

api_router = APIRouter()

api_router.include_router(service.service_router, prefix="", tags=["service"])
