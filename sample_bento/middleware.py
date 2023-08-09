from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import Receive, Scope, Send
from scalene import scalene_profiler

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, path_mapping: dict):
        self.app = app
        self.path_mapping = path_mapping

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['Custom'] = 'Example'
        print("CustomHeaderMiddleware")
        return response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        print("aaa")
        print(self.path_mapping)

        #scalene_profiler.start()
        await self.app(scope, receive, send)
        #scalene_profiler.stop()

        print("bbb")
