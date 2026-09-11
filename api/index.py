from app.main import app as fastapi_app


class StripApiPrefix:
    """
    Vercel routes /api/:path* to this function but forwards the original
    request path (e.g. "/api/health") unchanged. app.main's routes are
    unprefixed ("/health") so they can also be served directly by
    uvicorn/Docker. Strip the "/api" prefix here, at the Vercel boundary
    only, and tell Starlette about it via root_path so docs/OpenAPI still
    generate correct URLs.
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path == "/api" or path.startswith("/api/"):
                scope = dict(scope)
                scope["path"] = path[len("/api"):] or "/"
                scope["root_path"] = "/api"
        await self.asgi_app(scope, receive, send)


app = StripApiPrefix(fastapi_app)
__all__ = ["app"]
