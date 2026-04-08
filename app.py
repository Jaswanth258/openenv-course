"""
FastAPI server entry-point — one line of meaningful code.
create_fastapi_app() auto-generates: /ws /reset /step /state /health /docs /web
"""
from openenv.core.env_server import create_fastapi_app

from environment import SupportTriageEnvironment

app = create_fastapi_app(SupportTriageEnvironment)
