"""
FastAPI server entry-point — one line of meaningful code.
create_fastapi_app() auto-generates: /ws /reset /step /state /health /docs /web
"""
from openenv.core.env_server import create_fastapi_app
from environment import SupportTriageEnvironment
from models import SupportAction, SupportObservation, SupportState

app = create_fastapi_app(
    SupportTriageEnvironment,
    action_cls=SupportAction,
    observation_cls=SupportObservation
)
