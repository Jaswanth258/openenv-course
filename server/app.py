"""
FastAPI server entry-point for OpenEnv validator.
Moved into `server/` to comply with OpenEnv hackathon structure.
"""
import sys
import os

# Ensure the root directory is in the PYTHONPATH so we can import models and environment securely
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server import create_fastapi_app
from environment import SupportTriageEnvironment
from models import SupportAction, SupportObservation

app = create_fastapi_app(
    SupportTriageEnvironment,
    action_cls=SupportAction,
    observation_cls=SupportObservation
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
