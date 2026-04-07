# app.py

# FastAPI application for the MedAgent clinical triage environment.


try:
    
    # Import OpenEnv helper to automatically create a FastAPI server for our environment
    from openenv.core.env_server.http_server import create_app

# Error Handling
except Exception as e:
    
    raise ImportError(
        "openenv is required for the web interface. Install project dependencies first."
    ) from e

try:
    
    # Import components and schema definitions
    from ..models import MedAgentAction, MedAgentObservation
    from .environment import MedAgentEnvironment

# Error Handling
except ImportError:
    from models import MedAgentAction, MedAgentObservation
    from server.environment import MedAgentEnvironment

# Creation of the FastAPI app
app = create_app(
    MedAgentEnvironment,
    MedAgentAction,
    MedAgentObservation,
    env_name="medagent",
    max_concurrent_envs=4,
)

from fastapi.responses import RedirectResponse

@app.get("/")

def root ():
    return RedirectResponse(url="/docs")

# Run FastAPI server using Uvicorn
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    
    # Import Uvicorn for running the ASGI server
    import uvicorn

    uvicorn.run(app, host=host, port=port)

# Import OpenEnv helper to automatically create FastAPI server for the environment
if __name__ == "__main__":
    
    # Import argparse for command-line argument parsing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)