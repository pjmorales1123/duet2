"""Local-only HTTP controls and a stream of actual MuJoCo camera frames."""
from __future__ import annotations

import argparse
import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from .engine import LabEngine
from .recording import EPISODE_ROOT

STATIC = Path(__file__).parent / "web"
engine = LabEngine()


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(engine.start)
    yield
    engine.close()


app = FastAPI(title="Talos dinner-table simulator", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method != "GET" and origin and urlparse(origin).netloc != request.headers.get("host"):
        return Response("Only same-origin controls are accepted.", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera: Literal["center", "overview", "overhead", "opposite", "left_wrist_cam", "right_wrist_cam"] | None = None
    running: bool | None = None
    shadows: bool | None = None
    preset: Literal["home", "gentle"] | None = None
    arm: Literal["left", "right"] | None = None
    targets_deg: list[float] | None = Field(default=None, min_length=6, max_length=6)
    step: bool = False

    @model_validator(mode="after")
    def require_arm(self):
        if self.targets_deg is not None and self.arm is None:
            raise ValueError("Choose an arm for joint targets.")
        return self


class Reset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2147483647)
    rack_count: int = Field(default=3, ge=2, le=4)
    practice: bool = False
    transfer_side: Literal["left", "right"] | None = None
    # Preserve older tube-experiment clients. The viewer explicitly sends its
    # selected scenario; the server's initial scene is dinner.
    scenario: Literal["dinner", "chemistry"] = "chemistry"
    dinner_preset: Literal["task", "reference"] = "task"
    bottle_start: Literal['upright', 'sideways', 'wide_left', 'wide_rectangle'] = 'upright'


class TaskCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["start", "cancel"]
    arm: Literal["auto", "left", "right"] = "auto"
    tube_id: str | None = Field(default=None, pattern=r"^[A-D][1-6]$")
    object_id: Literal["bottle","plate","mug","fork","spoon"] | None = None
    kind: Literal["lift_return", "transfer", "set_table", "dinner_place"] = "lift_return"
    destination_slot: str | None = Field(default=None, pattern=r"^[A-D][1-6]$")
    record: bool = False
    record_images: bool = True


class Rack(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=-0.30, le=0.30)
    y: float = Field(ge=-0.015, le=0.30)
    yaw_deg: float = Field(ge=-180, le=180)

class LanguageCommand(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=500)
    record:bool=False
    mode:Literal['programmed','learned_bottle','learned_bottle_legacy','learned_dinner','learned_dinner_visual','learned_dinner_wide']='programmed'


class Layout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    racks: list[Rack] = Field(min_length=2, max_length=4)


def send(operation, payload):
    try:
        return engine.submit(operation, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(507, str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    if engine.error:
        raise HTTPException(503, engine.error)
    return engine.snapshot()


@app.post("/api/control")
def control(request: Control):
    return send("control", request.model_dump(exclude_none=True))


@app.post("/api/reset")
def reset(request: Reset):
    return send("reset", request.model_dump())


@app.post("/api/racks")
def racks(request: Layout):
    return send("layout", request.model_dump())


@app.post("/api/task")
def task(request: TaskCommand):
    return send("task", request.model_dump())

@app.post('/api/command')
def command(request:LanguageCommand):
    return send('language',request.model_dump())

@app.get('/api/speech/status')
def speech_status():
    from .speech import status
    return status()

@app.post('/api/speech/token')
async def speech_token():
    from .speech import temporary_token
    try:return await asyncio.to_thread(temporary_token)
    except ValueError as exc:raise HTTPException(503,str(exc)) from None
    except RuntimeError as exc:raise HTTPException(502,str(exc)) from None


@app.get("/frame.jpg")
def frame():
    if engine.error:
        raise HTTPException(503, engine.error)
    _, content = engine.image()
    return Response(content, media_type="image/jpeg")


@app.get("/api/recordings/{episode_id}/manifest")
def recording_manifest(episode_id: str):
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}_[a-f0-9]{10}", episode_id):
        raise HTTPException(404, "Recording not found.")
    path = EPISODE_ROOT/episode_id/"manifest.json"
    if not path.is_file():
        raise HTTPException(404, "Recording manifest is not ready.")
    return FileResponse(path, media_type="application/json")


@app.get("/stream")
async def stream(request: Request):
    async def frames():
        previous = -1
        while not engine.error and not await request.is_disconnected():
            number, content = engine.image()
            if content and number != previous:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(content)).encode() + b"\r\n\r\n" + content + b"\r\n"
                previous = number
            await asyncio.sleep(0.015)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--learned',action='store_true',help='Preload the learned runtime before accepting browser requests (training environment).')
    parser.add_argument('--dinner-suite',type=Path,help='Explicit neural dinner suite JSON; requires --learned.')
    args = parser.parse_args()
    if args.dinner_suite:
        if not args.learned:parser.error('--dinner-suite requires --learned')
        if not args.dinner_suite.is_file():parser.error('Dinner suite file does not exist.')
        engine.dinner_suite=args.dinner_suite.resolve()
    if args.learned:
        import torch
        # Set the CPU worker budget before HTTP/physics threads start. The
        # learned controller also applies it on its owning physics thread.
        torch.set_num_threads(2)
        from .learned_task import DEFAULT_CHECKPOINT
        # Import PyTorch on the main thread before the engine starts. Leave
        # OpenVINO model compilation on its owning physics thread: precompiling
        # a throwaway model here caused large live-thread inference overhead.
        if not (DEFAULT_CHECKPOINT/'primitive.json').is_file():
            raise RuntimeError('Packaged learned bottle model is missing.')
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
