"""FastAPI 接口层：校验输入 → 取得已加载的模型 → 推理 → 返回 JSON。"""
from contextlib import asynccontextmanager
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mutation_utils import (
    create_multi_mutant, mutation_set_label, normalize_sequence, parse_mutation_set,
)

BASE_DIR = Path(__file__).resolve().parent
MAX_MUTATIONS = 10
logger = logging.getLogger("ddg")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class PredictionRequest(BaseModel):
    # 保留旧客户端的 pos/wt/mt 输入；新接口统一使用 mutations。
    model_config = ConfigDict(extra="ignore")
    sequence: str = Field(min_length=1, description="Wild-type protein sequence")
    mutations: str = Field(min_length=1, description="For example F88Y_L91A")

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_fields(cls, data):
        if not isinstance(data, dict):
            return data  # 交给 Pydantic 拒绝数组、数字等非对象 JSON。
        data = dict(data)
        mutations = data.get("mutations")
        # 与原接口相同：非空 mutations 优先，否则尝试拼接完整的旧字段。
        if mutations is None or (isinstance(mutations, str) and not mutations.strip()):
            pos, wt, mt = data.get("pos"), data.get("wt", ""), data.get("mt", "")
            if pos not in (None, "") or wt or mt:
                if (isinstance(pos, bool) or not isinstance(pos, (int, str)) or
                        pos == "" or not isinstance(wt, str) or not wt.strip() or
                        not isinstance(mt, str) or not mt.strip()):
                    raise ValueError("Provide a complete mutation such as E94H.")
                data["mutations"] = f"{wt.strip()}{pos}{mt.strip()}"
        return data

    @field_validator("sequence")
    @classmethod
    def clean_sequence(cls, value):
        return normalize_sequence(value)

    @field_validator("mutations")
    @classmethod
    def clean_mutations(cls, value):
        return mutation_set_label(parse_mutation_set(value, max_mutations=MAX_MUTATIONS))

    @model_validator(mode="after")
    def check_positions(self):
        # 格式正确还不够：位置必须在序列范围内，原始氨基酸必须与序列一致。
        create_multi_mutant(self.sequence, parse_mutation_set(self.mutations))
        return self


class PredictionResponse(BaseModel):
    ddg: float = Field(allow_inf_nan=False)
    length: int
    mutation: str
    mutation_count: int


class ErrorResponse(BaseModel):
    error: str


def load_model_service():
    # 延迟导入，接口测试时可传入模拟模型，不必加载大型权重。
    from inference import ModelService
    return ModelService()


def get_service(request: Request):
    return request.app.state.service


def create_app(service_factory=load_model_service):
    # 工厂参数主要供测试替换模型；正常启动会加载真实的 ESM 和 model.pt。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading prediction models...")
        app.state.service = service_factory()
        app.state.inference_lock = Lock()
        logger.info("Models ready; device=%s", app.state.service.device)
        try:
            yield  # 启动准备完成，此后开始接收请求。
        finally:
            app.state.service.close()
            app.state.service = None

    app = FastAPI(title="DDG-Predictor", version="2.0.0", lifespan=lifespan)
    app.mount("/ddg-predictor/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # 保留原前端识别的 {error: ...} 格式及 400 状态码；不原样返回整个输入序列。
        message = exc.errors()[0]["msg"]
        return JSONResponse(status_code=400, content={"error": message.removeprefix("Value error, ")})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/ddg-predictor/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/healthz")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.service is not None,
                "device": str(request.app.state.service.device)}

    errors = {status: {"model": ErrorResponse} for status in (400, 500, 503)}

    @app.post("/predict", response_model=PredictionResponse, responses=errors)
    @app.post("/ddg-predictor/predict", response_model=PredictionResponse, responses=errors)
    def predict(body: PredictionRequest, request: Request, service: Annotated[object, Depends(get_service)]):
        # 普通 def 在 FastAPI 的线程池运行；不会直接阻塞异步事件循环。
        # 单进程一次只推理一个请求。忙时明确返回 503，避免多个请求同时挤占模型内存。
        lock = request.app.state.inference_lock
        if not lock.acquire(blocking=False):
            return JSONResponse(status_code=503, content={"error": "Model is busy. Please try again shortly."},
                                headers={"Retry-After": "2"})
        try:
            value, mutations = service.predict(body.sequence, body.mutations)
            if not math.isfinite(value):
                raise RuntimeError("Model returned a non-finite value")
            return PredictionResponse(ddg=round(value, 4), length=len(body.sequence),
                                      mutation=mutations, mutation_count=len(parse_mutation_set(mutations)))
        except Exception:
            logger.exception("Prediction failed")
            return JSONResponse(status_code=500, content={"error": "Prediction failed. Please try again later."})
        finally:
            lock.release()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    # 本地学习使用一个进程；多个 worker 会各自加载一份大模型。
    uvicorn.run(app, host="127.0.0.1", port=8088)
