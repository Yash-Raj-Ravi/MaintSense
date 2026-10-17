"""FastAPI application exposing the existing MaintSense inference pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from numbers import Real
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline
from starlette.middleware.cors import CORSMiddleware

from maintsense.baseline import FEATURES
from maintsense.inference import (
    DECISION_THRESHOLD,
    load_pipeline_artifact,
    predict_record,
)
from maintsense.model_artifact import ARTIFACT_PATH

logger = logging.getLogger(__name__)

SYNTHETIC_SCOPE = (
    "Machine-failure classification on the synthetic AI4I 2020 dataset; "
    "not a validated real-world early-warning system."
)


class PredictionRequest(BaseModel):
    """The six leakage-safe inputs accepted by the persisted pipeline."""

    model_config = ConfigDict(extra="forbid")

    Type: Literal["H", "L", "M"]
    air_temperature: Annotated[float, Field(alias="Air temperature [K]", allow_inf_nan=False)]
    process_temperature: Annotated[
        float, Field(alias="Process temperature [K]", allow_inf_nan=False)
    ]
    rotational_speed: Annotated[
        float, Field(alias="Rotational speed [rpm]", allow_inf_nan=False)
    ]
    torque: Annotated[float, Field(alias="Torque [Nm]", allow_inf_nan=False)]
    tool_wear: Annotated[float, Field(alias="Tool wear [min]", allow_inf_nan=False)]

    @field_validator(
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        mode="before",
    )
    @classmethod
    def require_json_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("must be a JSON number")
        return value

    def to_inference_record(self) -> dict[str, object]:
        """Map validated API fields to the existing inference feature names."""
        return {
            "Type": self.Type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
        }


class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_class: Literal[0, 1]
    threshold: float


def create_app(
    artifact_path: Path = ARTIFACT_PATH,
    *,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build an app that loads its artifact once during each lifespan."""

    if cors_origins is None:
        cors_origins = [
            origin.strip()
            for origin in os.getenv("MAINTSENSE_CORS_ORIGINS", "").split(",")
            if origin.strip()
        ]
    if "*" in cors_origins:
        raise ValueError(
            "CORS origins must be explicitly listed; wildcard origins are not allowed."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        app.state.model_error = None
        try:
            app.state.pipeline = load_pipeline_artifact(artifact_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            # Keep health and model-info available so an operator can diagnose
            # a missing/corrupt local artifact; prediction responds with 503.
            app.state.model_error = str(exc)
        yield
        app.state.pipeline = None

    app = FastAPI(
        title="MaintSense Prediction API",
        description=SYNTHETIC_SCOPE,
        version="0.1.0",
        lifespan=lifespan,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo invalid request values: non-finite floats cannot be
        # encoded as standards-compliant JSON in an error response.
        details = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health")
    def health(request: Request) -> dict[str, str | bool]:
        available = isinstance(request.app.state.pipeline, Pipeline)
        return {
            "status": "ok" if available else "degraded",
            "model_available": available,
        }

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str | bool]:
        """Return success only when the inference artifact is loaded."""
        if not isinstance(request.app.state.pipeline, Pipeline):
            raise HTTPException(
                status_code=503,
                detail="Prediction service is not ready.",
            )
        return {"status": "ready", "model_available": True}

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, object]:
        available = isinstance(request.app.state.pipeline, Pipeline)
        return {
            "model_type": "RandomForestClassifier",
            "threshold": DECISION_THRESHOLD,
            "required_features": FEATURES,
            "scope": SYNTHETIC_SCOPE,
            "model_available": available,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> dict[str, float | int]:
        pipeline = request.app.state.pipeline
        if not isinstance(pipeline, Pipeline):
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable; check the local model artifact.",
            )
        try:
            result = predict_record(payload.to_inference_record(), pipeline)
        except Exception as exc:
            # Estimator/runtime failures are server errors; keep implementation
            # details and local paths out of the public response.
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed due to an internal service error.",
            ) from exc
        return {**result, "threshold": DECISION_THRESHOLD}

    return app


app = create_app()
