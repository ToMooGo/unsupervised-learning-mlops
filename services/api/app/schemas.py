"""Request / response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# fmt: off
EXAMPLE_ZERO = [  # first image of sklearn's digits dataset (a "0")
    0, 0, 5, 13, 9, 1, 0, 0,
    0, 0, 13, 15, 10, 15, 5, 0,
    0, 3, 15, 2, 0, 11, 8, 0,
    0, 4, 12, 0, 0, 8, 8, 0,
    0, 5, 8, 0, 0, 9, 8, 0,
    0, 4, 11, 0, 1, 12, 7, 0,
    0, 2, 14, 5, 10, 12, 0, 0,
    0, 0, 6, 13, 10, 0, 0, 0,
]
# fmt: on


Pixel = Annotated[float, Field(ge=0, le=16, allow_inf_nan=False)]
SAFE_TEXT = r"^[A-Za-z0-9 _.:@-]+$"  # labels shown in the UI: no markup characters


class PredictRequest(BaseModel):
    pixels: list[Pixel] = Field(
        ...,
        min_length=64,
        max_length=64,
        description="64 pixel values (8x8 image, row-major), each a finite number in [0, 16]",
    )
    source: str = Field(
        "api",
        min_length=1,
        max_length=32,
        pattern=SAFE_TEXT,
        description="where the input came from (canvas, sample-noise, ...)",
    )

    model_config = {"json_schema_extra": {"examples": [{"pixels": EXAMPLE_ZERO, "source": "docs"}]}}


class ClassifierOutput(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


class AnomalyOutput(BaseModel):
    is_anomaly: bool
    log_density: float
    threshold: float
    density_percentile: float
    message: str


class PredictResponse(BaseModel):
    prediction_id: int | None
    supervised: ClassifierOutput
    semi_supervised: ClassifierOutput
    anomaly: AnomalyOutput
    model_versions: dict[str, str]
    latency_ms: float


class LabelItem(BaseModel):
    train_index: int = Field(..., ge=0)
    label: int = Field(..., ge=0, le=9)


class LabelsRequest(BaseModel):
    labels: list[LabelItem] = Field(..., min_length=1)
    labeler: str = Field("anonymous", min_length=1, max_length=64, pattern=SAFE_TEXT)
