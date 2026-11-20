"""FastAPI app that serves the Iris Decision Tree Classifier."""
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from predict import MODEL_INFO_PATH, MODEL_PATH, load_model, load_model_info, predict_data

logger = logging.getLogger("uvicorn.error")

MAX_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Startup: load the model once when the server starts (lifespan event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and its metadata once, before the server takes requests.

    The yielded dict is copied into `request.state` for every request. If the
    model files are missing, the server still starts, but the model endpoints
    return 503 until the model is trained and the server is restarted.
    """
    model, model_info, load_error = None, None, None
    try:
        model = load_model(MODEL_PATH)
        model_info = load_model_info(MODEL_INFO_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    except FileNotFoundError as e:
        model, model_info, load_error = None, None, str(e)
        logger.warning("Model not loaded: %s", load_error)

    yield {"model": model, "model_info": model_info, "load_error": load_error}


app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Serves a Decision Tree Classifier trained on the Iris dataset. "
        "Returns the predicted species with class probabilities, for one "
        "flower or a batch."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class IrisData(BaseModel):
    """Measurements of one flower, in centimetres. Each must be > 0 and <= 10."""

    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width (cm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.3}
            ]
        }
    }

    def to_features(self) -> List[float]:
        """Return the features in the order the model was trained on."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class IrisBatchRequest(BaseModel):
    samples: List[IrisData] = Field(
        ..., description=f"Between 1 and {MAX_BATCH_SIZE} flowers to classify"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
                        {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.3},
                    ]
                }
            ]
        }
    }


class IrisResponse(BaseModel):
    class_id: int = Field(..., description="0=setosa, 1=versicolor, 2=virginica")
    class_name: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Probability of every class")


class IrisBatchResponse(BaseModel):
    count: int
    predictions: List[IrisResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str


# Extra error responses, so they show up in /docs
MODEL_NOT_LOADED = {503: {"model": ErrorResponse, "description": "Model is not loaded (run train.py)"}}
BAD_BATCH = {400: {"model": ErrorResponse, "description": "Batch is empty or too large"}}


class ModelInfoResponse(BaseModel):
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    class_names: List[str]
    test_accuracy: float
    train_samples: int
    test_samples: int
    trained_at: str
    sklearn_version: str


# ---------------------------------------------------------------------------
# Dependencies and helpers
# ---------------------------------------------------------------------------
def get_model(request: Request):
    """Return the loaded model, or raise 503 if it is not loaded."""
    if request.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.state.load_error or "Model is not loaded.",
        )
    return request.state.model


def run_prediction(model, samples: List[IrisData]) -> List[IrisResponse]:
    """Run the model on the samples, turning unexpected errors into a 500."""
    try:
        results = predict_data(model, [s.to_features() for s in samples])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )
    return [IrisResponse(**r) for r in results]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_ping(request: Request):
    """Health check. Also reports whether the model is loaded."""
    return HealthResponse(status="healthy", model_loaded=request.state.model is not None)


@app.get("/model/info", response_model=ModelInfoResponse, responses=MODEL_NOT_LOADED)
async def model_info(request: Request, model=Depends(get_model)):
    """Metadata about the loaded model: type, settings, accuracy, training time."""
    return request.state.model_info


@app.post("/predict", response_model=IrisResponse, responses=MODEL_NOT_LOADED)
async def predict_iris(iris_features: IrisData, model=Depends(get_model)):
    """Predict the species of one flower, with class probabilities."""
    return run_prediction(model, [iris_features])[0]


@app.post(
    "/predict/batch",
    response_model=IrisBatchResponse,
    responses={**BAD_BATCH, **MODEL_NOT_LOADED},
    description=f"Predict the species of 1 to {MAX_BATCH_SIZE} flowers in one request.",
)
async def predict_iris_batch(batch: IrisBatchRequest, model=Depends(get_model)):
    n = len(batch.samples)
    if n == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`samples` must contain at least one flower.",
        )
    if n > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch has {n} samples; the maximum is {MAX_BATCH_SIZE}.",
        )
    predictions = run_prediction(model, batch.samples)
    return IrisBatchResponse(count=n, predictions=predictions)
