from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.25
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

model = YOLO(str(MODEL_PATH))

app = FastAPI(title="Brain Tumor Detection")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def get_class_name(class_id: int) -> str:
    """Read a class name from the names supplied by the trained YOLO model."""
    if isinstance(model.names, dict):
        return str(model.names.get(class_id, f"Class {class_id}"))
    if 0 <= class_id < len(model.names):
        return str(model.names[class_id])
    return f"Class {class_id}"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the single-page upload interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Validate an image, run YOLO, draw detections, and return JSON."""
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Please upload a JPG, JPEG, or PNG image.")

    if file.content_type and file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="The uploaded file must be a JPG, JPEG, or PNG image.")

    image_bytes = await file.read(MAX_FILE_SIZE + 1)
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="The image must be smaller than 10 MB.")

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        predictions = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = predictions[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))
            class_name = get_class_name(class_id)
            label = f"{class_name} {confidence:.1%}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (190, 90, 255), 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_top = max(y1, text_height + baseline + 6)
            cv2.rectangle(
                image,
                (x1, label_top - text_height - baseline - 6),
                (x1 + text_width + 8, label_top),
                (190, 90, 255),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 4, label_top - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (25, 12, 35),
                2,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 6),
                    "confidence_percent": round(confidence * 100, 2),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

        result_filename = f"{uuid4().hex}.jpg"
        result_path = RESULTS_DIR / result_filename
        if not cv2.imwrite(str(result_path), image):
            raise RuntimeError("OpenCV could not save the result image.")

    except HTTPException:
        raise
    except Exception as error:
        print(f"Prediction failed: {error}")
        raise HTTPException(status_code=500, detail="The image could not be analyzed. Please try again.") from error

    return {
        "result_url": f"/results/{result_filename}",
        "detections": detections,
        "detection_count": len(detections),
    }
