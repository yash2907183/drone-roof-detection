from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="Drone Roof Detection API")
model = YOLO('best.pt')

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body>
    <h1>🚁 Drone Roof Detection</h1>
    <form action="/predict-image" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Detect</button>
    </form>
    </body></html>
    """

@app.post("/predict-image")
async def predict_with_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(np.array(image), conf=0.5)
    
    annotated_img = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": f"{float(box.conf):.2f}"
            })
    
    html = f"""
    <html><body>
    <h1>Detection Results</h1>
    <img src="data:image/jpeg;base64,{img_base64}" style="max-width:600px;">
    <h3>Found {len(detections)} objects:</h3>
    {"".join([f"<p>• {d['class']}: {d['confidence']}</p>" for d in detections])}
    <a href="/">← Back</a>
    </body></html>
    """
    return HTMLResponse(content=html)
