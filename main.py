from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import base64, io, re

# --------------------------
# Load Model & Class Map
# --------------------------
model = tf.keras.models.load_model("hand2math_cnn.keras")

class_map = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "add": 10, "sub": 11, "mul": 12, "div": 13,
    "equal": 14, "dec": 15
}
inv_class_map = {v: k for k, v in class_map.items()}

# --------------------------
# Preprocess One Symbol
# --------------------------
def preprocess_symbol(roi):
    roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded[y_offset:y_offset+20, x_offset:x_offset+20] = roi
    padded = padded.astype("float32") / 255.0
    return np.expand_dims(padded, -1)

# --------------------------
# Predict Equation
# --------------------------
def predict_equation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbol_images, boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:  # skip noise
            continue
        roi = thresh[y:y+h, x:x+w]
        symbol_images.append(preprocess_symbol(roi))
        boxes.append((x, y, w, h))

    symbols_sorted = [s for _, s in sorted(zip(boxes, symbol_images), key=lambda b: b[0][0])]

    expression = ""
    for s in symbols_sorted:
        pred = model.predict(s.reshape(1, 28, 28, 1), verbose=0)
        label = inv_class_map[np.argmax(pred)]
        if label == "add": expression += "+"
        elif label == "sub": expression += "-"
        elif label == "mul": expression += "*"
        elif label == "div": expression += "/"
        elif label == "equal": expression += "="
        elif label == "dec": expression += "."
        else: expression += label

    # --------------------------
    # Post-processing fixes
    # --------------------------
    # Replace double minus with equal
    expression = expression.replace("--", "=")

    # Collapse multiple operators like +++ → +
    expression = re.sub(r'([+\-*/=])\1+', r'\1', expression)

    return expression

# --------------------------
# Render Expression in Handwriting Font
# --------------------------
def render_handwritten_font(expr, font_path="handwriting.ttf"):
    try:
        font = ImageFont.truetype(font_path, 48)
    except:
        font = ImageFont.load_default()

    img = Image.new("L", (400, 100), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), expr, font=font, fill=0)
    return img

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    image_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(image)

    # Predict
    expression = predict_equation(img)

    # Evaluate if "=" present
    if "=" in expression:
        try:
            lhs, rhs = expression.split("=")[0], expression.split("=")[1]
            result = eval(lhs)
            expression = f"{lhs}={result}"
        except Exception:
            pass

    # Render as handwriting
    handwriting_img = render_handwritten_font(expression)
    buf = io.BytesIO()
    handwriting_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_b64}", "expression": expression}

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
