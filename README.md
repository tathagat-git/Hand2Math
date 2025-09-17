# Hand2Math
Hand2Math – Real-Time Handwritten Equation Solver Built a FastAPI web app with a CNN model (TensorFlow/Keras) that recognizes handwritten digits &amp; math symbols from canvas drawings, evaluates equations instantly, and renders results back in a natural handwriting style.

# ✍️ Hand2Math - Real-Time Handwritten Equation Solver

Hand2Math is a FastAPI-powered web application that lets you **draw math equations by hand** ✍️ on a canvas, 
recognizes digits & symbols using a CNN model (TensorFlow/Keras), **solves the equation instantly**, 
and renders the result back in **handwriting style**.

## 🚀 Features
- Handwritten digit & symbol recognition (0-9, +, -, ×, ÷, =)
- Real-time evaluation of equations (e.g., `2+3=5`)
- Web interface with HTML5 Canvas for drawing
- Result rendered back in handwriting style

## 🛠️ Tech Stack
- **Backend**: FastAPI, TensorFlow/Keras
- **Frontend**: HTML5 Canvas, JavaScript
- **Others**: OpenCV, Pillow, NumPy

## 📦 Installation
```bash
git clone https://github.com/YOUR_USERNAME/Hand2Math.git
cd Hand2Math
pip install -r requirements.txt
uvicorn main:app --reload

