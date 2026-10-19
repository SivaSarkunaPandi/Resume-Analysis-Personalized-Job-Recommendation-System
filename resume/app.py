from flask import Flask, render_template, request
import os
import pdfplumber
import docx
import pickle
import re

app = Flask(__name__)

# ===============================
# 🔹 CONFIG
# ===============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# 🔹 TEXT EXTRACTION
# ===============================
def extract_text(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text

# ===============================
# 🔹 CLEAN TEXT
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ===============================
# 🔹 PREDICT FUNCTION
# ===============================
def predict_resume(file_path):
    text = extract_text(file_path)
    clean = clean_text(text)

    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    return prediction, top3

# ===============================
# 🔹 ROUTES
# ===============================

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Career Page
@app.route("/career")
def career():
    return render_template("career.html")

# Predict Page (Upload Form)
@app.route("/predict")
def predict():
    return render_template("predict.html")

# Result Page
@app.route("/result", methods=["POST"])
def result():
    if "resume" not in request.files:
        return "No file uploaded"

    file = request.files["resume"]

    if file.filename == "":
        return "No file selected"

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Predict
    job, top3 = predict_resume(file_path)

    return render_template(
        "result.html",
        job=job,
        top3=top3
    )

# ===============================
# 🔹 RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)