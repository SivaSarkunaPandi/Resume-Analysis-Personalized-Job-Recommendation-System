import pdfplumber
import docx
import pickle
import re

# ===============================
# 🔹 LOAD MODEL & VECTORIZER
# ===============================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# 🔹 EXTRACT TEXT FROM RESUME
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
# 🔹 MAIN PREDICTION FUNCTION
# ===============================
def predict_resume(file_path):

    # 1. Extract resume text
    resume_text = extract_text(file_path)

    # 2. Clean text
    clean = clean_text(resume_text)

    # 3. Convert to TF-IDF
    vec = vectorizer.transform([clean])

    # 4. Predict job
    prediction = model.predict(vec)[0]

    # 5. Get probabilities
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    # 6. Top 3 jobs
    top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    return prediction, top3


# ===============================
# 🔹 RUN PROGRAM
# ===============================
if __name__ == "__main__":

    file_path = input("Enter resume file path (.pdf/.docx): ")

    job, top3 = predict_resume(file_path)

    print("\n==============================")
    print("🎯 Recommended Job Role")
    print("==============================")
    print(job)

    print("\n🔥 Top 3 Career Recommendations:")
    for i, (j, score) in enumerate(top3, 1):
        print(f"{i}. {j} ({round(score*100,2)}%)")