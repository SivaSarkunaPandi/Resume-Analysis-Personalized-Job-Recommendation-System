# Resume Analysis Using ML for Personalized Job Recommendations

An AI and Machine Learning based web application that analyzes a candidate's resume and recommends the most suitable job roles. Candidates upload a resume (PDF or DOCX), and the system extracts key details using Natural Language Processing, compares the profile with job descriptions, and returns a ranked list of job roles.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Overview

Traditional resume screening is manual, slow, and often relies on simple keyword matching, which overlooks qualified candidates. This project automates the process:

1. The candidate uploads a resume (PDF/DOCX).
2. NLP extracts skills, education, experience, certifications, and projects.
3. TF-IDF converts the text into numerical vectors.
4. Cosine Similarity scores the resume against predefined job descriptions.
5. A Random Forest model trained on historical placement data refines the results.
6. The system displays ranked, personalized job recommendations and a skill gap analysis.

## Features

- Resume upload in **PDF and DOCX** formats with file type and size validation
- Automatic extraction of skills, education, experience, certifications, and projects
- **TF-IDF + Cosine Similarity** based job matching
- **Random Forest** model for improved recommendation accuracy
- Ranked, personalized job role recommendations
- Skill gap analysis for career improvement
- Supports both freshers and experienced professionals
- Scalable, modular design

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | HTML, CSS, Bootstrap, JavaScript |
| Backend | Python, Flask |
| NLP | NLTK / spaCy |
| Machine Learning | Scikit-learn (TF-IDF, Cosine Similarity, Random Forest) |
| File Handling | PDF and DOCX text extraction libraries |
| Tools | Python IDLE |

## System Architecture

```
User → Resume Upload → Resume Parsing → Data Preprocessing
     → Feature Extraction (NLP) → TF-IDF Vectorization
     → Cosine Similarity → Random Forest Prediction → Job Recommendation
```

## Modules

**1. Data Ingestion & Extraction**
- *Data Collection* – gathers resumes, job descriptions, and historical placement data
- *Data Preprocessing* – cleans text, removes noise and duplicates, handles missing values
- *Resume Upload* – accepts and validates PDF/DOCX files
- *Resume Parsing & Text Extraction* – converts files into machine-readable text

**2. Feature Engineering**
- *NLP-Based Feature Extraction* – identifies skills, education, experience, certifications, and projects
- *Feature Vectorization (TF-IDF)* – converts text into numerical vectors

**3. Matching & Recommendation**
- *Job Similarity Matching* – Cosine Similarity scores between resumes and job descriptions
- *Machine Learning Prediction* – Random Forest refines similarity-based results
- *Job Recommendation* – generates the final ranked list of job roles

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP resources (if using NLTK)
python -m nltk.downloader punkt stopwords

# 5. Run the application
python app.py
```

The app runs at `http://127.0.0.1:5000/`.

## Usage

1. Open the web app in your browser.
2. Upload your resume in PDF or DOCX format.
3. Wait for the system to analyze the resume.
4. View your recommended job roles and skill gap analysis.

## Project Structure

> Adjust this section to match your actual folders and files.

```
├── app.py                  # Flask application entry point
├── requirements.txt        # Python dependencies
├── model/                  # Trained Random Forest model and vectorizer
├── data/                   # Job descriptions and training dataset
├── uploads/                # Uploaded resumes
├── static/                 # CSS, JavaScript, images
├── templates/              # HTML templates
└── README.md
```

## System Requirements

**Hardware**
- Processor: Intel Core i7 (5th Gen) or equivalent
- RAM: 12 GB
- Storage: 500 GB
- GPU: NVIDIA GeForce GTX 1650 (optional)

**Software**
- OS: Windows 10
- Python 3.x
- Flask, Scikit-learn, NLTK/spaCy

## Future Enhancements

- Transformer-based deep learning models for better parsing and matching
- Multi-language support
- Integration with job portals and professional networking platforms
- AI chatbot for career guidance
- Course and certification suggestions for skill gaps
- Hybrid recommendation (content-based + collaborative filtering)
- Mobile application and cloud deployment
- Continuous learning from new data
- Support for internships and freelance roles

## Author

**Siva**
MCA Graduate
GitHub: [@SivaSarkunaPandi](https://github.com/SivaSarkunaPandi)
