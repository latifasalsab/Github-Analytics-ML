# GitHub Analytics ML

Model Machine Learning untuk platform analisis kinerja repository GitHub dan evaluasi kolaborasi tim.

Bagian dari Tugas Akhir:
**"Rancang Bangun Platform Analisis Kinerja Repository GitHub dan Evaluasi Kolaborasi Tim Berbasis Machine Learning"**

---

## Model Machine Learning

| Model | Algoritma | Tipe | Output |
|-------|-----------|------|--------|
| Model 1 — Productivity State | Random Forest | Klasifikasi | Active / Moderate / Inactive |
| Model 2 — Health Score | Gradient Boosting | Regresi | Score 0–100 + Grade A–E |
| Model 3 — Member Status | K-Means + Random Forest | Klasifikasi | Active / Passive / Inactive |

---

## Requirements

- Python 3.12+
- Virtual environment (disarankan)

---

## Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/github-analytics-ml.git
cd github-analytics-ml
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r api/requirements.txt
pip install jupyter ipykernel pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn python-dotenv requests ijson
```

### 4. Buat File .env

Buat file `.env` di root folder:

```
GITHUB_TOKEN=your_github_personal_access_token
```

> Personal Access Token bisa dibuat di: GitHub → Settings → Developer Settings → Personal Access Tokens

### 5. Jalankan FastAPI

```bash
cd api
uvicorn main:app --reload --port 8000
```

Buka browser ke `http://127.0.0.1:8000/docs` untuk melihat dokumentasi endpoint.

---
