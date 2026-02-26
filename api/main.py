from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import math
import os

app = FastAPI(title="GitHub Analytics ML Service")

# CORS supaya Next.js bisa akses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load semua model saat startup ──────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "models")

model1 = joblib.load(os.path.join(BASE, "model1_productivity.pkl"))
model2 = joblib.load(os.path.join(BASE, "model2_healthscore.pkl"))
model3 = joblib.load(os.path.join(BASE, "model3_memberstatus.pkl"))

with open(os.path.join(BASE, "model2_grade_thresholds.json")) as f:
    grade_thresholds = json.load(f)

print("✅ Semua model berhasil dimuat!")

# ── Helper functions ───────────────────────────────────────────
def get_grade(score):
    if score >= grade_thresholds["A"]:   return "A", "Excellent"
    elif score >= grade_thresholds["B"]: return "B", "Good"
    elif score >= grade_thresholds["C"]: return "C", "Fair"
    elif score >= grade_thresholds["D"]: return "D", "Poor"
    else:                                return "E", "Critical"

def productivity_recommendation(state, data):
    if state == "Active":
        return "Repository aktif dan konsisten, pertahankan ritme ini!"
    elif state == "Moderate":
        if data["commit_trend"] < 0:
            return "Aktivitas mulai menurun, coba jadwalkan sesi coding rutin"
        elif data["activity_consistency"] > 3.0:
            return "Commit tidak merata, coba bagi pekerjaan lebih konsisten tiap minggu"
        else:
            return "Tingkatkan frekuensi commit untuk mencapai status Active"
    else:
        return "Repository hampir tidak ada aktivitas, pertimbangkan untuk dilanjutkan atau diarsipkan"

def health_recommendations(breakdown):
    recs = []
    if breakdown["issueManagement"] < 12:
        recs.append("Tutup atau triage issue yang sudah terbuka lebih dari 30 hari")
    if breakdown["commitActivity"] < 15:
        recs.append("Tingkatkan frekuensi commit dan konsistensi aktivitas")
    if breakdown["dokumentasi"] < 10:
        recs.append("Lengkapi dokumentasi: README, LICENSE, dan deskripsi repo")
    if breakdown["konsistensi"] < 12:
        recs.append("Commit tidak merata, coba bagi pekerjaan lebih teratur")
    if breakdown["recency"] < 5:
        recs.append("Repo tidak aktif cukup lama, segera lanjutkan pengembangannya")
    if not recs:
        recs.append("Repo dalam kondisi baik, pertahankan kualitas kontribusi!")
    return recs

def member_recommendation(status, data):
    if status == "Active":
        return "Kontribusimu konsisten dan signifikan, pertahankan pola kerja ini!"
    elif status == "Passive":
        if data["activity_consistency"] > 3.0:
            return "Commit tidak merata, coba bagi pekerjaan lebih konsisten tiap minggu"
        else:
            return "Frekuensi commit masih rendah, coba tingkatkan keterlibatan di repo"
    else:
        return "Belum ada aktivitas signifikan, segera mulai berkontribusi!"

# ── Input schemas ──────────────────────────────────────────────
class ProductivityInput(BaseModel):
    commit_frequency:     float
    activity_consistency: float
    commit_trend:         float
    active_days_ratio:    float

class HealthInput(BaseModel):
    commit_frequency:     float
    activity_consistency: float
    commit_trend:         float
    active_days_ratio:    float
    velocity_stability:   float
    has_description:      int
    has_license:          int
    open_issues_count:    int
    stars:                int
    forks_count:          int
    commit_count_total:   int

class MemberInput(BaseModel):
    commit_velocity:      float
    contribution_share:   float
    activity_consistency: float
    active_weeks_ratio:   float

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "GitHub Analytics ML Service is running!"}

@app.post("/predict/productivity")
def predict_productivity(data: ProductivityInput):
    df = pd.DataFrame([data.model_dump()])
    features = ["commit_frequency", "activity_consistency",
                "commit_trend", "active_days_ratio"]

    state = model1.predict(df[features])[0]
    rec   = productivity_recommendation(state, data.model_dump())

    return {
        "productivityState":    state,
        "commitFrequency":      data.commit_frequency,
        "activityConsistency":  data.activity_consistency,
        "commitTrend":          data.commit_trend,
        "activeDaysRatio":      data.active_days_ratio,
        "recommendation":       rec
    }

@app.post("/predict/health")
def predict_health(data: HealthInput):
    d  = data.model_dump()
    df = pd.DataFrame([d])
    features = ["commit_frequency", "activity_consistency", "commit_trend",
                "active_days_ratio", "velocity_stability", "has_description",
                "has_license", "open_issues_count", "stars", "forks_count",
                "commit_count_total"]

    score        = float(model2.predict(df[features])[0])
    grade, label = get_grade(score)

    # Hitung breakdown
    import math
    breakdown = {
        "dokumentasi":      round(8 * d["has_license"] + 4 * d["has_license"] + 3 * d["has_description"], 1),
        "issueManagement":  round(min(max(0, 20 - (math.log1p(d["open_issues_count"] / (d["stars"] + 1) * 100) * 5)), 20), 1),
        "commitActivity":   round(min(d["commit_frequency"] / 3, 1.0) * 15 + d["active_days_ratio"] * 10, 1),
        "konsistensi":      round((1 - min(d["velocity_stability"] / 4, 1.0)) * 20, 1),
        "popularitas":      round(min(d["stars"] / 100000, 1.0) * 10, 1),
        "recency":          round(min(max(d["commit_trend"] + 0.5, 0) / 1.0, 1.0) * 10, 1),
    }

    recs = health_recommendations(breakdown)

    return {
        "healthScore":      round(score, 2),
        "grade":            grade,
        "label":            label,
        "breakdown":        breakdown,
        "recommendations":  recs
    }

@app.post("/predict/member")
def predict_member(data: MemberInput):
    df      = pd.DataFrame([data.model_dump()])
    features = ["commit_velocity", "contribution_share",
                "activity_consistency", "active_weeks_ratio"]

    status = model3.predict(df[features])[0]
    rec    = member_recommendation(status, data.model_dump())

    return {
        "memberStatus":         status,
        "commitVelocity":       data.commit_velocity,
        "contributionShare":    data.contribution_share,
        "activityConsistency":  data.activity_consistency,
        "activeWeeksRatio":     data.active_weeks_ratio,
        "recommendation":       rec
    }

@app.post("/predict/all")
def predict_all(productivity: ProductivityInput, health: HealthInput):
    prod   = predict_productivity(productivity)
    health = predict_health(health)
    return {
        "productivity": prod,
        "health":       health
    }