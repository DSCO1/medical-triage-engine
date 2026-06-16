"""
NabzAI – Decision Engine API Test Server
A lightweight FastAPI app for testing the decision engine via HTTP.

Usage:
    cd backend
    pip install fastapi uvicorn
    uvicorn test_api:app --reload --port 8100

Swagger UI:  http://localhost:8100/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from decision_engine import analyze_case

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="NabzAI – Decision Engine Tester",
    description=(
        "Test endpoint for the rule-based triage decision engine. "
        "Submit symptoms, severity, and duration to receive an "
        "urgency assessment with specialist recommendation."
    ),
    version="1.0.0",
)

# Allow all origins for local testing convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        example=["chest pain", "breathing issue"],
        description="List of symptom strings reported by the patient.",
    )
    severity: str = Field(
        default="normal",
        example="high",
        description="Patient-reported severity: low, normal, or high.",
    )
    duration_days: int = Field(
        default=1,
        ge=0,
        example=2,
        description="Number of days the symptoms have persisted.",
    )


class AnalyzeResponse(BaseModel):
    urgency: str = Field(..., example="High")
    risk_level: str = Field(..., example="Critical")
    specialist: str = Field(..., example="Cardiologist")
    secondary_specialist: str | None = Field(None, example="Pulmonologist")
    confidence: str = Field(..., example="98%")
    reason: str = Field(
        ...,
        example=(
            "Chest pain combined with breathing difficulty suggests "
            "a potential cardiac emergency requiring immediate attention"
        ),
    )
    matched_rules: list[str] = Field(
        ...,
        example=[
            "combo: chest pain + breathing issue -> cardiac emergency",
            "chest pain -> cardiac risk",
            "breathing issue -> respiratory distress",
        ],
    )
    steps: list[str] = Field(
        ...,
        example=[
            "Step 1: Parsed 2 symptom(s): chest pain, breathing issue",
            "Step 2: Analyzed severity='high' and duration=2 day(s)",
            "Step 3: Matched 2 individual rule(s) + 1 combo rule",
            "Step 4: Determined urgency level -> High",
            "Step 5: Mapped primary specialist -> Cardiologist | secondary -> Pulmonologist",
            "Step 6: Generated recommendation and confidence score",
        ],
    )


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a patient case",
    description=(
        "Submit patient symptoms, severity, and duration. "
        "The decision engine will return urgency, risk level, specialist, "
        "secondary specialist, confidence score, matched rules, "
        "reasoning steps, and a human-readable explanation."
    ),
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    result = analyze_case(
        symptoms=request.symptoms,
        severity=request.severity,
        duration_days=request.duration_days,
    )
    return result


@app.get("/", summary="Health check")
async def root():
    return {
        "status": "online",
        "service": "NabzAI Decision Engine Tester",
        "docs": "/docs",
    }
