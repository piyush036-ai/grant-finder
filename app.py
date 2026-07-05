import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from orchestrator import Orchestrator

load_dotenv()

app = FastAPI(
    title="Grant Finder - Agentic AI",
    description="Multi-agent grant discovery and proposal generation",
    version="1.0.0"
)

# ── Request/Response Models ──────────────────────────────────────────────────
class StartupProfile(BaseModel):
    name: str
    domain: str          # e.g. "AgriTech", "HealthTech", "EdTech"
    stage: str           # e.g. "Idea", "MVP", "Early Revenue"
    location: str        # e.g. "Pune, Maharashtra"
    description: str     # what the startup does
    team_size: int

class GrantResponse(BaseModel):
    grants: list
    proposals: list
    agent_logs: list

# ── Orchestrator instance ────────────────────────────────────────────────────
orchestrator = Orchestrator()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/find-grants", response_model=GrantResponse)
async def find_grants(profile: StartupProfile):
    try:
        result = await orchestrator.run(profile.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "running", "agents": ["scout", "extractor", "matcher", "drafter"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)