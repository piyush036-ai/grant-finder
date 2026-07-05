import asyncio
from agents.scout import ScoutAgent
from agents.extractor import ExtractorAgent
from agents.matcher import MatcherAgent
from agents.drafter import DrafterAgent

class Orchestrator:
    """
    Master controller that coordinates all 4 agents.
    
    Pipeline:
    1. Scout finds raw grant listings (web search)
    2. Extractor parses and structures grant data (RAG + NLP)
    3. Matcher scores grants against startup profile (ReAct)
    4. Drafter generates proposals for top matches (generative)
    """

    def __init__(self):
        self.scout     = ScoutAgent()
        self.extractor = ExtractorAgent()
        self.matcher   = MatcherAgent()
        self.drafter   = DrafterAgent()
        self.logs      = []

    def log(self, agent: str, message: str):
        entry = f"[{agent.upper()}] {message}"
        self.logs.append(entry)
        print(entry)  # visible in Railway logs

    async def run(self, profile: dict) -> dict:
        self.logs = []  # reset logs for each run

        # ── Phase 1: Scout ───────────────────────────────────────────────
        self.log("orchestrator", f"Starting grant search for {profile['name']}")
        raw_grants = await self.scout.find_grants(
            domain=profile["domain"],
            stage=profile["stage"],
            location=profile["location"]
        )
        self.log("scout", f"Found {len(raw_grants)} raw grant sources")

        # ── Phase 2: Extractor ───────────────────────────────────────────
        self.log("orchestrator", "Extracting structured data from grant sources")
        structured_grants = await self.extractor.extract(raw_grants)
        self.log("extractor", f"Extracted {len(structured_grants)} structured grants")

        # ── Phase 3: Matcher ─────────────────────────────────────────────
        self.log("orchestrator", "Scoring grants against startup profile")
        matched_grants = await self.matcher.match(structured_grants, profile)
        top_grants = matched_grants[:3]  # take top 3
        self.log("matcher", f"Top match score: {top_grants[0]['score'] if top_grants else 0}/100")

        # ── Phase 4: Drafter ─────────────────────────────────────────────
        self.log("orchestrator", "Generating proposals for top grants")
        proposals = await self.drafter.draft(top_grants, profile)
        self.log("drafter", f"Generated {len(proposals)} proposals")

        return {
            "grants": matched_grants,
            "proposals": proposals,
            "agent_logs": self.logs
        }