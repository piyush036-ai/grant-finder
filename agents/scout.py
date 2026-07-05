import os
import httpx
from tools.search import search_grants
from tools.scraper import scrape_page

class ScoutAgent:
    def __init__(self):
        self.sources = [
            "startupindia.gov.in",
            "birac.nic.in",
            "sidbi.in",
            "msme.gov.in",
        ]

    def _build_queries(self, domain: str, stage: str, location: str) -> list:
        state = location.split(",")[-1].strip() if "," in location else location
        queries = [
            f"government grants {domain} startups India 2025 2026",
            f"startup funding {domain} {state} eligibility apply",
            f"MSME grant scheme {domain} {stage} stage India",
        ]
        return queries

    async def find_grants(self, domain: str, stage: str, location: str) -> list:
        queries = self._build_queries(domain, stage, location)
        raw_results = []

        for query in queries[:3]:
            try:
                results = await search_grants(query)
                print(f"[SCOUT] Query '{query[:40]}...' → {len(results)} results")
                raw_results.extend(results)
            except Exception as e:
                print(f"[SCOUT] Search failed: {e}")

        # deduplicate by URL
        seen = set()
        unique = []
        for r in raw_results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        return unique[:10]