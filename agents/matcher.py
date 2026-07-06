import os
import asyncio
import re
from tools.parser import granite

class MatcherAgent:

    async def match(self, grants: list, profile: dict) -> list:
        tasks = [self._score_grant(grant, profile) for grant in grants]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored = []
        for grant, result in zip(grants, results):
            if isinstance(result, Exception):
                print(f"[MATCHER] Scoring failed for {grant.get('name','?')}: {result}")
                score, reasoning = 0, "Scoring failed"
            else:
                score, reasoning = result

            scored.append({
                **grant,
                "score": score,
                "reasoning": reasoning,
                "match_level": self._score_to_label(score)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _score_grant(self, grant: dict, profile: dict) -> tuple:
        prompt = f"""You are an expert grant eligibility analyst.

Startup Profile:
- Name: {profile['name']}
- Domain: {profile['domain']}
- Stage: {profile['stage']}
- Location: {profile['location']}
- Description: {profile['description']}
- Team Size: {profile['team_size']}

Grant Details:
- Name: {grant.get('name', 'Unknown')}
- Provider: {grant.get('provider', 'Unknown')}
- Amount: {grant.get('amount', 'Unknown')}
- Eligibility: {grant.get('eligibility', [])}
- Sectors: {grant.get('sectors', [])}
- Stages: {grant.get('stages', [])}

Analyze step by step:
STEP 1 - Domain match: Does startup domain match grant sectors? (0-25 points)
STEP 2 - Stage match: Is startup stage eligible? (0-25 points)
STEP 3 - Location match: Is location eligible? (0-25 points)
STEP 4 - Profile strength: Team size and description quality (0-25 points)

After each step state the points awarded and reason.
End with: TOTAL_SCORE: [number]
Then: SUMMARY: [one sentence explaining the match]"""

        # Run the blocking granite.generate_text call in a thread so it
        # doesn't block the event loop and can run concurrently with others
        response = await asyncio.to_thread(granite.generate_text, prompt=prompt)

        print(f"[MATCHER] Raw response for {grant.get('name','?')[:20]}: '{response[:80]}'")

        score = self._extract_score(response)
        return score, response

    def _extract_score(self, reasoning: str) -> int:
        match = re.search(r"TOTAL_SCORE:\s*(\d+)", reasoning)
        if match:
            return min(int(match.group(1)), 100)
        return 0

    def _score_to_label(self, score: int) -> str:
        if score >= 75: return "Excellent Match"
        if score >= 50: return "Good Match"
        if score >= 25: return "Partial Match"
        return "Low Match"