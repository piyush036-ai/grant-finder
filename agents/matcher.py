import re
from tools.parser import granite

class MatcherAgent:

    async def match(self, grants: list, profile: dict) -> list:
        scored = []

        for grant in grants:
            score, reasoning = await self._score_grant(grant, profile)
            scored.append({
                **grant,
                "score": score,
                "reasoning": reasoning,
                "match_level": self._score_to_label(score)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _score_grant(self, grant: dict, profile: dict) -> tuple:
        prompt = f"""Score this grant match from 0 to 100.

Startup domain: {profile['domain']}
Startup stage: {profile['stage']}
Grant name: {grant.get('name', 'Unknown')}
Grant sectors: {grant.get('sectors', ['General'])}

Give only a number between 0 and 100. Example: 75"""

        try:
            response = granite.generate_text(prompt=prompt).strip()
            print(f"[MATCHER] Raw response for {grant.get('name','?')[:20]}: '{response}'")

            score = self._extract_score(response)
            print(f"[MATCHER] Score: {score}")
            return score, response

        except Exception as e:
            print(f"[MATCHER] Error: {e}")
            return 50, "Default"

    def _extract_score(self, text: str) -> int:
        # Find first number in response
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            for n in numbers:
                val = int(n)
                if 0 <= val <= 100:
                    return val

        return 40

    def _score_to_label(self, score: int) -> str:
        if score >= 75: return "Excellent Match"
        if score >= 50: return "Good Match"
        if score >= 25: return "Partial Match"
        return "Low Match"