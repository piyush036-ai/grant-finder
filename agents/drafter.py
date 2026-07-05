from tools.parser import granite

class DrafterAgent:
    """
    Agent 4: Proposal Drafter

    Generates customized grant application drafts.
    Each proposal is tailored to:
    - The specific grant's requirements and tone
    - The startup's unique strengths
    - The grant provider's known preferences
    """

    async def draft(self, top_grants: list, profile: dict) -> list:
        proposals = []

        for grant in top_grants:
            proposal = await self._generate_proposal(grant, profile)
            proposals.append({
                "grant_name": grant.get("name"),
                "grant_url": grant.get("url"),
                "match_score": grant.get("score"),
                "proposal": proposal
            })

        return proposals

    async def _generate_proposal(self, grant: dict, profile: dict) -> str:
        prompt = f"""You are an expert grant writer with 15 years of experience
writing successful grant applications for Indian startups.

Write a compelling grant application for:

STARTUP:
- Name: {profile['name']}
- Domain: {profile['domain']}
- Stage: {profile['stage']}
- Location: {profile['location']}
- What we do: {profile['description']}
- Team size: {profile['team_size']}

GRANT:
- Name: {grant.get('name')}
- Provider: {grant.get('provider')}
- Amount: {grant.get('amount')}
- Requirements: {grant.get('eligibility')}

Write the application with these sections:
1. Executive Summary (3 sentences, compelling hook)
2. Problem Statement (the problem we solve)
3. Our Solution (what we built/are building)
4. Market Opportunity (size and potential)
5. Why We Qualify (address specific eligibility criteria)
6. Fund Utilization Plan (how we'll use the grant)
7. Expected Impact (measurable outcomes in 12 months)

Be specific, professional, and persuasive. Use data where possible."""

        return granite.generate_text(prompt=prompt)