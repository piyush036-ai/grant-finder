import os
import json
import re
import asyncio
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv

load_dotenv()

_credentials = Credentials(
    url=os.environ.get("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
    api_key=os.environ.get("IBM_API_KEY")
)

granite = ModelInference(
    model_id="ibm/granite-4-h-small",
    credentials=_credentials,
    project_id=os.environ.get("IBM_PROJECT_ID"),
    params={"max_new_tokens": 800, "temperature": 0.1}
)

def _extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.
    Handles markdown fences, extra text, partial responses.
    """
    # Strip markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None

async def parse_grant_with_llm(content: str, title: str, url: str) -> dict:
    prompt = f"""You are a JSON extraction tool. Extract grant information and return ONLY a JSON object.
No explanation. No markdown. No text before or after. Just the JSON object.

Title: {title}
Content: {content[:1500]}

Return exactly this JSON structure with real values extracted from content:
{{"name": "{title}", "provider": "organization name", "amount": "funding amount", "deadline": "deadline date or ongoing", "eligibility": ["criterion 1", "criterion 2"], "sectors": ["sector"], "stages": ["Idea", "MVP", "Early Revenue"], "url": "{url}", "description": "brief description"}}"""

    try:
        response = await asyncio.to_thread(granite.generate_text, prompt=prompt)
        result = _extract_json(response)

        # If still None, return a basic structured fallback
        # so the pipeline doesn't drop this grant entirely
        if result is None:
            return {
                "name": title,
                "provider": "Unknown",
                "amount": "Check website",
                "deadline": "Check website",
                "eligibility": ["Check official website for eligibility"],
                "sectors": ["General"],
                "stages": ["Idea", "MVP", "Early Revenue"],
                "url": url,
                "description": content[:200] if content else title
            }

        return result

    except Exception as e:
        print(f"[PARSER] Error: {e}")
        return {
            "name": title,
            "provider": "Unknown",
            "amount": "Check website",
            "deadline": "Check website",
            "eligibility": ["Check official website for eligibility"],
            "sectors": ["General"],
            "stages": ["Idea", "MVP", "Early Revenue"],
            "url": url,
            "description": title
        }