import os
import json
import asyncio
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv

load_dotenv()

# Shared Granite model instance
_credentials = Credentials(
    url=os.environ.get("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
    api_key=os.environ.get("IBM_API_KEY")
)

granite = ModelInference(
    model_id="ibm/granite-4-h-small",
    credentials=_credentials,
    project_id=os.environ.get("IBM_PROJECT_ID"),
    params={"max_new_tokens": 600, "temperature": 0.1}
    # temperature=0.1 → near-deterministic output
    # critical for structured extraction — we want JSON, not creativity
)

# Limit concurrent calls to watsonx — prevents throttling/empty responses.
# Tune this number: 2-3 is safe, raise it later if watsonx handles it fine.
_llm_semaphore = asyncio.Semaphore(3)


async def call_granite(prompt: str, retries: int = 1) -> str:
    """
    Wrapper around granite.generate_text that:
    - Limits concurrency via a semaphore (prevents API throttling)
    - Runs the blocking call in a thread (so it doesn't block the event loop)
    - Retries once automatically if the response comes back empty
    """
    async with _llm_semaphore:
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(granite.generate_text, prompt=prompt)
                if response and response.strip():
                    return response
                print(f"[GRANITE] Empty response (attempt {attempt + 1}/{retries + 1})")
            except Exception as e:
                print(f"[GRANITE] Call failed (attempt {attempt + 1}/{retries + 1}): {e}")

            if attempt < retries:
                await asyncio.sleep(1)  # brief pause before retry

    return ""


async def parse_grant_with_llm(content: str, title: str, url: str) -> dict:
    """
    Uses Granite to extract structured grant information.
    temperature=0.1 ensures consistent JSON output.
    """

    prompt = f"""Extract grant information from the following content and return ONLY a valid JSON object.
Do not include any explanation, markdown, or text outside the JSON.

Grant Title: {title}
Source URL: {url}
Content: {content[:2000]}

Return this exact JSON structure:
{{
  "name": "grant name",
  "provider": "organization providing the grant",
  "amount": "funding amount or range",
  "deadline": "application deadline",
  "eligibility": ["criterion 1", "criterion 2", "criterion 3"],
  "sectors": ["sector1", "sector2"],
  "stages": ["Idea", "MVP", "Early Revenue"],
  "url": "{url}",
  "description": "2 sentence description"
}}

JSON:"""

    response = await call_granite(prompt)

    if not response:
        print(f"[PARSER] No response for {title[:40]} — skipping")
        return None

    try:
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"[PARSER] Invalid JSON for {title[:40]}: {response[:100]}")
        return None