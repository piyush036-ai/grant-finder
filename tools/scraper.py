import httpx
from bs4 import BeautifulSoup

async def scrape_page(url: str) -> str:
    """
    Fetches a web page and extracts clean text.
    Removes scripts, styles, nav elements — just content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            response = await client.get(url, follow_redirects=True)

            if response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "ads"]):
                tag.decompose()

            # Extract clean text
            text = soup.get_text(separator=" ", strip=True)

            # Normalize whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:5000]  # limit to 5000 chars per page

    except Exception as e:
        print(f"Scraping failed for {url}: {e}")
        return ""