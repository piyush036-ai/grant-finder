import os
from serpapi import GoogleSearch

async def search_grants(query: str) -> list:
    """
    Uses SerpAPI to search Google and returns
    structured results (title, url, snippet).
    
    SerpAPI handles Google's bot detection,
    JavaScript rendering, and rate limiting.
    """
    params = {
        "q": query,
        "api_key": os.environ.get("SERPAPI_KEY"),
        "num": 5,
        "gl": "in",   # geo-locate to India
        "hl": "en"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    grants = []
    for r in results.get("organic_results", []):
        grants.append({
            "title":   r.get("title", ""),
            "url":     r.get("link", ""),
            "snippet": r.get("snippet", ""),
            "source":  r.get("displayed_link", "")
        })

    return grants