import os
import chromadb
from sentence_transformers import SentenceTransformer
from tools.scraper import scrape_page
from tools.parser import parse_grant_with_llm

class ExtractorAgent:
    """
    Agent 2: Intelligence Extractor
    
    RAG Pipeline:
    1. Scrape full page content from URLs
    2. Chunk content into manageable pieces
    3. Embed chunks using sentence-transformers
    4. Store in ChromaDB vector store
    5. Query ChromaDB for relevant grant details
    6. Use Granite to extract structured JSON from retrieved chunks
    """

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.chroma = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.chroma.get_or_create_collection(
            name="grants",
            metadata={"hnsw:space": "cosine"}
        )

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list:
        words = text.split()
        chunks = []
        overlap = 50

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    async def extract(self, raw_grants: list) -> list:
        structured = []

        for grant in raw_grants:
            print(f"[EXTRACTOR] Processing: {grant['title'][:50]}")

            # Scrape full page
            content = await scrape_page(grant["url"])
            print(f"[EXTRACTOR] Scraped {len(content)} chars from {grant['url'][:40]}")

            # Use snippet as fallback if scraping fails
            if not content:
                content = grant.get("snippet", "")
                print(f"[EXTRACTOR] Using snippet fallback")

            # Chunk the content
            chunks = self._chunk_text(content) if content else []

            if chunks:
                for i, chunk in enumerate(chunks[:5]):
                    embedding = self.embedder.encode(chunk).tolist()
                    self.collection.upsert(
                        ids=[f"{grant['url']}_{i}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"url": grant["url"], "title": grant["title"]}]
                    )

            # Use Granite to extract structured data
            context = content if content else grant.get("snippet", grant["title"])
            extracted = await parse_grant_with_llm(
                content=context,
                title=grant["title"],
                url=grant["url"]
            )

            if extracted:
                structured.append(extracted)
                print(f"[EXTRACTOR] ✓ Extracted: {extracted.get('name', 'Unknown')}")

        return structured