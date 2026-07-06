import os
import chromadb
from tools.scraper import scrape_page
from tools.parser import parse_grant_with_llm

class ExtractorAgent:
    def __init__(self):
        # Lazy load to avoid startup crash on Railway
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.use_embedder = True
        except Exception as e:
            print(f"[EXTRACTOR] Embedder failed to load: {e}")
            self.embedder = None
            self.use_embedder = False

        try:
            self.chroma = chromadb.PersistentClient(path="data/chroma_db")
            self.collection = self.chroma.get_or_create_collection(
                name="grants",
                metadata={"hnsw:space": "cosine"}
            )
            self.use_chroma = True
        except Exception as e:
            print(f"[EXTRACTOR] ChromaDB failed: {e}")
            self.use_chroma = False

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
            content = await scrape_page(grant["url"])
            print(f"[EXTRACTOR] Scraped {len(content)} chars")

            if not content:
                content = grant.get("snippet", "")

            # Only use ChromaDB if available
            if self.use_embedder and self.use_chroma and content:
                chunks = self._chunk_text(content)
                for i, chunk in enumerate(chunks[:5]):
                    try:
                        embedding = self.embedder.encode(chunk).tolist()
                        self.collection.upsert(
                            ids=[f"{grant['url']}_{i}"],
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{"url": grant["url"], "title": grant["title"]}]
                        )
                    except Exception as e:
                        print(f"[EXTRACTOR] Embed error: {e}")

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