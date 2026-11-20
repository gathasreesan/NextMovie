import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

print("Loading BAAI/bge-m3 model...")
model = SentenceTransformer("BAAI/bge-m3")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate normalized 1024-dim dense embeddings using BAAI/bge-m3."""
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def run_encoder(batch_size: int = 32):
    resp = (
        supabase.table("movies")
        .select("tmdb_id, metadata_soup")
        .is_("embedding", "null")
        .execute()
    )
    movies = resp.data or []
    if not movies:
        print("All movies already have embeddings!")
        return

    print(f"Generating BAAI/bge-m3 embeddings for {len(movies)} movies...")
    updated = 0

    for i in range(0, len(movies), batch_size):
        batch = movies[i : i + batch_size]
        texts = [m["metadata_soup"] for m in batch]
        embeddings = embed_batch(texts)

        for movie, vector in zip(batch, embeddings):
            supabase.table("movies").update({"embedding": vector}).eq(
                "tmdb_id", movie["tmdb_id"]
            ).execute()
            updated += 1

        print(f"Embedded {updated}/{len(movies)} movies...")

    print(f"Done! Total movies embedded: {updated}")


if __name__ == "__main__":
    run_encoder()