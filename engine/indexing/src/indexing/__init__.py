import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from llama_cpp import Llama

OVERLAP = 50
CHUNK_LENGTH = 500
BATCH_SIZE = 16

CORPUS_DIR = "./../../corpus/exploits/linux/local/"
EMBEDDINGS_PATH = "embeddings.npy"
CHUNKS_PATH = "chunks.tsv"
DEFAULT_QUERY = "io_uring send"

TOP_K_CHUNKS = 30
TOP_K_DOCS = 10


def load_model():
    return Llama.from_pretrained(
        repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
        filename="*Q8_0.gguf",
        embedding=True,
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        verbose=False,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (or reuse) an embedding index and run a query."
    )
    parser.add_argument(
        "-q",
        "--query",
        default=DEFAULT_QUERY,
        help=f"Query text to search for (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument(
        "--corpus",
        default=CORPUS_DIR,
        help=f"Directory of documents to index (default: {CORPUS_DIR!r})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-chunk and re-embed even if the index files already exist.",
    )
    args = parser.parse_args(argv)

    llm = load_model()

    # Reuse the saved index when possible so runs are fast.
    index = None if args.force else load_index()
    if index is not None:
        chunks, vectors = index
        print(
            f"Using existing index with {len(chunks)} chunks "
            f"({EMBEDDINGS_PATH}, {CHUNKS_PATH}). "
            "Pass --force to rebuild."
        )
    else:
        chunks = chunk_dir(args.corpus)
        if not chunks:
            raise SystemExit(f"No documents found under {args.corpus!r}.")
        vectors = embed_chunks(llm, chunks)
        save_index(chunks, vectors)
        print(f"Saved index with {len(chunks)} chunks.")

    for doc in search(llm, chunks, vectors, args.query):
        print(doc)


def chunk_file(document):
    chunks = []
    with open(document, "r", encoding="utf-8", errors="ignore") as file:
        words = file.read().split()
    step = CHUNK_LENGTH - OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i : i + CHUNK_LENGTH]
        chunks.append((document, " ".join(chunk_words)))
        if i + CHUNK_LENGTH >= len(words):
            break
    return chunks


def chunk_dir(entry):
    chunks = []
    # Walk recursively so a nested corpus (e.g. exploits/linux/local/...)
    # is indexed too, not just files sitting directly in the root.
    for root, dirs, files in os.walk(entry):
        dirs.sort()
        for name in sorted(files):
            chunks += chunk_file(os.path.join(root, name))
    return chunks


def embed_chunks(llm, chunks):
    vectors = []
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = [
            f"search_document: {text}"
            for _, text in chunks[start : start + BATCH_SIZE]
        ]
        out = llm.create_embedding(batch)
        vectors.extend(item["embedding"] for item in out["data"])
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def save_index(chunks, vectors):
    np.save(EMBEDDINGS_PATH, vectors)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        for doc, text in chunks:
            file.write(f"{doc}\t{text.replace(chr(9), ' ')}\n")


def load_index():
    """Return (chunks, vectors) from disk, or None if they are unusable."""
    if not os.path.isfile(EMBEDDINGS_PATH) or not os.path.isfile(CHUNKS_PATH):
        return None

    chunks = []
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                if not line:
                    continue
                doc, _, text = line.partition("\t")
                chunks.append((doc, text))

        vectors = np.load(EMBEDDINGS_PATH)
    except (OSError, ValueError) as exc:
        print(f"Ignoring unreadable index ({exc}). Rebuilding.", file=sys.stderr)
        return None

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        print(
            f"Ignoring stale index: {EMBEDDINGS_PATH} has {vectors.shape[0]} "
            f"rows but {CHUNKS_PATH} has {len(chunks)} chunks. Rebuilding.",
            file=sys.stderr,
        )
        return None

    # Normalize defensively in case the saved vectors are not unit length.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return chunks, (vectors / norms).astype(np.float32)


# Returs sorted list of top k (score, document, matched chunk)
def search(llm, chunks, vectors, query, k=10):
    out = llm.create_embedding(f"search_query: {query}")
    q = np.array(out["data"][0]["embedding"], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vectors @ q  # cosine similarity (assumes `vectors` are L2-normalized)

    top_docs = {}  # dict preserves insertion order (best score first)
    for i in np.argsort(scores)[::-1]:
        doc = chunks[i][0]
        if doc not in top_docs:  # first hit = highest-scoring chunk for that doc
            top_docs[doc] = float(scores[i])
            if len(top_docs) >= TOP_K_DOCS:
                break

    return list(top_docs.items())  # [(doc, score), ...]


if __name__ == "__main__":
    main()
