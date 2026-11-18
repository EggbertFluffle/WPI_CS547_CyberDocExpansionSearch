import sys
from pathlib import Path
from typing import Any, List

import numpy as np
from llama_cpp import Llama
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus" / "exploits"
EMBEDDINGS_PATH = PROJECT_ROOT / "engine" / "indexing" / "embeddings.npy"
CHUNKS_PATH = PROJECT_ROOT / "engine" / "indexing" / "chunks.tsv"

MODEL_NAME = "saidutta69/LFM2.5-8B-A1B-heretic"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

ORIGINAL_MARKER = "#<-- Original Content -->#"

TOP_K_CHUNKS = 30  # each retriever (dense, BM25) returns this many; reranker keeps this many
TOP_N_EXPLOITS = 5
MAX_NEW_TOKENS = 1500  # room for the model's <think> block + the final explanation


# Load the prebuilt index
def to_corpus_rel(raw_path):
    """chunks.tsv stores paths like ./../../corpus/exploits/linux/local/3.c -> 'linux/local/3.c'."""
    p = raw_path.replace("\\", "/")
    marker = "corpus/exploits/"
    return p.split(marker, 1)[1] if marker in p else p


def load_index():
    """Return (chunks as LangChain Documents, unit-length embedding matrix)."""
    for path in (EMBEDDINGS_PATH, CHUNKS_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Expected embeddings.npy and chunks.tsv in engine/indexing/.")

    chunks = []
    last_doc, chunk_index = None, 0
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            raw_doc, _, text = line.partition("\t")
            doc_path = to_corpus_rel(raw_doc)
            chunk_index = chunk_index + 1 if doc_path == last_doc else 0
            last_doc = doc_path
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_path": doc_path,
                        "exploit_id": Path(doc_path).stem,
                        "chunk_index": chunk_index,
                    },
                )
            )

    vectors = np.load(EMBEDDINGS_PATH)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise SystemExit(
            f"embeddings.npy has {vectors.shape[0]} rows but chunks.tsv has {len(chunks)} chunks; "
            "they must come from the same run."
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)

    print(f"Loaded {len(chunks)} chunks, embeddings shape {vectors.shape}")
    return chunks, vectors


# Query embedding
def load_embedding_model():
    return Llama.from_pretrained(
        repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
        filename="*Q8_0.gguf",
        embedding=True,
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        verbose=False,
    )


def embed_query(llm, query):
    out = llm.create_embedding(f"search_query: {query}")
    q = np.array(out["data"][0]["embedding"], dtype=np.float32)
    return q / np.linalg.norm(q)


class NpyDenseRetriever(BaseRetriever):
    """Dense retriever over embeddings.npy: cosine similarity = dot product (vectors are unit length)."""

    llm: Any
    vectors: Any
    chunks: List[Document]
    k: int = TOP_K_CHUNKS

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        scores = self.vectors @ embed_query(self.llm, query)
        top = np.argsort(scores)[::-1][: self.k]
        return [
            Document(
                page_content=self.chunks[i].page_content,
                metadata={**self.chunks[i].metadata, "cosine": float(scores[i])},
            )
            for i in top
        ]


# Retrieval: hybrid top-K chunks -> rerank -> group by doc -> top unique docs
def retrieve_top_exploits(query, retriever, dense_retriever, reranker):
    dense_hits = dense_retriever.invoke(query)
    print(f"\nDense retrieval - top {len(dense_hits)} chunks by cosine similarity:")
    for c in dense_hits:
        print(f"  {c.metadata['doc_path']:<35} chunk {c.metadata['chunk_index']:<3} cosine={c.metadata['cosine']:.4f}")

    # Hybrid retrieval (dense + BM25), then rerank
    reranked_chunks = retriever.invoke(query)

    rerank_scores = reranker.model.score(
        [(query, c.page_content) for c in reranked_chunks]
    )
    print("\nAfter reranking - reranker score:")
    for c, s in zip(reranked_chunks, rerank_scores):
        print(f"  {c.metadata['doc_path']:<35} chunk {c.metadata['chunk_index']:<3} score={s:.4f}")

    # Chunks are ordered by relevance, so the first chunk seen for a doc
    # decides that doc's position.
    unique_doc_paths = []
    for chunk in reranked_chunks:
        doc_path = chunk.metadata["doc_path"]
        if doc_path not in unique_doc_paths:
            unique_doc_paths.append(doc_path)

    return unique_doc_paths[:TOP_N_EXPLOITS]


# Re-read each doc from the corpus and keep only the ORIGINAL content
def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_original(text):
    """Return everything below the 'Original Content' comment, or None if it is missing."""
    idx = text.find(ORIGINAL_MARKER)
    if idx == -1:
        return None
    newline = text.find("\n", idx)
    return "" if newline == -1 else text[newline + 1:].lstrip("\r\n")


_name_index = None


def find_doc_file(doc_path):
    """CORPUS_DIR / doc_path, falling back to a lookup by file name if that path does not exist."""
    global _name_index
    path = CORPUS_DIR / doc_path
    if path.is_file():
        return path
    if _name_index is None:
        _name_index = {p.name: p for p in CORPUS_DIR.rglob("*") if p.is_file()}
    return _name_index.get(Path(doc_path).name)


def retrieve_original_documents(doc_paths):
    originals = []
    for doc_path in doc_paths:
        path = find_doc_file(doc_path)
        if path is None:
            print(f"Warning: {doc_path} not found under {CORPUS_DIR}; skipping")
            continue
        text = read_text(path)
        content = extract_original(text)
        if content is None:
            print(f"Warning: no '{ORIGINAL_MARKER}' marker in {doc_path}; using the whole file")
            content = text
        originals.append(
            {
                "exploit_id": path.stem,
                "doc_path": doc_path,
                "path": str(path),
                "content": content,
            }
        )
    return originals


# LLM explanations (query + one original document at a time)
PROMPT = ChatPromptTemplate.from_template(
    """You are a cybersecurity document retrieval assistant.
The user is searching for cybersecurity exploit documentation.

User Query:
{query}

Original Exploit Document:
{document}

Explain why this exploit document is relevant to the user's query.

Instructions:
- Use only information supported by the original document.
- Explain the vulnerability and its relevance to the query.
- If the document is not directly relevant to the query, start your answer with "Not directly relevant:" and briefly say what the document is actually about. Do not stretch the connection.
- Do not invent technical details.
- Keep the explanation concise and understandable.
- Do not provide instructions for unauthorized exploitation.

Explanation:"""
)


def build_explanation_chain():
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id=MODEL_NAME,
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    chat_model = ChatHuggingFace(llm=hf_pipeline)
    return PROMPT | chat_model | StrOutputParser()


def strip_think(text):
    """Remove the model's <think>...</think> reasoning and keep only the final answer."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        return "[Model ran out of tokens while reasoning - increase MAX_NEW_TOKENS]"
    return text.strip()


def generate_explanations(query, documents, explanation_chain):
    results = []
    for doc in documents:
        explanation = explanation_chain.invoke(
            {"query": query, "document": doc["content"]}
        )
        results.append({**doc, "explanation": strip_think(explanation)})
    return results


# Full search + display
def search_exploits(query, retriever, dense_retriever, reranker, explanation_chain):
    doc_paths = retrieve_top_exploits(query, retriever, dense_retriever, reranker)
    documents = retrieve_original_documents(doc_paths)
    if not documents:
        raise SystemExit("None of the retrieved documents could be found in the corpus; check CORPUS_DIR.")
    return generate_explanations(query, documents, explanation_chain)


def main():
    query = " ".join(sys.argv[1:]) or (
        "How can an attacker execute commands remotely through a vulnerable web application?"
    )

    print(f"Corpus directory: {CORPUS_DIR}")
    chunks, vectors = load_index()
    embed_llm = load_embedding_model()

    # Hybrid retrieval: dense (embeddings.npy) + BM25 (chunks.tsv), merged by the ensemble
    dense_retriever = NpyDenseRetriever(
        llm=embed_llm, vectors=vectors, chunks=chunks, k=TOP_K_CHUNKS
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K_CHUNKS
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL),
        top_n=TOP_K_CHUNKS,  # keep all reranked chunks; grouping picks the top 5 docs
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )

    explanation_chain = build_explanation_chain()

    results = search_exploits(query, retriever, dense_retriever, reranker, explanation_chain)

    print(f"\nQuery: {query}")
    for i, r in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result {i}")
        print(f"Exploit ID: {r['exploit_id']}")
        print(f"File: {r['path']}")
        print("\nAI-Generated Explanation:")
        print(r["explanation"])


if __name__ == "__main__":
    main()