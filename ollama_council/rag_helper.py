#!/usr/bin/env python3
"""
rag_helper.py

RAG helper using:
  - SQLite DB: ./filebase.sqlite
  - Ollama embeddings (nomic-embed-text)
  - NumPy cosine similarity

Provides:
    - get_context(question, k=6) -> str
    - ask_with_filebase(question) -> str
"""

import os
import json
import sqlite3
import textwrap

import numpy as np
import requests

DB_PATH = "/home/landon/Documents/Code-Repos/ollama_council/filebase.sqlite"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# -----------------------------
# EMBEDDINGS
# -----------------------------

def get_embedding(text: str):
    """Call Ollama embedding API and return a NumPy vector."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    return np.array(data["embedding"], dtype=np.float32)


# -----------------------------
# DB ACCESS
# -----------------------------

def get_connection():
    if not os.path.exists(DB_PATH):
        raise RuntimeError(
            "SQLite DB not found at ./filebase.sqlite — "
            "Run file_indexing.py first to build the index."
        )
    return sqlite3.connect(DB_PATH)


def load_all_chunks(conn: sqlite3.Connection):
    """
    Load all chunks from the database.

    Returns list of dicts:
    {
        "path": str,
        "chunk_index": int,
        "content": str,
        "embedding": np.ndarray
    }
    """
    cur = conn.cursor()
    cur.execute("SELECT path, chunk_index, content, embedding FROM chunks")
    rows = cur.fetchall()

    chunks = []
    for path, chunk_idx, content, emb_json in rows:
        try:
            emb_list = json.loads(emb_json)
            emb_vec = np.array(emb_list, dtype=np.float32)
        except Exception:
            continue

        chunks.append({
            "path": path,
            "chunk_index": chunk_idx,
            "content": content,
            "embedding": emb_vec
        })

    return chunks


# -----------------------------
# COSINE SIMILARITY
# -----------------------------

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray):
    """
    Compute cosine similarity between query_vec (d,)
    and doc_vecs (n, d).
    """
    q_norm = np.linalg.norm(query_vec)
    d_norms = np.linalg.norm(doc_vecs, axis=1)

    if q_norm == 0.0:
        return np.zeros_like(d_norms)

    sims = np.zeros_like(d_norms)
    valid = d_norms > 0
    sims[valid] = (doc_vecs[valid] @ query_vec) / (d_norms[valid] * q_norm)

    return sims


# -----------------------------
# RAG RETRIEVAL
# -----------------------------

def get_context(question: str, k: int = 6) -> str:
    """
    Search the vector DB and return the top-k chunks as a formatted context string.
    """
    conn = get_connection()
    chunks = load_all_chunks(conn)
    conn.close()

    if not chunks:
        return "No chunks indexed. Run file_indexing.py first."

    q_vec = get_embedding(question)

    doc_vecs = np.stack([c["embedding"] for c in chunks], axis=0)
    sims = cosine_similarity(q_vec, doc_vecs)

    k = min(k, len(chunks))
    top_indices = np.argsort(-sims)[:k]

    parts = []
    for idx in top_indices:
        c = chunks[idx]
        similarity = float(sims[idx])

        parts.append(
            textwrap.dedent(f"""
            File: {c['path']} (chunk {c['chunk_index']})  [score={similarity:.3f}]
            --------------------------------------------------
            {c['content']}
            """).strip()
        )

    return "\n\n==================================================\n\n".join(parts)


# -----------------------------
# OPTIONAL DIRECT QA
# -----------------------------

def ask_with_filebase(question: str) -> str:
    """
    Send question + retrieved context directly to Ollama using /api/generate.
    """
    context = get_context(question)

    prompt = textwrap.dedent(f"""
    You are a software assistant with access to a local filebase.

    Use the context below to answer the question.
    If something is not in the context, say so.

    === CONTEXT START ===
    {context}
    === CONTEXT END ===

    Question: {question}
    """)

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    # /api/generate returns: { "response": "...", ... }
    return data["response"]

