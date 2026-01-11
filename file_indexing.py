#!/usr/bin/env python3
"""
file_indexing.py

Index a local "filebase" directory into a SQLite database using
Ollama's embedding API (nomic-embed-text).

DB file: ./filebase.sqlite

Run:
    python file_indexing.py
"""

import os
import sys
import json
import requests
import sqlite3

FILEBASE_DIR = "/home/landon/Documents/Code-Repos/ollama_council"
DB_PATH = "./filebase.sqlite"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".sh"
}

MAX_CHARS = 1500
OVERLAP_CHARS = 200
BATCH_SIZE = 50


def debug(msg: str):
    print(msg, file=sys.stderr)


def iter_files(root: str):
    """Yield full file paths of allowed extensions."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                yield os.path.join(dirpath, name)


def read_file(path: str) -> str:
    """Return the file contents, or '' if unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        debug(f"[skip] {path}: {e}")
        return ""


def chunk_text(text: str):
    """Simple character-based chunking with overlap."""
    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + MAX_CHARS, n)
        chunk = text[start:end]
        chunks.append(chunk)

        # move forward but keep overlap
        start = end - OVERLAP_CHARS
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def get_embedding(text: str):
    """Call Ollama's embedding API and return a list[float]."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    if "embedding" not in data:
        raise RuntimeError(f"Bad embedding response: {data}")

    return data["embedding"]


def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()


def clear_db(conn: sqlite3.Connection):
    """Full rebuild each time: wipe all chunks."""
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()


def main():
    if not os.path.isdir(FILEBASE_DIR):
        print(f"ERROR: FILEBASE_DIR does not exist: {FILEBASE_DIR}")
        sys.exit(1)

    print(f"[*] Indexing directory: {FILEBASE_DIR}")
    print(f"[*] SQLite DB: {DB_PATH}")
    print(f"[*] Embedding model: {EMBED_MODEL}")
    print()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    clear_db(conn)

    cur = conn.cursor()

    total_chunks = 0
    batch_count = 0

    for abs_path in iter_files(FILEBASE_DIR):
        rel_path = os.path.relpath(abs_path, FILEBASE_DIR)
        content = read_file(abs_path)
        if not content.strip():
            continue

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk{i}"
            debug(f"[chunk] {chunk_id}")

            try:
                vector = get_embedding(chunk)
            except Exception as e:
                debug(f"[error] embedding failed for {chunk_id}: {e}")
                continue

            cur.execute(
                """
                INSERT OR REPLACE INTO chunks (id, path, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, rel_path, i, chunk, json.dumps(vector)),
            )

            total_chunks += 1
            batch_count += 1

            if batch_count >= BATCH_SIZE:
                conn.commit()
                batch_count = 0

    if batch_count > 0:
        conn.commit()

    conn.close()
    print(f"[*] Done. Indexed {total_chunks} chunks into {DB_PATH}.")


if __name__ == "__main__":
    main()
