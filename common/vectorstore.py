"""Vector stores behind one interface: add(), search(), len().

* NumpyVectorStore  - exact cosine search over an in-memory matrix, with metadata filters.
                      Persisted as .npz + .json. Fine up to ~1M vectors on a laptop.
* ChromaVectorStore - same interface over Chroma (a local persistent vector database).
* PgVectorStore     - same interface over PostgreSQL + pgvector (Cloud SQL in production).
                      Needs PGVECTOR_DSN and `pip install psycopg[binary]`; not run by the offline tests.

make_store() picks one from the FDE_VECTOR_BACKEND env var (numpy | chroma | pgvector).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _matches(meta: dict, where: dict | None) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


class NumpyVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def __len__(self):
        return len(self.ids)

    def add(self, ids, vectors, texts, metadatas=None):
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        self.vecs = np.vstack([self.vecs, vectors])
        self.ids += list(ids)
        self.texts += list(texts)
        self.metas += list(metadatas or [{} for _ in ids])

    def search(self, query_vec, k: int = 5, where: dict | None = None) -> list[Hit]:
        if not self.ids:
            return []
        sims = self.vecs @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if where:
            mask = np.array([_matches(m, where) for m in self.metas])
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Hit(self.ids[i], float(sims[i]), self.texts[i], self.metas[i]) for i in top if np.isfinite(sims[i])]

    def save(self, path: str):
        np.savez_compressed(path + ".npz", vecs=self.vecs)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "ids": self.ids, "texts": self.texts, "metas": self.metas}, f)

    @classmethod
    def load(cls, path: str) -> "NumpyVectorStore":
        with open(path + ".json", encoding="utf-8") as f:
            d = json.load(f)
        s = cls(d["dim"])
        s.vecs = np.load(path + ".npz")["vecs"]
        s.ids, s.texts, s.metas = d["ids"], d["texts"], d["metas"]
        return s


class ChromaVectorStore:
    """Chroma with our own embeddings (vectors are passed in, so Chroma never downloads a model)."""

    def __init__(self, dim: int, path: str | None = None, collection: str = "fde"):
        import chromadb

        self.dim = dim
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def __len__(self):
        return self.col.count()

    def add(self, ids, vectors, texts, metadatas=None):
        metas = [m or {"_": 0} for m in (metadatas or [None] * len(ids))]  # chroma rejects empty dicts
        vectors = np.asarray(vectors)
        for i in range(0, len(ids), 1000):
            self.col.add(ids=list(ids[i:i + 1000]), embeddings=vectors[i:i + 1000].tolist(),
                         documents=list(texts[i:i + 1000]), metadatas=metas[i:i + 1000])

    def search(self, query_vec, k: int = 5, where: dict | None = None) -> list[Hit]:
        if where and len(where) > 1:
            where = {"$and": [{a: b} for a, b in where.items()]}
        r = self.col.query(query_embeddings=[np.asarray(query_vec).tolist()], n_results=k, where=where or None)
        return [Hit(i, 1.0 - d, t, m) for i, d, t, m in
                zip(r["ids"][0], r["distances"][0], r["documents"][0], r["metadatas"][0])]


class PgVectorStore:
    """PostgreSQL + pgvector (e.g. Cloud SQL for PostgreSQL). Table and HNSW index are created on first use."""

    def __init__(self, dim: int, dsn: str | None = None, table: str = "fde_chunks"):
        import psycopg

        self.dim, self.table = dim, table
        self.conn = psycopg.connect(dsn or os.environ["PGVECTOR_DSN"], autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id text PRIMARY KEY, text text, "
                          f"metadata jsonb, embedding vector({dim}))")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} "
                          f"USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def _lit(v) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v).reshape(-1)) + "]"

    def __len__(self):
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def add(self, ids, vectors, texts, metadatas=None):
        metas = metadatas or [{} for _ in ids]
        with self.conn.cursor() as cur:
            cur.executemany(f"INSERT INTO {self.table} VALUES (%s,%s,%s,%s::vector) ON CONFLICT (id) DO UPDATE "
                            f"SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                            [(i, t, json.dumps(m), self._lit(v)) for i, v, t, m in zip(ids, vectors, texts, metas)])

    def search(self, query_vec, k: int = 5, where: dict | None = None) -> list[Hit]:
        q = self._lit(query_vec)
        cond, params = "", [q]
        if where:
            cond = "WHERE metadata @> %s::jsonb"   # JSONB containment = metadata filter
            params.append(json.dumps(where))
        params += [q, k]
        rows = self.conn.execute(
            f"SELECT id, text, metadata, 1 - (embedding <=> %s::vector) FROM {self.table} {cond} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s", params).fetchall()
        return [Hit(r[0], float(r[3]), r[1], r[2]) for r in rows]


def make_store(dim: int, backend: str | None = None, **kw):
    backend = backend or os.environ.get("FDE_VECTOR_BACKEND", "numpy")
    if backend == "chroma":
        return ChromaVectorStore(dim, **kw)
    if backend == "pgvector":
        return PgVectorStore(dim, **kw)
    return NumpyVectorStore(dim)
