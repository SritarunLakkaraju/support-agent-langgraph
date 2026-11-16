"""Shared building blocks used by all six FDE projects.

Everything here runs fully offline. Cloud-backed variants (Vertex AI embeddings,
Gemini, Chroma, pgvector) are selected only when their credentials / packages exist.
"""
