"""RAG retrieval and grounded tailoring package."""

from doot.rag.retriever import BM25Retriever, EvidenceChunk, parse_markdown_evidence
from doot.rag.tailor import collect_evidence_chunks, load_stories_markdown, tailor_for_job

__all__ = [
    "BM25Retriever",
    "EvidenceChunk",
    "collect_evidence_chunks",
    "load_stories_markdown",
    "parse_markdown_evidence",
    "tailor_for_job",
]
