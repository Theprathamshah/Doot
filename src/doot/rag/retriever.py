"""Local BM25 retriever over verified resume and STAR story markdown blocks."""

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

_TAG_RE = re.compile(r"\[story:([a-zA-Z0-9_-]+)\]")
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.-]*")


class EvidenceChunk(BaseModel):
    """Single verifiable evidence block extracted from resume.md or stories.md."""

    chunk_id: str
    source_file: str
    heading: str
    content: str
    tags: list[str] = Field(default_factory=list)

    def render_for_prompt(self) -> str:
        """Render chunk with explicit ID for citation-grounded generation."""
        return f"[{self.chunk_id}] ({self.heading})\n{self.content}"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:48] or "section"


def parse_markdown_evidence(markdown_text: str, *, source_file: str) -> list[EvidenceChunk]:
    """Split markdown into heading-scoped EvidenceChunks preserving [story:<id>] tags."""
    if not markdown_text.strip():
        return []

    chunks: list[EvidenceChunk] = []
    current_heading = "Overview"
    current_id: str | None = None
    body_lines: list[str] = []
    seen_ids: set[str] = set()

    def _flush() -> None:
        nonlocal current_id, body_lines
        body = "\n".join(line for line in body_lines if line.strip()).strip()
        if not body:
            body_lines = []
            return
        base_id = current_id or f"{PathSlug(source_file)}:{_slugify(current_heading)}"
        cid = base_id
        suffix = 2
        while cid in seen_ids:
            cid = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(cid)
        chunks.append(
            EvidenceChunk(
                chunk_id=cid,
                source_file=source_file,
                heading=current_heading,
                content=body,
                tags=_tokenize(f"{current_heading} {body}"),
            )
        )
        body_lines = []

    for line in markdown_text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            _flush()
            raw_heading = match.group(2).strip()
            tag_match = _TAG_RE.search(raw_heading)
            current_id = tag_match.group(1) if tag_match else None
            current_heading = _TAG_RE.sub("", raw_heading).strip()
            continue
        if line.lstrip().startswith("# "):
            continue
        body_lines.append(line)

    _flush()
    return chunks


def PathSlug(name: str) -> str:
    """Return short normalized stem from filename."""
    return _slugify(name.rsplit("/", 1)[-1].split(".")[0])


class BM25Retriever:
    """Pure-Python BM25 lexical retriever over EvidenceChunks."""

    def __init__(
        self,
        chunks: list[EvidenceChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[Counter[str]] = [
            Counter(_tokenize(f"{c.heading} {c.content}")) for c in chunks
        ]
        self._doc_lens = [sum(cnt.values()) for cnt in self._doc_tokens]
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 1.0
        self._df: Counter[str] = Counter()
        for cnt in self._doc_tokens:
            for term in cnt:
                self._df[term] += 1

    def retrieve(self, query: str, *, top_k: int = 5) -> list[EvidenceChunk]:
        """Return top_k most relevant EvidenceChunks for the query."""
        if not self.chunks or top_k <= 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return self.chunks[:top_k]

        n_docs = len(self.chunks)
        scored: list[tuple[float, int, EvidenceChunk]] = []
        for idx, (chunk, tf_map, doc_len) in enumerate(
            zip(self.chunks, self._doc_tokens, self._doc_lens, strict=True)
        ):
            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            scored.append((score, -idx, chunk))

        scored.sort(reverse=True)
        return [item[2] for item in scored[:top_k]]
