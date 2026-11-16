"""Text utilities: tokenisation, sentence splitting, chunking and BM25 keyword search."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being do does did
this that these those it its into than then there here what which who whom whose when where why how
can could should would may might must will shall i you he she we they me my your our their his her
not no yes any all each per about over under up down out so such only also very just more most
many much need needed get got take happen happens tell know please
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ments", "ment", "ness", "ies", "es", "ed", "ly", "s")


IRREGULAR = {"taught": "teach", "teaches": "teach", "teaching": "teach", "won": "win", "paid": "pay"}


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer: 'graduation'/'graduate' -> 'graduat', 'required'/'requires' -> 'requir'.
    Crude compared with Porter, but deterministic, dependency-free and good enough for matching."""
    if len(word) <= 4 or not word.isalpha():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            # graduation -> graduat (matches graduate -> graduat); studies -> study
            word = word[: -len(suf)] + {"ies": "y", "ation": "at", "ations": "at"}.get(suf, "")
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, drop_stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Lower-case word tokens. Keeps things like '7.5', 'cs-101' and 'b.tech' as single tokens."""
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    # "4-credit" / "end-semester": keep the whole token and also index its parts
    toks = toks + [p for t in toks if "-" in t for p in t.split("-") if p and p not in STOPWORDS]
    if stemming:
        toks = [IRREGULAR.get(t, stem(t)) for t in toks]
    return toks


def split_sentences(text: str) -> list[str]:
    # split after . ! ? followed by a capital/digit, but not after abbreviations like "Dr." or "Rs."
    parts = re.split(r"(?<=[.!?])(?<!\bDr\.)(?<!\bRs\.)(?<!\bNo\.)(?<!\bSt\.)\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunking: pack whole sentences up to max_words, overlapping by N sentences
    so a fact that straddles a boundary is still retrievable."""
    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        n = len(s.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class BM25:
    """Okapi BM25 over a fixed corpus (k1=1.5, b=0.75). Used for the keyword half of hybrid search."""

    def __init__(self, docs: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.n = len(self.docs_tokens)
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(t) for t in self.docs_tokens]
        df = Counter()
        for t in self.docs_tokens:
            df.update(set(t))
        self.idf = {w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        # inverted index: a query only touches documents that share at least one term
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tf):
            for w in tf:
                self.postings.setdefault(w, []).append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: -x[1])[:k]
