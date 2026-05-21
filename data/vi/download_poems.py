"""Download a multi-author poetry corpus, tokenize, and save bag-of-words counts.

Sources (all Project Gutenberg, public domain):
    - Shakespeare's Sonnets               (PG 1041)
    - Keats, Poems (1820)                 (PG 23684)
    - Wordsworth, Poems in Two Volumes    (PG 8774)
    - Whitman, Leaves of Grass            (PG 1322)
    - Dickinson, Poems, Three Series      (PG 12242)

We chunk each text into fixed-size token windows after stopword removal,
so each "document" is a roughly equal-sized slice of one poet's corpus.
Chunking sidesteps per-poet parsing of poem boundaries.

Run once from the project root with the project venv activated:

    python data/vi/download_poems.py

Writes `data/vi/poems.pt` with:
    counts:  LongTensor (D, V)
    vocab:   list[str] of length V
    authors: list[str] of length D    (one entry per chunk)
"""

import os
import re
import ssl
import urllib.request
from collections import Counter

import certifi
import torch

OUT_PATH = "data/vi/poems.pt"

SOURCES = [
    ("shakespeare", "https://www.gutenberg.org/cache/epub/1041/pg1041.txt"),
    ("keats", "https://www.gutenberg.org/cache/epub/23684/pg23684.txt"),
    ("wordsworth", "https://www.gutenberg.org/cache/epub/8774/pg8774.txt"),
    ("whitman", "https://www.gutenberg.org/cache/epub/1322/pg1322.txt"),
    ("dickinson", "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"),
]

CHUNK_SIZE = 200  # tokens per document, after stopword removal
MIN_DF = 5  # vocabulary: word must appear in >= MIN_DF chunks
MAX_CHUNKS_PER_AUTHOR = 80  # cap per author for class balance

# Modern + Early-Modern English function words.
STOPWORDS = set("""
a about above after again against all am an an' an's and any are aren as at
be because been before being below between both but by can cannot could did
didst do does doing don done dost doth down during each ev ev'ry even every
except few for from further had hadst has hast hath have having he her here
hers herself him himself his how i if in into is it its itself just like ll
made make many may me might mine more most must my myself near nor not now
of off oft on once one only or other ought our ours ourselves out over own
re same shall shalt she should so some such than that the thee their theirs
them themselves then there these they thine this those thou though through
thus thy till to too unto until up upon us very was we were what when where
whereof which while whilst who whom whose why will wilt with within without
would ye yet you your yours yourself yourselves
""".split())

# Editorial / bibliographic apparatus that survives in some Gutenberg
# editions (tables of contents, footnotes, editor's notes). Author names
# go here too — they appear only in headers and contents pages.
STOPWORDS |= set("""
keats shakespeare wordsworth whitman dickinson byron shelley
page pages note notes footnote footnotes vol volume edition
contents preface introduction index editor poem poems ode
written first second third book line lines stanza chapter
""".split())


def fetch_text(url: str) -> str:
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\*\s*END OF.*?\*\*\*", text)
    if start and end:
        text = text[start.end() : end.start()]
    return text


def tokenize(text: str):
    return [
        w
        for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= 3 and w not in STOPWORDS
    ]


def chunk(tokens, size: int):
    return [
        tokens[i : i + size]
        for i in range(0, len(tokens), size)
        if len(tokens[i : i + size]) >= size // 2
    ]


def main():
    chunks_per_author = {}
    for author, url in SOURCES:
        raw = strip_gutenberg(fetch_text(url))
        toks = tokenize(raw)
        chs = chunk(toks, CHUNK_SIZE)
        if len(chs) > MAX_CHUNKS_PER_AUTHOR:
            # Sample evenly across the source rather than truncating.
            idx = torch.linspace(0, len(chs) - 1, MAX_CHUNKS_PER_AUTHOR).long()
            chs = [chs[i] for i in idx.tolist()]
        chunks_per_author[author] = chs
        print(f"{author:12s}: {len(toks):6d} tokens -> {len(chs):3d} chunks")

    # Build vocabulary from all chunks.
    df = Counter()
    all_chunks = []
    authors = []
    for author, chs in chunks_per_author.items():
        for c in chs:
            all_chunks.append(c)
            authors.append(author)
            for w in set(c):
                df[w] += 1

    vocab = sorted(w for w, c in df.items() if c >= MIN_DF)
    word2idx = {w: i for i, w in enumerate(vocab)}

    D, V = len(all_chunks), len(vocab)
    counts = torch.zeros(D, V, dtype=torch.long)
    for d, c in enumerate(all_chunks):
        for w in c:
            j = word2idx.get(w)
            if j is not None:
                counts[d, j] += 1

    print(
        f"\nTotal: D={D} chunks, V={V} vocab, " f"total tokens kept={int(counts.sum())}"
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(
        {"counts": counts, "vocab": vocab, "authors": authors},
        OUT_PATH,
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
