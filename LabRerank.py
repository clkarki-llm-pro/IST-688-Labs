"""Lab: Re-ranking a RAG pipeline (STARTER).

Two-stage retrieval over SciFact, a set of 5,183 biomedical abstracts and 300
test claims with known relevant abstracts. Stage 1 (given) is a fast bi-encoder
search. Stage 2 (yours) is a cross-encoder that re-scores the stage 1 candidates.

Search for "TODO" to find the four places you write code.
"""
# # ---------- Lab answers ----------
# Step 5a: Why does build_index() put the title in front of every chunk?
# Long abstracts are split into chunks of about 120 words. Chunks from the
# middle of an abstract often don't mention what the paper is about (e.g.
# "dialysis rates were higher in..." without saying what caused it). Adding
# the title gives every chunk that topic, so the search can match it to the
# right claim instead of missing it.
#
# Step 5b: Why is to_doc_ids() needed?
# The search ranks chunks, but one abstract can produce several chunks in the
# results. to_doc_ids() turns the ranked chunks into a ranked list of unique
# abstracts, keeping the search's order and skipping repeats. This matters
# because (1) the LLM should get k different papers, not the same paper twice,
# and (2) the answer key lists abstracts, not chunks, so we need abstract IDs
# to check results and score recall and MRR.

import json
import time

import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA_DIR = "Lab-08-Data/scifact"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "gpt-4.1-mini"

CHUNK_WORDS = 120    # words per chunk
CHUNK_OVERLAP = 30   # words shared by neighbouring chunks
CANDIDATES = 30      # chunks stage 1 hands to the reranker
CONTEXT_K = 3        # abstracts that go into the prompt
EVAL_QUERIES = 100   # test claims used in Part C

# TODO (Part D): write the system prompt. It must
#   - tell the model to use only the abstracts in {context}
#   - ask for a verdict: SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE
#   - ask it to cite the abstracts it used by their [id]
SYSTEM_TEMPLATE = "{context}"


# ---------- Given: loading ----------
@st.cache_data
def load_scifact(data_dir=DATA_DIR):
    """Return (corpus, queries, qrels).

    corpus:  doc_id -> {"title", "text"}
    queries: query_id -> claim text (the 300 test claims)
    qrels:   query_id -> set of relevant doc_ids (the answer key)
    """
    corpus = {}
    with open(f"{data_dir}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}

    queries = {}
    with open(f"{data_dir}/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels = {}
    with open(f"{data_dir}/qrels/test.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            query_id, doc_id, _ = line.strip().split("\t")
            qrels.setdefault(query_id, set()).add(doc_id)

    return corpus, queries, qrels


# ---------- Given: chunking ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into windows of `size` words that overlap by `overlap` words."""
    words = text.split()
    step = size - overlap
    starts = range(0, max(len(words) - overlap, 1), step)
    return [" ".join(words[start:start + size]) for start in starts]


# ---------- Given: embedding ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def build_index(size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Chunk every abstract, embed every chunk, keep a chunk -> doc_id map."""
    corpus, _, _ = load_scifact()
    doc_ids, texts = [], []
    for doc_id, doc in corpus.items():
        for piece in chunk_text(doc["text"], size, overlap):
            doc_ids.append(doc_id)
            texts.append(f"{doc['title']}. {piece}")

    vectors = load_embedder().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return {"doc_ids": doc_ids, "texts": texts, "vectors": vectors}


# ---------- Given: stage 1 retrieval ----------
def retrieve(index, query, k=CANDIDATES):
    """Return the top k chunks as [(chunk_index, cosine_score), ...]."""
    query_vector = load_embedder().encode(query, normalize_embeddings=True)
    scores = index["vectors"] @ query_vector
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def to_doc_ids(index, hits):
    """Collapse ranked chunks into ranked abstracts (first chunk wins)."""
    ranked = []
    for chunk_index, _ in hits:
        doc_id = index["doc_ids"][chunk_index]
        if doc_id not in ranked:
            ranked.append(doc_id)
    return ranked


# ---------- Part B: stage 2 re-ranking ----------
@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL)


def rerank(index, query, hits):
    """Re-score stage 1 hits with the cross-encoder, best first."""
    chunk_indexes = [chunk_index for chunk_index, _ in hits]
    pairs = [(query, index["texts"][chunk_index]) for chunk_index in chunk_indexes]
    scores = load_reranker().predict(pairs)
    reranked = [(chunk_index, float(score)) for chunk_index, score in zip(chunk_indexes, scores)]
    return sorted(reranked, key=lambda hit: hit[1], reverse=True)

# ---------- Given: metrics ----------
def recall_at_k(ranked, relevant, k):
    """Share of the relevant abstracts that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked, relevant):
    """1 / rank of the first relevant abstract (0 if none was retrieved)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


# ---------- Part C: measure before and after ----------
def evaluate(index, n_queries=EVAL_QUERIES, candidates=CANDIDATES, k=CONTEXT_K):
    """Average recall@k, MRR and latency, with and without re-ranking.

    TODO (Part C): for the first n_queries claims in qrels,
      - run retrieve(), then rerank(), timing each with time.perf_counter()
      - turn both hit lists into doc ids with to_doc_ids()
      - score both with recall_at_k() and reciprocal_rank()
    Return a dict of averages with these keys:
      recall_before, recall_after, rr_before, rr_after, retrieve_ms, rerank_ms
    """
    raise NotImplementedError("Part C")


# ---------- Part D: generation ----------
def build_context(doc_ids):
    corpus, _, _ = load_scifact()
    return "\n\n".join(
        f"[{doc_id}] {corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
        for doc_id in doc_ids
    )


def verify_claim(claim, doc_ids):
    """TODO (Part D): send SYSTEM_TEMPLATE (filled with build_context(doc_ids))
    and the claim to CHAT_MODEL, and return the model's reply text."""
    raise NotImplementedError("Part D")


# ---------- UI ----------
def show_docs(doc_ids, relevant):
    corpus, _, _ = load_scifact()
    for rank, doc_id in enumerate(doc_ids, start=1):
        mark = " ✅" if doc_id in relevant else ""
        st.markdown(f"**{rank}. [{doc_id}]**{mark} {corpus[doc_id]['title']}")


def main():
    st.title("Lab: Re-ranking a RAG Pipeline")
    corpus, queries, qrels = load_scifact()

    candidates = st.sidebar.slider("Stage 1 candidates", 5, 100, CANDIDATES, step=5)
    k = st.sidebar.slider("Abstracts sent to the LLM", 1, 5, CONTEXT_K)

    with st.spinner("Chunking and embedding 5,183 abstracts (first run only)..."):
        index = build_index()
    st.caption(f"{len(corpus):,} abstracts, {len(index['texts']):,} chunks indexed")

    # ----- Part A: pick a claim, look at stage 1 -----
    test_ids = list(qrels)
    query_id = st.selectbox(
        "Pick a test claim", test_ids, format_func=lambda qid: queries[qid]
    )
    custom = st.text_input("...or type your own claim (no answer key)")
    claim = custom or queries[query_id]
    relevant = set() if custom else qrels[query_id]

    hits = retrieve(index, claim, candidates)
    before = to_doc_ids(index, hits)

    # ----- Part B: side by side -----
    start = time.perf_counter()
    after = to_doc_ids(index, rerank(index, claim, hits))
    rerank_ms = (time.perf_counter() - start) * 1000

    left, right = st.columns(2)
    with left:
        st.subheader("Without re-ranking")
        show_docs(before[:k], relevant)
    with right:
        st.subheader("With re-ranking")
        show_docs(after[:k], relevant)
    st.caption(f"Re-ranking {candidates} candidates took {rerank_ms:.0f} ms. ✅ = in the answer key.")

    # ----- Part C -----
    # TODO (Part C): add a button that runs evaluate(index, EVAL_QUERIES, candidates, k)
    # and shows Recall@k, MRR and added latency with st.metric (use delta= for the change).

    # ----- Part D -----
    # TODO (Part D): add a button that calls verify_claim() twice, once with before[:k]
    # and once with after[:k], and shows the two answers in side-by-side columns.


if __name__ == "__main__":
    main()
