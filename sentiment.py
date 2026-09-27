"""Review sentiment via Jev (TypeSafe). CLI: python sentiment.py ["review text" ...]"""

import asyncio
import sys

from typesafe_sdk import AsyncTypeSafeClient, Score, TypeSafeError

import db

SENTIMENT = Score(
    instructions="Overall sentiment the customer expresses about the product or company",
    criteria=[
        "Very negative: angry, cancelling or leaving, strong complaints",
        "Negative: disappointed, reports clear problems",
        "Neutral or mixed: states facts, or praise and complaints balance out",
        "Positive: satisfied, at most minor complaints",
        "Very positive: delighted, praises strongly or recommends it",
    ],
)
LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
CONCURRENCY = 5


state = {"running": False, "failed": 0, "last_error": None}


def label(score):
    return LABELS[min(int(score + 0.5), len(LABELS) - 1)]


async def analyze(client, text):
    """Returns (label, score 0..4, confidence 0..1)."""
    r = await client.system_one(state=text, questions={"sentiment": SENTIMENT})
    a = r.answers["sentiment"]
    return label(a.score), a.score, a.confidence


async def run_pending(client=None):
    """Analyze every review without sentiment. Saves each result as it lands, so reruns resume."""
    state.update(running=True, failed=0, last_error=None)
    conn = db.connect()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(client, row_id, text):
        async with sem:
            try:
                result = await analyze(client, text)
            except TypeSafeError as e:
                state["failed"] += 1
                state["last_error"] = f"{type(e).__name__}: {e}"
                return
        db.save_sentiment(conn, row_id, *result)

    try:
        pending = db.pending_reviews(conn)
        if client:
            await asyncio.gather(*(one(client, *r) for r in pending))
        else:
            async with AsyncTypeSafeClient() as c:
                await asyncio.gather(*(one(c, *r) for r in pending))
    except Exception as e:
        state["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        state["running"] = False


async def _cli(texts):
    async with AsyncTypeSafeClient() as client:
        results = await asyncio.gather(*(analyze(client, t) for t in texts))
    for t, (lab, score, conf) in zip(texts, results):
        print(f"{lab:14} score={score:.2f} conf={conf:.2f}  {t}")


if __name__ == "__main__":
    import csv

    from dotenv import load_dotenv

    load_dotenv()
    texts = sys.argv[1:]
    if not texts:
        with open("sample.csv") as f:
            texts = sorted({r["review"] for r in csv.DictReader(f) if r["review"]})
    asyncio.run(_cli(texts))
