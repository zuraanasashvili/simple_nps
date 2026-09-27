import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from typesafe_sdk import TypeSafeError

import db
import sentiment
from main import parse_csv


def fresh(scores, dates=None):
    conn = db.connect(":memory:")
    dates = dates or ["2026-01-15"] * len(scores)
    db.insert_responses(conn, [(d, s, None) for d, s in zip(dates, scores)])
    return conn


def test_nps_summary():
    assert db.nps_summary(fresh([9, 10, 10]))["nps"] == 100
    assert db.nps_summary(fresh([0, 6, 9, 10]))["nps"] == 0
    assert db.nps_summary(fresh([7, 8]))["nps"] == 0
    assert db.nps_summary(fresh([0, 3]))["nps"] == -100
    assert db.nps_summary(fresh([]))["nps"] is None
    s = db.nps_summary(fresh([2, 7, 9, 10]))
    assert (s["promoters"], s["passives"], s["detractors"], s["nps"]) == (2, 1, 1, 25)


def test_date_filter_and_series():
    conn = fresh([10, 0, 9], ["2026-01-10", "2026-02-10", "2026-02-20"])
    assert db.nps_summary(conn, "2026-02-01", "2026-02-28")["nps"] == 0
    assert [(r["period"], r["nps"], r["n"]) for r in db.nps_series(conn, freq="month")] == [
        ("2026-01", 100.0, 1),
        ("2026-02", 0.0, 2),
    ]
    assert db.score_counts(conn) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert db.score_counts(conn, "2026-01-01", "2026-01-31") == [0] * 10 + [1]


def test_parse_csv():
    rows, errors = parse_csv("date,score,review\n2026-01-01,9,great\n2026-13-01,5,\n2026-01-02,11,x\n2026-01-03,4,  \n")
    assert rows == [("2026-01-01", 9, "great"), ("2026-01-03", 4, None)]
    assert len(errors) == 2


def test_sentiment_label():
    assert [sentiment.label(s) for s in (0, 0.49, 0.5, 2.5, 3.6, 4)] == [
        "very_negative", "very_negative", "negative", "positive", "very_positive", "very_positive"]


def test_run_pending_with_fake_client():
    """Stub Jev: 'bad' reviews score 0, others 4, 'boom' raises. Checks DB writes, skips, failures."""
    class FakeClient:
        async def system_one(self, state, questions):
            if state == "boom":
                raise TypeSafeError("fake failure")
            a = SimpleNamespace(score=0.0 if "bad" in state else 4.0, confidence=0.9)
            return SimpleNamespace(answers={"sentiment": a})

    # run_pending opens its own connection, so point the default DB path at a temp file
    tmp = tempfile.TemporaryDirectory()
    orig = db.connect.__defaults__
    db.connect.__defaults__ = (Path(tmp.name) / "t.db",)
    try:
        _check_run_pending(FakeClient)
    finally:
        db.connect.__defaults__ = orig
        tmp.cleanup()


def _check_run_pending(FakeClient):
    conn = db.connect()
    db.insert_responses(conn, [("2026-01-01", 2, "bad app"), ("2026-01-01", 10, "love it"),
                               ("2026-01-01", 5, "boom"), ("2026-01-01", 8, None)])
    asyncio.run(sentiment.run_pending(FakeClient()))
    got = conn.execute("SELECT review, sentiment FROM responses ORDER BY id").fetchall()
    assert [tuple(r) for r in got] == [("bad app", "very_negative"), ("love it", "very_positive"),
                                       ("boom", None), (None, None)]
    assert db.sentiment_progress(conn) == {"analyzed": 2, "pending": 1}
    assert sentiment.state["failed"] == 1 and not sentiment.state["running"]


def test_sentiment_views():
    conn = db.connect(":memory:")
    db.insert_responses(conn, [
        ("2026-01-05", 10, "promoter, angry"),   # mismatch
        ("2026-01-06", 9, "promoter, happy"),
        ("2026-02-01", 2, "detractor, happy"),   # mismatch
        ("2026-02-02", 3, "detractor, unsure"),
        ("2026-02-03", 8, "not analyzed"),
        ("2026-02-04", 7, None),
    ])
    for row_id, score, conf in [(1, 0.2, 0.9), (2, 4.0, 1.0), (3, 3.0, 0.95), (4, 1.4, 0.6)]:
        db.save_sentiment(conn, row_id, sentiment.label(score), score, conf)

    s = db.sentiment_summary(conn)
    assert s["breakdown"] == {"very_negative": 1, "very_positive": 1, "positive": 1, "negative": 1}
    assert [(r["period"], r["avg_score"], r["n"]) for r in s["series"]] == [("2026-01", 2.1, 2), ("2026-02", 2.2, 2)]
    assert {(r["category"], r["sentiment"]): r["n"] for r in s["by_category"]} == {
        ("promoter", "very_negative"): 1, ("promoter", "very_positive"): 1,
        ("detractor", "positive"): 1, ("detractor", "negative"): 1}

    ids = lambda **kw: [r["id"] for r in db.reviews(conn, **kw)["rows"]]
    assert ids() == [5, 4, 3, 2, 1]  # newest first, row without review excluded
    assert ids(mismatch=True) == [3, 1]
    assert ids(max_conf=0.75) == [4]
    assert ids(sentiment="pending") == [5]
    assert ids(sentiment="positive") == [3]
    assert ids(start="2026-02-01", limit=2, offset=1) == [4, 3]
    assert db.reviews(conn, limit=1)["total"] == 5


if __name__ == "__main__":
    test_nps_summary()
    test_date_filter_and_series()
    test_parse_csv()
    test_sentiment_label()
    test_run_pending_with_fake_client()
    test_sentiment_views()
    print("ok")
