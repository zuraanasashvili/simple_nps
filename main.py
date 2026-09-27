import csv
import io
import os
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

import db
import sentiment

load_dotenv()

app = FastAPI(title="simple_nps")
INDEX = Path(__file__).with_name("static") / "index.html"


def parse_csv(text):
    """Returns (rows, errors). rows: list of (iso_date, score, review_or_None)."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"date", "score"} <= set(reader.fieldnames):
        raise ValueError("CSV needs 'date' and 'score' columns")
    rows, errors = [], []
    for line, rec in enumerate(reader, start=2):
        try:
            d = date.fromisoformat(rec["date"].strip()).isoformat()
            score = int(rec["score"])
            if not 0 <= score <= 10:
                raise ValueError(f"score {score} not in 0-10")
        except (ValueError, AttributeError) as e:
            errors.append(f"line {line}: {e}")
            continue
        review = (rec.get("review") or "").strip() or None
        rows.append((d, score, review))
    return rows, errors


@app.post("/api/upload")
async def upload(file: UploadFile):
    try:
        text = (await file.read()).decode("utf-8-sig")
        rows, errors = parse_csv(text)
    except (UnicodeDecodeError, ValueError) as e:
        raise HTTPException(400, str(e))
    db.insert_responses(db.connect(), rows)
    return {"inserted": len(rows), "rejected": len(errors), "errors": errors[:20]}


def _range(start, end):
    return (start.isoformat() if start else None, end.isoformat() if end else None)


@app.get("/api/nps")
def nps(
    start: date | None = Query(None, alias="from"),
    end: date | None = Query(None, alias="to"),
    freq: Literal["week", "month"] = "month",
):
    conn = db.connect()
    s, e = _range(start, end)
    return {**db.nps_summary(conn, s, e), "series": db.nps_series(conn, s, e, freq)}


@app.get("/api/scores")
def scores(
    start: date | None = Query(None, alias="from"),
    end: date | None = Query(None, alias="to"),
):
    return db.score_counts(db.connect(), *_range(start, end))


@app.post("/api/sentiment/run")
async def sentiment_run(background: BackgroundTasks):
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise HTTPException(400, "TYPESAFE_API_KEY not set, add it to .env")
    if sentiment.state["running"]:
        raise HTTPException(409, "Sentiment run already in progress")
    pending = db.sentiment_progress(db.connect())["pending"]
    sentiment.state["running"] = True
    background.add_task(sentiment.run_pending)
    return {"queued": pending}


@app.get("/api/sentiment/status")
def sentiment_status():
    return {**db.sentiment_progress(db.connect()), **sentiment.state}


@app.get("/api/sentiment/summary")
def sentiment_summary(
    start: date | None = Query(None, alias="from"),
    end: date | None = Query(None, alias="to"),
    freq: Literal["week", "month"] = "month",
):
    return db.sentiment_summary(db.connect(), *_range(start, end), freq)


@app.get("/api/reviews")
def reviews(
    start: date | None = Query(None, alias="from"),
    end: date | None = Query(None, alias="to"),
    sentiment: (
        Literal[
            "very_negative",
            "negative",
            "neutral",
            "positive",
            "very_positive",
            "pending",
        ]
        | None
    ) = None,
    max_conf: float | None = Query(None, ge=0, le=1),
    mismatch: bool = False,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return db.reviews(
        db.connect(), *_range(start, end), sentiment, max_conf, mismatch, limit, offset
    )


@app.get("/")
def index():
    return FileResponse(INDEX)
