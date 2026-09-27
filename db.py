import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("nps.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL,
  score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 10),
  review TEXT,
  sentiment TEXT,
  sentiment_score REAL,
  sentiment_confidence REAL,
  topic TEXT
);
CREATE INDEX IF NOT EXISTS idx_date ON responses(date);
"""

PERIODS = {"week": "%Y-W%W", "month": "%Y-%m"}
MIN_DATE, MAX_DATE = "0000-01-01", "9999-12-31"


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def insert_responses(conn, rows):
    """rows: iterable of (date, score, review)."""
    with conn:
        conn.executemany("INSERT INTO responses (date, score, review) VALUES (?, ?, ?)", rows)


def nps_summary(conn, start=None, end=None):
    r = conn.execute(
        """SELECT COUNT(*) AS n,
                  COALESCE(SUM(score >= 9), 0) AS promoters,
                  COALESCE(SUM(score BETWEEN 7 AND 8), 0) AS passives,
                  COALESCE(SUM(score <= 6), 0) AS detractors
           FROM responses WHERE date BETWEEN ? AND ?""",
        (start or MIN_DATE, end or MAX_DATE),
    ).fetchone()
    n = r["n"]
    nps = round(100 * (r["promoters"] - r["detractors"]) / n, 1) if n else None
    return {**dict(r), "nps": nps}


def nps_series(conn, start=None, end=None, freq="month"):
    rows = conn.execute(
        """SELECT strftime(?, date) AS period,
                  ROUND(100.0 * (SUM(score >= 9) - SUM(score <= 6)) / COUNT(*), 1) AS nps,
                  COUNT(*) AS n,
                  SUM(score >= 9) AS promoters,
                  SUM(score BETWEEN 7 AND 8) AS passives,
                  SUM(score <= 6) AS detractors
           FROM responses WHERE date BETWEEN ? AND ?
           GROUP BY period ORDER BY period""",
        (PERIODS[freq], start or MIN_DATE, end or MAX_DATE),
    ).fetchall()
    return [dict(r) for r in rows]


def pending_reviews(conn):
    return conn.execute(
        "SELECT id, review FROM responses WHERE review IS NOT NULL AND sentiment IS NULL"
    ).fetchall()


def save_sentiment(conn, row_id, label, score, confidence):
    with conn:
        conn.execute(
            "UPDATE responses SET sentiment = ?, sentiment_score = ?, sentiment_confidence = ? WHERE id = ?",
            (label, score, confidence, row_id),
        )


def sentiment_progress(conn):
    r = conn.execute(
        """SELECT COUNT(sentiment) AS analyzed,
                  COALESCE(SUM(review IS NOT NULL AND sentiment IS NULL), 0) AS pending
           FROM responses"""
    ).fetchone()
    return dict(r)


CATEGORY_SQL = "CASE WHEN score >= 9 THEN 'promoter' WHEN score >= 7 THEN 'passive' ELSE 'detractor' END"
# Promoter who writes negatively, or detractor who writes positively (thresholds = label boundaries).
MISMATCH_SQL = "((score >= 9 AND sentiment_score < 1.5) OR (score <= 6 AND sentiment_score >= 2.5))"


def sentiment_summary(conn, start=None, end=None, freq="month"):
    where = "sentiment IS NOT NULL AND date BETWEEN ? AND ?"
    args = (start or MIN_DATE, end or MAX_DATE)
    breakdown = dict(conn.execute(
        f"SELECT sentiment, COUNT(*) FROM responses WHERE {where} GROUP BY sentiment", args
    ).fetchall())
    series = conn.execute(
        f"""SELECT strftime(?, date) AS period, ROUND(AVG(sentiment_score), 2) AS avg_score, COUNT(*) AS n
            FROM responses WHERE {where} GROUP BY period ORDER BY period""",
        (PERIODS[freq], *args),
    ).fetchall()
    by_category = conn.execute(
        f"""SELECT {CATEGORY_SQL} AS category, sentiment, COUNT(*) AS n
            FROM responses WHERE {where} GROUP BY category, sentiment""",
        args,
    ).fetchall()
    return {
        "breakdown": breakdown,
        "series": [dict(r) for r in series],
        "by_category": [dict(r) for r in by_category],
    }


def reviews(conn, start=None, end=None, sentiment=None, max_conf=None, mismatch=False, limit=50, offset=0):
    """Filtered review page. WHERE clauses are fixed strings; all values go through placeholders."""
    where, args = ["review IS NOT NULL", "date BETWEEN ? AND ?"], [start or MIN_DATE, end or MAX_DATE]
    if sentiment == "pending":
        where.append("sentiment IS NULL")
    elif sentiment:
        where.append("sentiment = ?")
        args.append(sentiment)
    if max_conf is not None:
        where.append("sentiment_confidence < ?")
        args.append(max_conf)
    if mismatch:
        where.append(MISMATCH_SQL)
    w = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM responses WHERE {w}", args).fetchone()[0]
    rows = conn.execute(
        f"""SELECT id, date, score, review, sentiment, sentiment_score, sentiment_confidence,
                   COALESCE({MISMATCH_SQL}, 0) AS mismatch
            FROM responses WHERE {w} ORDER BY date DESC, id DESC LIMIT ? OFFSET ?""",
        (*args, limit, offset),
    ).fetchall()
    return {"total": total, "rows": [dict(r) for r in rows]}


def score_counts(conn, start=None, end=None):
    """Count per score, index 0..10."""
    counts = [0] * 11
    for r in conn.execute(
        "SELECT score, COUNT(*) FROM responses WHERE date BETWEEN ? AND ? GROUP BY score",
        (start or MIN_DATE, end or MAX_DATE),
    ):
        counts[r[0]] = r[1]
    return counts
