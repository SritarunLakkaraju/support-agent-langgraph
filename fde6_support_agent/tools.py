"""The agent's knowledge and tools.

FaqIndex     RAG over the FAQ (hybrid BM25 + hashing embeddings, scored by term coverage -> a confidence)
OrderBook    repair-status lookup. Local: the CSV export of the shop's Google Sheet. Production: SheetsOrderBook
             reads the live sheet with the Sheets API (service-account credentials).
Bookings     appointment slots + bookings (SQLite locally; Firestore in production)
Handoffs     queue of conversations a human must pick up, with the reason
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import sqlite3
import threading

from common.embed import HashingEmbedder
from common.text import BM25, tokenize

from .data import ALT_PHRASINGS, FAQS, OPEN_HOURS, ORDERS_CSV, generate_orders


class FaqIndex:
    def __init__(self, faqs=FAQS):
        self.faqs = faqs
        texts = [f"{q} {ALT_PHRASINGS.get(fid, '')} {a}" for fid, q, a in faqs]
        self.bm25 = BM25(texts)
        self.emb = HashingEmbedder(dim=512).fit(texts)
        self.vecs = self.emb.embed(texts)
        self.q_terms = [set(tokenize(f"{q} {ALT_PHRASINGS.get(fid, '')}")) for fid, q, _ in faqs]
        self.all_terms = [set(tokenize(t)) for t in texts]
        self.unseen = max(self.bm25.idf.values())

    def search(self, query: str) -> tuple[str, str, float]:
        """Return (faq_id, answer, confidence in [0, 1])."""
        qt = set(tokenize(query))
        if not qt:
            return "", "", 0.0
        dense = self.vecs @ self.emb.embed([query])[0]
        sparse = self.bm25.scores(query)
        best, best_score, best_cov = 0, -1.0, 0.0
        total = sum(self.bm25.idf.get(t, self.unseen) for t in qt)
        for i in range(len(self.faqs)):
            cov = sum(self.bm25.idf.get(t, self.unseen) for t in qt if t in self.all_terms[i]) / total
            qcov = sum(self.bm25.idf.get(t, self.unseen) for t in qt if t in self.q_terms[i]) / total
            score = 0.45 * cov + 0.25 * qcov + 0.2 * float(dense[i]) + 0.1 * min(1.0, sparse.get(i, 0) / 10)
            if score > best_score:
                best, best_score, best_cov = i, score, cov
        fid, _, ans = self.faqs[best]
        return fid, ans, round(min(1.0, 0.7 * best_cov + 0.3 * best_score), 3)


class OrderBook:
    def __init__(self, path: str = ORDERS_CSV):
        if not os.path.exists(path):
            generate_orders(path=path)
        with open(path, encoding="utf-8") as f:
            self.rows = {r["order_id"].upper(): r for r in csv.DictReader(f)}
        self.by_phone = {r["customer_phone"]: r for r in self.rows.values()}

    def lookup(self, order_id: str | None = None, phone: str | None = None) -> dict | None:
        if order_id:
            return self.rows.get(order_id.upper())
        if phone:
            return self.by_phone.get(re.sub(r"\D", "", phone)[-10:])
        return None


class SheetsOrderBook(OrderBook):
    """Reads the live Google Sheet (pip install google-api-python-client google-auth). Not run offline."""

    def __init__(self, sheet_id: str, rng: str = "Orders!A:G"):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
        values = build("sheets", "v4", credentials=creds).spreadsheets().values().get(
            spreadsheetId=sheet_id, range=rng).execute()["values"]
        header, *data = values
        self.rows = {r[0].upper(): dict(zip(header, r)) for r in data}
        self.by_phone = {r["customer_phone"]: r for r in self.rows.values()}


class Store:
    """Bookings, handoffs, conversation log and feedback in one SQLite file (Firestore in production)."""

    def __init__(self, path: str = ":memory:"):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS bookings (booking_id TEXT PRIMARY KEY, conversation_id TEXT, slot TEXT UNIQUE,
                device TEXT, issue TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS handoffs (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
                reason TEXT, last_message TEXT, status TEXT DEFAULT 'open', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
                channel TEXT, user_text TEXT, reply TEXT, intent TEXT, outcome TEXT, confidence REAL, latency_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS feedback (conversation_id TEXT, rating INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

    def q(self, sql, args=()):
        with self.lock, self.con:
            return self.con.execute(sql, args).fetchall()

    # bookings
    def free_slots(self, day: dt.date) -> list[str]:
        if day.weekday() not in OPEN_HOURS:
            return []
        start, end = OPEN_HOURS[day.weekday()]
        taken = {r[0] for r in self.q("SELECT slot FROM bookings WHERE slot LIKE ?", (f"{day.isoformat()}%",))}
        return [f"{day.isoformat()} {h:02d}:00" for h in range(start, end) if f"{day.isoformat()} {h:02d}:00" not in taken]

    def book(self, conversation_id: str, slot: str, device: str, issue: str) -> str | None:
        n = self.q("SELECT count(*) FROM bookings")[0][0]
        bid = f"BK-{501 + n}"
        try:
            self.q("INSERT INTO bookings (booking_id, conversation_id, slot, device, issue) VALUES (?,?,?,?,?)",
                   (bid, conversation_id, slot, device, issue))
        except sqlite3.IntegrityError:
            return None  # slot was just taken by someone else
        return bid

    def handoff(self, conversation_id: str, reason: str, last_message: str):
        self.q("INSERT INTO handoffs (conversation_id, reason, last_message) VALUES (?,?,?)",
               (conversation_id, reason, last_message))

    def log(self, **kw):
        self.q("INSERT INTO messages (conversation_id, channel, user_text, reply, intent, outcome, confidence, latency_ms)"
               " VALUES (:conversation_id, :channel, :user_text, :reply, :intent, :outcome, :confidence, :latency_ms)", kw)

