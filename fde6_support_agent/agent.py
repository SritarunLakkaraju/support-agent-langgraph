"""QuickFix support agent: a LangGraph graph with per-conversation memory (thread_id = conversation id).

    classify --> faq      (RAG; low confidence -> handoff)
             --> order    (needs an order id or phone; asks for it, remembers it is waiting)
             --> booking  (slot filling over several turns: device, issue, day, time -> checks free slots -> books)
             --> handoff  (user asks for a person, is upset, or the bot is not confident)
             --> smalltalk (greeting / thanks)

Every answer is either a verbatim FAQ answer, a value from the order sheet, or a booking confirmation, so the
bot cannot invent prices or policies. Anything it is unsure about goes to a human with the reason recorded.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .data import DEVICES, ISSUES, SHOP
from .tools import FaqIndex, OrderBook, Store

FAQ_THRESHOLD = 0.5
HUMAN = re.compile(r"\b(human|real person|someone|agent|manager|staff|call me|complain\w*|refund|angry|worst|useless|"
                   r"terrible|cheat\w*|fraud|lawyer)\b", re.I)
ORDER = re.compile(r"\b(order|status|ready|track\w*|my phone|my repair|qf-?\d+|ticket|collect|pick ?up my)\b", re.I)
BOOK = re.compile(r"\b(book\w*|appointment|slot|schedule|reserve|visit you|come in)\b", re.I)
GREET = re.compile(r"^\s*(hi|hello|hey|namaste|good (morning|evening|afternoon))\b[\s!.]*$", re.I)
BYE = re.compile(r"^\s*(thanks?|thank you|ok thanks|bye|great,? thanks)\b", re.I)
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
BRANDS = ["iphone", "samsung", "galaxy", "oneplus", "redmi", "xiaomi", "pixel", "vivo", "oppo", "realme"]


def today() -> dt.date:
    return dt.date.fromisoformat(os.environ["FDE6_TODAY"]) if os.environ.get("FDE6_TODAY") else dt.date.today()


class ChatState(TypedDict, total=False):
    conversation_id: str
    message: str
    intent: str
    pending: dict
    reply: str
    outcome: str
    confidence: float
    failed_lookups: int


def parse_day(text: str) -> dt.date | None:
    low, t = text.lower(), today()
    if "today" in low:
        return t
    if "tomorrow" in low:
        return t + dt.timedelta(days=1)
    for i, d in enumerate(DAYS):
        if re.search(rf"\b{d}\b", low):
            return t + dt.timedelta(days=(i - t.weekday()) % 7 or 7)
    if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", low):
        return dt.date.fromisoformat(m.group(1))
    return None


def parse_hour(text: str) -> int | None:
    if m := re.search(r"\b(\d{1,2})(?::00)?\s*(am|pm)\b", text.lower()):
        h = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        return h
    if m := re.search(r"\b(\d{1,2}):00\b", text):
        return int(m.group(1))
    return None


def parse_device(text: str) -> str | None:
    low = text.lower()
    for d in DEVICES:
        if d.lower() in low:
            return d
    for b in BRANDS:
        if m := re.search(rf"\b{b}\b[\w ]{{0,12}}", low):
            return m.group(0).strip().title()
    return None


def parse_issue(text: str) -> str | None:
    low = text.lower()
    aliases = {"cracked screen": ["screen", "display", "cracked", "broken glass"], "battery drain": ["battery"],
               "charging port": ["charging", "charger", "port"], "water damage": ["water", "wet"],
               "camera fault": ["camera"], "speaker issue": ["speaker", "sound", "mic"]}
    for issue, kws in aliases.items():
        if any(k in low for k in kws):
            return issue
    return None


def build_agent(store: Store | None = None, faq: FaqIndex | None = None, orders: OrderBook | None = None,
                checkpointer=None):
    store, faq, orders = store or Store(), faq or FaqIndex(), orders or OrderBook()

    def classify(s: ChatState):
        msg, pending = s["message"], s.get("pending") or {}
        if HUMAN.search(msg):
            intent = "handoff"
        elif GREET.match(msg) or BYE.match(msg):
            intent = "smalltalk"
        elif pending.get("flow") and not (BOOK.search(msg) and pending["flow"] != "booking"):
            intent = pending["flow"]                    # we asked a follow-up question: this is the answer
        elif re.search(r"\b(reschedul\w*|cancel\w*)\b", msg, re.I) and not re.search(r"\bbk-\d+", msg, re.I):
            intent = "faq"                              # "how do I reschedule?" is a policy question
        elif re.search(r"\bqf-?\d+\b", msg, re.I) or (ORDER.search(msg) and not BOOK.search(msg)
                                                       and re.search(r"\b(status|ready|track\w*|order)\b", msg, re.I)):
            intent = "order"
        elif BOOK.search(msg):
            intent = "booking"
        else:
            intent = "faq"
        return {"intent": intent}

    def smalltalk(s: ChatState):
        if BYE.match(s["message"]):
            return {"reply": "You're welcome! Have a great day.", "outcome": "answered", "confidence": 1.0, "pending": {}}
        return {"reply": f"Hi! I'm the {SHOP} assistant. I can answer questions about repairs and prices, check your "
                         f"repair status, or book an appointment.", "outcome": "answered", "confidence": 1.0}

    def faq_node(s: ChatState):
        fid, ans, conf = faq.search(s["message"])
        if conf < FAQ_THRESHOLD:
            store.handoff(s["conversation_id"], f"low confidence ({conf})", s["message"])
            return {"reply": "I'm not sure about that one, so I've passed your question to our team. Someone will "
                             "reply here shortly (Mon-Sat, 10 AM to 8 PM).", "outcome": "handoff", "confidence": conf,
                    "pending": {}}
        return {"reply": ans, "outcome": f"faq:{fid}", "confidence": conf, "pending": {}}

    def order_node(s: ChatState):
        msg = s["message"]
        oid = re.search(r"\bqf-?(\d+)\b", msg, re.I)
        phone = re.search(r"\b(?:\+?91[\s-]?)?([6-9]\d{9})\b", msg.replace(" ", ""))
        if not oid and not phone:
            return {"reply": "Sure, I can check that. What is your order ID (like QF-1023) or the phone number you "
                             "gave at the counter?", "outcome": "asked_followup", "confidence": 1.0,
                    "pending": {"flow": "order"}}
        row = orders.lookup(order_id=f"QF-{oid.group(1)}" if oid else None, phone=phone.group(1) if phone else None)
        if not row:
            fails = s.get("failed_lookups", 0) + 1
            if fails >= 2:
                store.handoff(s["conversation_id"], "order not found twice", msg)
                return {"reply": "I still can't find that order, so I've asked a team member to check it for you.",
                        "outcome": "handoff", "confidence": 0.0, "pending": {}, "failed_lookups": 0}
            return {"reply": "I couldn't find that order. Could you double-check the ID (it starts with QF-)?",
                    "outcome": "asked_followup", "confidence": 0.0, "pending": {"flow": "order"}, "failed_lookups": fails}
        extra = " You can collect it any time during opening hours." if row["status"] == "Ready for pickup" else ""
        return {"reply": f"Order {row['order_id']} ({row['device']}, {row['issue']}): status is '{row['status']}'. "
                         f"Quoted Rs {row['quote_rs']}, promised by {row['promised_date']}.{extra}",
                "outcome": "order_status", "confidence": 1.0, "pending": {}, "failed_lookups": 0}

    def booking_node(s: ChatState):
        msg = s["message"]
        p = dict(s.get("pending") or {})
        p["flow"] = "booking"
        p.setdefault("device", parse_device(msg))
        p["device"] = p["device"] or parse_device(msg)
        p["issue"] = p.get("issue") or parse_issue(msg)
        d = parse_day(msg)
        h = parse_hour(msg)
        if d:
            p["day"] = d.isoformat()
        if h is not None:
            p["hour"] = h
        for key, question in (("device", "Which phone model is it?"),
                              ("issue", "What's the problem with it (screen, battery, charging, water damage...)?"),
                              ("day", "Which day would you like to come in?")):
            if not p.get(key):
                return {"reply": question, "outcome": "asked_followup", "confidence": 1.0, "pending": p}
        day = dt.date.fromisoformat(p["day"])
        free = store.free_slots(day)
        if not free:
            p.pop("day")
            return {"reply": f"Sorry, we're closed or fully booked on {day:%A %d %b}. Which other day works?",
                    "outcome": "asked_followup", "confidence": 1.0, "pending": p}
        if p.get("hour") is None:
            return {"reply": f"On {day:%A %d %b} we have {', '.join(x[-5:] for x in free[:4])} free. Which time?",
                    "outcome": "asked_followup", "confidence": 1.0, "pending": p}
        slot = f"{p['day']} {p['hour']:02d}:00"
        if slot not in free:
            p.pop("hour")
            return {"reply": f"{slot[-5:]} isn't available that day. Free times: {', '.join(x[-5:] for x in free[:4])}.",
                    "outcome": "asked_followup", "confidence": 1.0, "pending": p}
        bid = store.book(s["conversation_id"], slot, p["device"], p["issue"])
        if not bid:
            p.pop("hour")
            return {"reply": "That slot was just taken. Please pick another time.", "outcome": "asked_followup",
                    "confidence": 1.0, "pending": p}
        return {"reply": f"Booked! {bid}: {p['device']} ({p['issue']}) on {day:%A %d %b} at {slot[-5:]}. "
                         f"Diagnosis is free. Reply with your booking ID to reschedule.",
                "outcome": "booked", "confidence": 1.0, "pending": {}}

    def handoff_node(s: ChatState):
        store.handoff(s["conversation_id"], "customer asked for a person / is upset", s["message"])
        return {"reply": "I'm sorry about this. I've passed your conversation to the shop owner, who will reply here "
                         "personally. You can also call us at 98450 12345.", "outcome": "handoff", "confidence": 1.0,
                "pending": {}}

    g = StateGraph(ChatState)
    for name, fn in (("classify", classify), ("faq", faq_node), ("order", order_node), ("booking", booking_node),
                     ("handoff", handoff_node), ("smalltalk", smalltalk)):
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", lambda s: s["intent"], ["faq", "order", "booking", "handoff", "smalltalk"])
    for n in ("faq", "order", "booking", "handoff", "smalltalk"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class SupportAgent:
    def __init__(self, store: Store | None = None, checkpointer=None):
        self.store = store or Store()
        self.graph = build_agent(self.store, checkpointer=checkpointer)

    def reply(self, conversation_id: str, text: str, channel: str = "web") -> dict:
        t0 = time.perf_counter()
        s = self.graph.invoke({"conversation_id": conversation_id, "message": text[:1000]},
                              {"configurable": {"thread_id": conversation_id}})
        ms = round((time.perf_counter() - t0) * 1000, 2)
        self.store.log(conversation_id=conversation_id, channel=channel, user_text=text, reply=s["reply"],
                       intent=s["intent"], outcome=s["outcome"], confidence=s.get("confidence"), latency_ms=ms)
        return {"reply": s["reply"], "intent": s["intent"], "outcome": s["outcome"], "latency_ms": ms}
