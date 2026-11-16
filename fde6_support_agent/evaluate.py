"""FDE-6 evaluation with scripted customer conversations (python -m fde6_support_agent.evaluate).

There is no real customer on this machine, so 40 conversations written the way customers actually message a
repair shop (typos, short answers, multi-turn) stand in for the pilot week. Each turn has an expected outcome.
Writes results/metrics.json with turn accuracy, % resolved without a human, handoff precision and latency.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = "2026-11-16"  # a Monday, so "tomorrow" and weekday names resolve the same on every run

# (conversation, [(user message, expected outcome prefix)])
SCRIPTS = [
    ["what time do you open on saturday", "faq:hours"],
    ["are you open sunday?", "faq:hours"],
    ["where is your shop", "faq:location"],
    ["how much to replace a cracked screen on samsung", "faq:screen-price"],
    ["iphone screen replacement price?", "faq:screen-price"],
    ["battery replacement cost for oneplus", "faq:battery-price"],
    ["how long will the repair take", "faq:repair-time"],
    ["is there any warranty on the repair", "faq:warranty"],
    ["my phone fell in water, can you fix it?", "faq:water-damage"],
    ["will you look at my photos during repair", "faq:data-privacy"],
    ["do you take UPI payment", "faq:payment"],
    ["is diagnosis free?", "faq:diagnosis-fee"],
    ["do you repair google pixel phones", "faq:brands"],
    ["can you pick up the phone from my home", "faq:pickup"],
    ["do you use original parts or duplicate", "faq:original-parts"],
    ["how can i reschedule my appointment", "faq:cancel-booking"],
    ["what if i forget to collect the phone", "faq:unclaimed"],
    ["do you sell new phones?", "handoff"],
    ["can you unlock an iCloud locked phone", "handoff"],
    ["do you give EMI on laptops", "handoff"],
    ["what is the price of ps5 controller repair", "handoff"],
]
MULTI = [
    [("hi", "answered"), ("is my phone ready?", "asked_followup"), ("QF-1004", "order_status"), ("thanks", "answered")],
    [("status of QF-1010 please", "order_status")],
    [("track my repair", "asked_followup"), ("QF-9999", "asked_followup"), ("QF-8888", "handoff")],
    [("whats the status of my order", "asked_followup"), ("QF-1020", "order_status")],
    [("can i collect my phone? order QF-1031", "order_status")],
    [("I want to book an appointment", "asked_followup"), ("Redmi Note 12", "asked_followup"),
     ("charging port not working", "asked_followup"), ("tomorrow", "asked_followup"), ("4 pm", "booked")],
    [("book a slot for my iPhone 13 screen on wednesday at 11 am", "booked")],
    [("book appointment for Pixel 7 battery", "asked_followup"), ("sunday", "asked_followup"), ("monday", "asked_followup"),
     ("10 am", "booked")],
    [("can i book a visit for my oneplus camera issue friday", "asked_followup"), ("9 pm", "asked_followup"),
     ("12 pm", "booked")],
    [("schedule repair Samsung Galaxy S22 water damage tomorrow 11 am", "booked")],
    [("you guys are useless, my phone is still not fixed", "handoff")],
    [("I want a refund", "handoff")],
    [("can I talk to a real person", "handoff")],
    [("hello", "answered"), ("how much is battery replacement", "faq:battery-price"), ("thank you", "answered")],
    [("is my phone ready", "asked_followup"), ("actually can i book an appointment instead", "asked_followup"),
     ("Vivo V27 speaker issue saturday 2 pm", "booked")],
    [("warranty?", "faq:warranty"), ("and do you accept cards", "faq:payment")],
    [("my number is 98 not sure", "handoff")],  # unclear message: handing off is the right call
    [("where are you located and is there parking", "faq:location")],
    [("my order QF-1002", "order_status")],
]

# Held-out messages written after tuning and never used to change rules or FAQ phrasings.
HELDOUT = [
    ("till what time are you open today", "faq:hours"),
    ("what's the address", "faq:location"),
    ("price for changing the display of a redmi", "faq:screen-price"),
    ("my battery dies fast, how much to change it", "faq:battery-price"),
    ("same day repair possible?", "faq:repair-time"),
    ("what guarantee do i get after repair", "faq:warranty"),
    ("phone got wet in rain", "faq:water-damage"),
    ("can i pay by card", "faq:payment"),
    ("do you fix realme phones", "faq:brands"),
    ("are the spare parts genuine", "faq:original-parts"),
    ("where is QF-1012", "order_status"),
    ("book me in for an iphone 12 battery tomorrow at 5 pm", "booked"),
    ("do you repair smart watches", "handoff"),
    ("this is a scam, give me my money back", "handoff"),
    ("do you buy old phones for cash", "handoff"),
]


def run(verbose=True) -> dict:
    os.environ["FDE6_TODAY"] = TODAY
    from .agent import SupportAgent
    from .report import weekly_report

    agent = SupportAgent()
    convs = [[(m, exp)] for m, exp in SCRIPTS] + MULTI
    turns = ok = 0
    fails = []
    lat = []
    for i, conv in enumerate(convs):
        cid = f"eval-{i}"
        for msg, expected in conv:
            r = agent.reply(cid, msg, channel="eval")
            lat.append(r["latency_ms"])
            turns += 1
            good = r["outcome"].startswith(expected)
            ok += good
            if not good:
                fails.append({"conversation": cid, "message": msg, "expected": expected, "got": r["outcome"]})
    held_ok, held_fail = 0, []
    held_agent = SupportAgent()  # separate store, so held-out turns do not change the main numbers
    for j, (msg, expected) in enumerate(HELDOUT):
        r = held_agent.reply(f"heldout-{j}", msg, channel="eval")
        held_ok += r["outcome"].startswith(expected)
        if not r["outcome"].startswith(expected):
            held_fail.append({"message": msg, "expected": expected, "got": r["outcome"]})
    expected_handoff = {f"eval-{i}" for i, c in enumerate(convs) if any(e == "handoff" for _, e in c)}
    got_handoff = {r[0] for r in agent.store.q("SELECT DISTINCT conversation_id FROM handoffs")
                   if r[0].startswith("eval-")}
    tp = len(expected_handoff & got_handoff)
    rep = weekly_report(agent.store)
    out = {"conversations": len(convs), "turns": turns, "turn_accuracy": round(ok / turns, 4),
           "resolved_without_human_pct": rep["resolved_without_human_pct"],
           "expected_resolvable_pct": round(100 * (len(convs) - len(expected_handoff)) / len(convs), 1),
           "handoff_precision": round(tp / len(got_handoff), 4) if got_handoff else None,
           "handoff_recall": round(tp / len(expected_handoff), 4) if expected_handoff else None,
           "bookings_made": rep["bookings"], "median_latency_ms": sorted(lat)[len(lat) // 2],
           "max_latency_ms": max(lat), "top_unanswered_questions": rep["top_unanswered_questions"],
           "failures": fails,
           "heldout": {"messages": len(HELDOUT), "accuracy": round(held_ok / len(HELDOUT), 4),
                       "wrong_answers": sum(1 for f in held_fail if f["got"] != "handoff"),
                       "safe_handoffs": sum(1 for f in held_fail if f["got"] == "handoff"), "failures": held_fail}}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
