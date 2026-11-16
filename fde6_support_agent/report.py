"""The weekly review the FDE does with the shop owner: what the bot handled, what it handed off, and which
questions it could not answer (each one is a candidate new FAQ entry)."""
from __future__ import annotations

from collections import Counter

from .tools import Store


def weekly_report(store: Store) -> dict:
    msgs = store.q("SELECT conversation_id, outcome, latency_ms, user_text FROM messages")
    convs: dict[str, list[str]] = {}
    for cid, outcome, _, _ in msgs:
        convs.setdefault(cid, []).append(outcome)
    resolved = sum(1 for outs in convs.values() if "handoff" not in outs)
    ratings = [r[0] for r in store.q("SELECT rating FROM feedback")]
    unanswered = Counter(r[0].strip().lower() for r in
                         store.q("SELECT last_message FROM handoffs WHERE reason LIKE 'low confidence%'"))
    lat = sorted(m[2] for m in msgs)
    return {
        "conversations": len(convs),
        "messages": len(msgs),
        "resolved_without_human_pct": round(100 * resolved / len(convs), 1) if convs else None,
        "handoffs": store.q("SELECT count(*) FROM handoffs")[0][0],
        "bookings": store.q("SELECT count(*) FROM bookings")[0][0],
        "outcomes": dict(Counter(o.split(":")[0] for _, o, _, _ in msgs)),
        "csat_avg_1_to_5": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "median_response_ms": lat[len(lat) // 2] if lat else None,
        "top_unanswered_questions": unanswered.most_common(5),
    }
