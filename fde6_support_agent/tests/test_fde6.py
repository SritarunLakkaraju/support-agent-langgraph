import importlib

import pytest
from fastapi.testclient import TestClient

from fde6_support_agent.agent import SupportAgent, parse_day, parse_device, parse_hour, parse_issue
from fde6_support_agent.tools import FaqIndex, OrderBook, Store


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("FDE6_TODAY", "2026-11-16")  # a Monday


def test_parsers():
    assert parse_day("tomorrow").isoformat() == "2026-11-17"
    assert parse_day("on friday").isoformat() == "2026-11-20"
    assert parse_hour("at 4 pm") == 16 and parse_hour("11:00") == 11 and parse_hour("12 am") == 0
    assert parse_device("my iPhone 13 broke") == "iPhone 13"
    assert parse_issue("the display is cracked") == "cracked screen"


def test_faq_confidence():
    f = FaqIndex()
    fid, ans, conf = f.search("is there any warranty on the repair")
    assert fid == "warranty" and conf >= 0.5
    assert f.search("do you sell laptops on EMI")[2] < 0.5


def test_order_lookup_by_id_and_phone():
    ob = OrderBook()
    row = next(iter(ob.rows.values()))
    assert ob.lookup(order_id=row["order_id"].lower())["device"] == row["device"]
    assert ob.lookup(phone="+91 " + row["customer_phone"])["order_id"] == row["order_id"]


def test_multi_turn_booking_and_double_booking_protection():
    store = Store()
    a = SupportAgent(store)
    for msg in ("book an appointment", "Pixel 7", "battery", "tomorrow"):
        r = a.reply("c1", msg)
        assert r["outcome"] == "asked_followup"
    assert a.reply("c1", "11 am")["outcome"] == "booked"
    b = SupportAgent(store)
    r = b.reply("c2", "book my iPhone 12 screen tomorrow at 11 am")
    assert r["outcome"] == "asked_followup" and "isn't available" in r["reply"]
    assert b.reply("c2", "12 pm")["outcome"] == "booked"


def test_closed_on_sunday():
    a = SupportAgent()
    r = a.reply("c", "book Pixel 7 battery on sunday")
    assert "closed" in r["reply"]


def test_order_followup_then_handoff_after_two_misses():
    a = SupportAgent()
    assert a.reply("c", "is my phone ready")["outcome"] == "asked_followup"
    assert a.reply("c", "QF-0001")["outcome"] == "asked_followup"
    assert a.reply("c", "QF-0002")["outcome"] == "handoff"
    assert a.store.q("SELECT count(*) FROM handoffs")[0][0] == 1


def test_upset_customer_goes_to_human():
    a = SupportAgent()
    assert a.reply("c", "this is the worst service, I want a refund")["outcome"] == "handoff"


def test_evaluation_bar():
    from fde6_support_agent.evaluate import run

    m = run(verbose=False)
    assert m["turn_accuracy"] >= 0.95 and m["handoff_recall"] == 1.0
    assert m["heldout"]["wrong_answers"] <= 2  # on unseen phrasing the bot should mostly hand off, not guess


def test_channels_api():
    api = importlib.reload(importlib.import_module("fde6_support_agent.api"))
    with TestClient(api.app) as c:
        assert "QuickFix" in c.get("/").text
        r = c.post("/chat", json={"conversation_id": "w1", "message": "what are your opening hours?"})
        assert r.json()["outcome"] == "faq:hours"
        tg = c.post("/telegram/webhook", json={"update_id": 1, "message": {"chat": {"id": 42}, "text": "status QF-1003"}})
        assert "QF-1003" in tg.json()["reply"]
        assert c.post("/telegram/webhook", json={"update_id": 2, "message": {"chat": {"id": 42}}}).json()["ok"]
        v = c.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "quickfix-verify",
                                               "hub.challenge": "123"})
        assert v.text == "123"
        assert c.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "bad",
                                                  "hub.challenge": "1"}).status_code == 403
        wa = c.post("/whatsapp/webhook", json={"entry": [{"changes": [{"value": {
            "metadata": {"phone_number_id": "1"},
            "messages": [{"from": "919800000000", "type": "text", "text": {"body": "do you charge for diagnosis"}}]}}]}]})
        assert "free" in wa.json()["replies"][0].lower()
        assert c.post("/feedback", json={"conversation_id": "w1", "rating": 5}).json()["ok"]
        rep = c.get("/admin/report").json()
        assert rep["conversations"] == 3 and rep["csat_avg_1_to_5"] == 5.0
        assert c.post("/chat", json={"conversation_id": "w1", "message": ""}).status_code == 422
    api.agent.cache_clear()
