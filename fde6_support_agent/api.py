"""FDE-6 channels (FastAPI on Cloud Run). Locally: uvicorn fde6_support_agent.api:app --port 8006

GET  /                     website chat widget (HTML/JS)
POST /chat                 {"conversation_id", "message"} -> reply (the widget calls this)
POST /telegram/webhook     Telegram Bot API update -> replies via sendMessage when TELEGRAM_BOT_TOKEN is set
GET  /whatsapp/webhook     Meta webhook verification (hub.challenge, WHATSAPP_VERIFY_TOKEN)
POST /whatsapp/webhook     WhatsApp Cloud API messages -> replies via Graph API when WHATSAPP_TOKEN is set
POST /feedback             {"conversation_id", "rating": 1-5}
GET  /admin/report         weekly review numbers for the owner (conversations, % resolved, CSAT, unanswered)
GET  /admin/handoffs       open handoffs a human must answer
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .agent import SupportAgent
from .report import weekly_report
from .tools import Store

app = FastAPI(title="QuickFix Support Agent (FDE-6)", version="1.0")


@lru_cache(maxsize=1)
def agent() -> SupportAgent:
    path = os.environ.get("FDE6_DB", ":memory:")
    cp = None
    if path != ":memory:":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        cp = SqliteSaver(sqlite3.connect(path + ".ckpt", check_same_thread=False))
    return SupportAgent(Store(path), checkpointer=cp)


class ChatIn(BaseModel):
    conversation_id: str = Field(pattern=r"^[\w\-]{1,64}$")
    message: str = Field(min_length=1, max_length=1000)


class FeedbackIn(BaseModel):
    conversation_id: str
    rating: int = Field(ge=1, le=5)


@app.post("/chat")
def chat(m: ChatIn):
    return agent().reply(m.conversation_id, m.message, channel="web")


@app.post("/telegram/webhook")
async def telegram(req: Request):
    upd = await req.json()
    msg = upd.get("message") or upd.get("edited_message")
    if not msg or "text" not in msg:
        return {"ok": True}  # stickers, joins etc.: acknowledge and ignore
    chat_id = msg["chat"]["id"]
    out = agent().reply(f"tg-{chat_id}", msg["text"], channel="telegram")
    if token := os.environ.get("TELEGRAM_BOT_TOKEN"):
        import httpx

        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": chat_id, "text": out["reply"]}, timeout=10)
    return {"ok": True, "reply": out["reply"]}


@app.get("/whatsapp/webhook", response_class=PlainTextResponse)
def whatsapp_verify(mode: str = Query(alias="hub.mode"), token: str = Query(alias="hub.verify_token"),
                    challenge: str = Query(alias="hub.challenge")):
    if mode == "subscribe" and token == os.environ.get("WHATSAPP_VERIFY_TOKEN", "quickfix-verify"):
        return challenge
    raise HTTPException(403, "verification failed")


@app.post("/whatsapp/webhook")
async def whatsapp(req: Request):
    body = await req.json()
    replies = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for m in value.get("messages", []):
                if m.get("type") != "text":
                    continue
                out = agent().reply(f"wa-{m['from']}", m["text"]["body"], channel="whatsapp")
                replies.append(out["reply"])
                if token := os.environ.get("WHATSAPP_TOKEN"):
                    import httpx

                    httpx.post(f"https://graph.facebook.com/v20.0/{value['metadata']['phone_number_id']}/messages",
                               headers={"Authorization": f"Bearer {token}"}, timeout=10,
                               json={"messaging_product": "whatsapp", "to": m["from"], "type": "text",
                                     "text": {"body": out["reply"]}})
    return {"ok": True, "replies": replies}


@app.post("/feedback")
def feedback(f: FeedbackIn):
    agent().store.q("INSERT INTO feedback (conversation_id, rating) VALUES (?,?)", (f.conversation_id, f.rating))
    return {"ok": True}


@app.get("/admin/report")
def report():
    return weekly_report(agent().store)


@app.get("/admin/handoffs")
def handoffs():
    rows = agent().store.q("SELECT id, conversation_id, reason, last_message, created_at FROM handoffs "
                           "WHERE status='open' ORDER BY id")
    return [dict(zip(("id", "conversation_id", "reason", "last_message", "created_at"), r)) for r in rows]


WIDGET = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuickFix chat</title><style>
body{font-family:system-ui,sans-serif;background:#f1f5f9;margin:0;display:flex;justify-content:center}
.box{width:100%;max-width:420px;height:100vh;display:flex;flex-direction:column;background:#fff}
header{background:#0f766e;color:#fff;padding:14px 16px;font-weight:600}
#log{flex:1;overflow:auto;padding:12px}.m{margin:6px 0;padding:9px 12px;border-radius:12px;max-width:85%;line-height:1.35}
.u{background:#0f766e;color:#fff;margin-left:auto}.b{background:#e2e8f0}
form{display:flex;gap:6px;padding:10px;border-top:1px solid #e2e8f0}input{flex:1;padding:10px;border:1px solid #cbd5e1;border-radius:8px;font-size:15px}
button{background:#0f766e;color:#fff;border:0;border-radius:8px;padding:0 14px}.rate{font-size:12px;color:#64748b;padding:0 12px 8px}
.rate span{cursor:pointer;margin-right:6px}</style></head><body><div class="box"><header>QuickFix Mobile Repairs</header>
<div id="log"><div class="m b">Hi! Ask me about repairs and prices, check a repair (QF-1001...) or book a visit.</div></div>
<div class="rate">Rate this chat: <span onclick="rate(5)">&#128077;</span><span onclick="rate(1)">&#128078;</span></div>
<form onsubmit="send(event)"><input id="t" autocomplete="off" placeholder="Type a message"><button>Send</button></form></div>
<script>const cid='web-'+Math.random().toString(36).slice(2,10);
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;const l=document.getElementById('log');l.appendChild(d);l.scrollTop=1e9}
async function send(e){e.preventDefault();const i=document.getElementById('t');const t=i.value.trim();if(!t)return;i.value='';add(t,'u');
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:cid,message:t})});
const j=await r.json();add(j.reply||'Sorry, something went wrong.','b')}
async function rate(n){await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:cid,rating:n})});
document.querySelector('.rate').textContent='Thanks for the feedback!'}</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def widget():
    return WIDGET
