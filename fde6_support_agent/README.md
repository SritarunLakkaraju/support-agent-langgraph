# FDE-6 · Customer Support Agent for a Small Business

A support bot for a fictional phone-repair shop. It answers FAQs (RAG), checks repair status from the shop's
order sheet, and books appointments over several turns (slot filling, with no double booking). It hands off to a
human when unsure or when the customer is upset. It is built on LangGraph with per-conversation memory and serves
a web widget plus Telegram and WhatsApp webhooks. A weekly report gives the % resolved without a human, CSAT and
the unanswered questions.

```bash
python -m fde6_support_agent.evaluate            # 40 scripted conversations + 15 held-out messages
uvicorn fde6_support_agent.api:app --port 8006   # chat widget at http://localhost:8006
```
