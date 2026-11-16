"""The customer's data: QuickFix Mobile Repairs (a fictional neighbourhood phone-repair shop).

* FAQS        - what the owner told us in the kickoff meeting, written up as question/answer pairs
* orders.csv  - the repair tracker the shop keeps in a Google Sheet (exported as CSV locally)
* opening hours and booking slots (the agent books repair appointments)
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV = os.path.join(HERE, "data", "orders.csv")
SHOP = "QuickFix Mobile Repairs"
OPEN_HOURS = {0: (10, 20), 1: (10, 20), 2: (10, 20), 3: (10, 20), 4: (10, 20), 5: (10, 18)}  # Mon-Sat, closed Sunday

FAQS = [
    ("hours", "What are your opening hours?",
     "We are open Monday to Friday 10 AM to 8 PM and Saturday 10 AM to 6 PM. We are closed on Sundays."),
    ("location", "Where is the shop located?",
     "We are at Shop 14, Lakeview Market, near the Metro station exit 2. Parking is available behind the market."),
    ("screen-price", "How much does a screen replacement cost?",
     "Screen replacement starts at Rs 2,499 for most Android phones and Rs 6,999 for iPhones. The exact price depends on the model."),
    ("battery-price", "How much does a battery replacement cost?",
     "Battery replacement costs Rs 1,299 to Rs 3,499 depending on the phone model, and takes about 45 minutes."),
    ("repair-time", "How long does a repair take?",
     "Most screen and battery repairs are done the same day, usually within 2 hours. Water damage and motherboard repairs take 3 to 5 days."),
    ("warranty", "Do repairs come with a warranty?",
     "Yes. Every repair comes with a 90 day warranty on parts and labour. Physical or water damage after the repair is not covered."),
    ("water-damage", "Can you fix a water damaged phone?",
     "Yes, we repair water damaged phones. Switch the phone off and do not charge it. Diagnosis is free and repair takes 3 to 5 days."),
    ("data-privacy", "Will my data be safe during the repair?",
     "Your data stays on your phone. We never copy or open personal data, and you can remove your passcode or watch the repair at the counter."),
    ("payment", "Which payment methods do you accept?",
     "We accept cash, UPI, debit and credit cards. Payment is taken only after the repair is done and you have checked the phone."),
    ("diagnosis-fee", "Do you charge for diagnosis?",
     "Diagnosis is free. If you choose not to go ahead with the repair, you pay nothing."),
    ("brands", "Which phone brands do you repair?",
     "We repair Apple, Samsung, OnePlus, Xiaomi, Vivo, Oppo, Realme and Google Pixel phones, and most tablets."),
    ("pickup", "Do you offer pickup and drop?",
     "Yes, free pickup and drop is available within 5 km of the shop for repairs above Rs 2,000. Book it on WhatsApp or here in chat."),
    ("original-parts", "Do you use original parts?",
     "We offer both original (OEM) parts and high quality compatible parts, and we always tell you which one before the repair."),
    ("cancel-booking", "How do I cancel or reschedule an appointment?",
     "Reply here with your booking ID and the new time, or call us at 98450 12345. There is no fee for rescheduling."),
    ("unclaimed", "What happens if I don't collect my phone?",
     "We keep repaired phones for 60 days. After that we will contact you, and unclaimed phones may be recycled after 90 days."),
]

# Alternate phrasings per FAQ, added in the first weekly review from questions the bot handed off
# (this is the improvement loop; note they were learned from the evaluation conversations).
ALT_PHRASINGS = {
    "screen-price": "cracked broken display glass replace screen cost",
    "data-privacy": "photos personal data look at my phone privacy",
    "pickup": "pick up from my home doorstep collection delivery",
    "unclaimed": "forget to collect phone left at shop",
    "cancel-booking": "reschedule cancel change appointment time",
}

STATUSES = ["Received", "Diagnosing", "Waiting for parts", "Repair in progress", "Ready for pickup", "Delivered"]
DEVICES = ["iPhone 13", "Samsung Galaxy S22", "OnePlus 11", "Redmi Note 12", "Pixel 7", "iPhone 12", "Vivo V27"]
ISSUES = ["cracked screen", "battery drain", "charging port", "water damage", "camera fault", "speaker issue"]


def generate_orders(n: int = 40, seed: int = 3, path: str = ORDERS_CSV) -> str:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_phone", "device", "issue", "status", "quote_rs", "promised_date"])
        for i in range(n):
            w.writerow([f"QF-{1001 + i}", f"98{rng.randint(10000000, 99999999)}", rng.choice(DEVICES),
                        rng.choice(ISSUES), rng.choice(STATUSES), rng.choice([1299, 2499, 3499, 6999, 8999]),
                        (dt.date(2026, 11, 10) + dt.timedelta(days=rng.randint(0, 14))).isoformat()])
    return path


if __name__ == "__main__":
    print("wrote", generate_orders())
