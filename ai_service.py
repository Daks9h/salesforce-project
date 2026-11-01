"""
AI Automation Service for Customer Support and Service Automation
Provides automated ticket classification, priority estimation, suggested responses,
and relevant troubleshooting recommendations.
Supports Google Gemini AI when GOOGLE_API_KEY is configured,
with full graceful rule-based fallback when offline or unconfigured.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Realistic Electronics Products list
SUPPORTED_PRODUCTS = [
    "Wireless Headphones",
    "Smartwatch",
    "Laptop",
    "Bluetooth Speaker",
    "Wireless Earbuds"
]

# Standard Categories
CATEGORIES = [
    "Product Issue",
    "Order Tracking",
    "Warranty Claim",
    "General Inquiry"
]

PRIORITIES = ["Low", "Medium", "High", "Urgent"]

# Rule-based heuristics for classification
CATEGORY_KEYWORD_RULES = {
    "Order Tracking": [
        "order", "track", "tracking", "shipment", "delivery", "shipping", "courier",
        "package", "parcel", "dispatched", "transit", "dhl", "fedex", "ups",
        "where is my", "order id", "not received", "wrong address", "delay"
    ],
    "Warranty Claim": [
        "warranty", "claim", "replacement", "rma", "defective unit", "repair",
        "guarantee", "serial number", "receipt", "invoice", "hardware failure",
        "manufacture defect", "broken on arrival", "dead on arrival", "exchange unit"
    ],
    "Product Issue": [
        "not working", "pair", "pairing", "bluetooth", "audio", "sound", "volume",
        "stutter", "mic", "microphone", "charge", "charging", "battery", "drain",
        "overheat", "overheating", "fan", "screen", "flicker", "display", "black screen",
        "freeze", "frozen", "sync", "disconnect", "reset", "power", "turn on", "button"
    ],
    "General Inquiry": [
        "return", "refund", "hours", "contact", "phone", "email", "support",
        "international", "store", "manual", "spec", "specs", "specification",
        "price", "discount", "trade-in", "policy", "faq"
    ]
}

# Rule-based heuristics for priority
PRIORITY_RULES = {
    "Urgent": [
        "fire", "smoke", "spark", "swelling", "swollen", "exploded", "burning",
        "safety", "shock", "danger", "hazard", "battery expand", "emergency"
    ],
    "High": [
        "completely dead", "won't turn on", "will not power", "black screen",
        "lost package", "delivered but not received", "broken out of box",
        "hardware defect", "defective", "unable to work", "shuts down immediately",
        "no sound at all"
    ],
    "Medium": [
        "disconnect", "audio stutter", "battery drain", "battery low", "slow",
        "overheating", "sync", "delay", "tracking update", "flicker", "charging slow",
        "one side", "static noise"
    ],
    "Low": [
        "manual", "guide", "specs", "specification", "color", "how to use",
        "general question", "feature request", "inquiry", "store location"
    ]
}

# Product-specific troubleshooting recommendations
PRODUCT_TROUBLESHOOTING = {
    "Wireless Headphones": [
        "Ensure headphones are charged for at least 30 minutes with a standard 5V adapter.",
        "Perform a Bluetooth reset: Hold the Power/Pairing button for 7 seconds until the LED flashes red/blue.",
        "Remove the headphones from phone/laptop paired list and reconnect.",
        "Verify no high-frequency wireless interference (microwaves, routers) is nearby."
    ],
    "Smartwatch": [
        "Check that phone Bluetooth and GPS Location services are toggled ON.",
        "Open the ApexFit companion app, pull down to sync, and check connection status.",
        "Restart both the smartwatch and smartphone to clear cached BLE state.",
        "Lower screen brightness and disable Always-On Display (AOD) to extend battery life."
    ],
    "Laptop": [
        "Ensure air vents are unobstructed and place the device on a flat, solid surface.",
        "Open Task Manager (Ctrl + Shift + Esc) to check high CPU background tasks.",
        "Update graphics and BIOS firmware from the ApexTech Download Center.",
        "Test external display via HDMI/USB-C if the built-in screen flickers."
    ],
    "Bluetooth Speaker": [
        "Connect using the original USB-C cable to a 5V/2A power source for 30 minutes.",
        "Press the pinhole hardware reset button adjacent to the charging port for 5 seconds.",
        "Disconnect any multi-point paired secondary phones or tablets.",
        "Keep speaker within 10 meters (30 feet) line-of-sight of audio source."
    ],
    "Wireless Earbuds": [
        "Clean golden charging pins on both earbuds and inside the charging case with a dry swab.",
        "Place earbuds in case and hold the rear function button for 10-15s until LED flashes amber 3 times.",
        "Forget device from Bluetooth settings and re-pair both buds simultaneously.",
        "Ensure the charging case has sufficient remaining battery power."
    ]
}

def detect_product(text: str) -> Optional[str]:
    """Detect product mention in text."""
    lower = text.lower()
    if any(k in lower for k in ["headphone", "headset", "over-ear"]):
        return "Wireless Headphones"
    if any(k in lower for k in ["smartwatch", "watch", "fitness tracker", "band"]):
        return "Smartwatch"
    if any(k in lower for k in ["laptop", "notebook", "pc", "computer"]):
        return "Laptop"
    if any(k in lower for k in ["speaker", "boombox", "soundbox"]):
        return "Bluetooth Speaker"
    if any(k in lower for k in ["earbud", "earbuds", "earphone", "earphones", "airpod", "buds"]):
        return "Wireless Earbuds"
    return None

def rule_based_classify(text: str, product_name: Optional[str] = None) -> Dict[str, Any]:
    """Perform rule-based classification and analysis when LLM is unavailable."""
    lower = text.lower()

    # 1. Determine Category
    category_scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in CATEGORY_KEYWORD_RULES.items():
        for kw in keywords:
            if kw in lower:
                category_scores[cat] += 2 if len(kw.split()) > 1 else 1

    detected_category = max(category_scores, key=category_scores.get)
    if category_scores[detected_category] == 0:
        detected_category = "General Inquiry"

    # 2. Determine Priority
    detected_priority = "Normal"
    for priority in ["Urgent", "High", "Medium", "Low"]:
        if any(kw in lower for kw in PRIORITY_RULES[priority]):
            detected_priority = priority
            break
    if detected_priority == "Normal":
        detected_priority = "High" if detected_category in ["Warranty Claim", "Product Issue"] else "Medium"

    # 3. Product context
    prod = product_name or detect_product(text) or "Electronics Device"

    # 4. Generate Recommended Troubleshooting
    troubleshooting_steps = PRODUCT_TROUBLESHOOTING.get(prod, [
        "Verify device has sufficient battery charge or is connected to a certified power outlet.",
        "Perform a full device power cycle (power off, wait 30 seconds, power on).",
        "Check manufacturer documentation for hard-reset key combinations.",
        "Verify device firmware is updated to the latest stable release."
    ])

    # 5. Generate Professional Suggested Response
    if detected_category == "Order Tracking":
        suggested_response = (
            f"Dear Customer,\n\n"
            f"Thank you for contacting ApexTech Customer Care regarding your order. "
            f"We have located your inquiry and our logistics team is verifying the latest courier status. "
            f"Standard transit updates typically refresh within 24 hours. "
            f"If your tracking indicates any abnormal delay, we will expedite an investigation with the carrier immediately.\n\n"
            f"Best regards,\nApexTech Support Team"
        )
    elif detected_category == "Warranty Claim":
        suggested_response = (
            f"Dear Customer,\n\n"
            f"Thank you for reaching out to ApexTech Warranty Services. "
            f"We are sorry to hear about the issue with your {prod}. "
            f"All ApexTech devices are backed by our 1-Year Limited Manufacturer Warranty. "
            f"Our technical warranty team is reviewing your claim details and will provide next steps, "
            f"including a prepaid RMA shipping label if repair or replacement is warranted.\n\n"
            f"Warm regards,\nApexTech Warranty Department"
        )
    elif detected_category == "Product Issue":
        suggested_response = (
            f"Dear Customer,\n\n"
            f"Thank you for reaching out. We apologize for the inconvenience you are experiencing with your {prod}. "
            f"To assist in resolving this promptly, our technical team recommends the following initial troubleshooting steps:\n"
            + "\n".join([f"- {s}" for s in troubleshooting_steps[:3]]) + "\n\n"
            f"If these steps do not remedy the problem, an assigned technician will follow up to assist you further.\n\n"
            f"Sincerely,\nApexTech Technical Support"
        )
    else:
        suggested_response = (
            f"Dear Customer,\n\n"
            f"Thank you for contacting ApexTech Customer Support. "
            f"We have received your inquiry and our support representatives will respond with full details shortly. "
            f"For immediate help, our online Knowledge Base and 24/7 hotline are also available.\n\n"
            f"Best regards,\nApexTech Customer Support"
        )

    return {
        "category": detected_category,
        "priority": detected_priority,
        "suggested_response": suggested_response,
        "suggested_troubleshooting": "\n".join([f"{i+1}. {s}" for i, s in enumerate(troubleshooting_steps)]),
        "detected_product": prod,
        "method": "Rule-Based Intelligence"
    }

def analyze_ticket_with_gemini(text: str, product_name: Optional[str] = None, api_key: str = "") -> Optional[Dict[str, Any]]:
    """Use Google Gemini API to classify ticket and generate suggestions."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an AI Support Engineer for ApexTech Electronics & Gadgets Global.
Analyze the following customer ticket:
Customer Complaint: "{text}"
Product Mentioned: "{product_name or 'Unknown'}"

Respond in STRICT JSON with the following keys:
- "category": One of ["Product Issue", "Order Tracking", "Warranty Claim", "General Inquiry"]
- "priority": One of ["Low", "Medium", "High", "Urgent"]
- "suggested_response": A professional, empathetic customer service reply.
- "suggested_troubleshooting": 3-4 numbered technical troubleshooting steps tailored for electronics.
"""
        response = model.generate_content(prompt)
        content = response.text.strip()
        # Clean potential markdown wrapping
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        data["method"] = "Gemini AI Automation"
        return data
    except Exception as e:
        logger.warning(f"Gemini AI ticket analysis failed ({e}). Falling back to rule-based analysis.")
        return None

def analyze_ticket(text: str, product_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for AI ticket automation.
    Tries Google Gemini if API key is set, otherwise gracefully falls back to rule-based engine.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        result = analyze_ticket_with_gemini(text, product_name=product_name, api_key=api_key)
        if result and "category" in result:
            return result

    # Rule-based fallback
    return rule_based_classify(text, product_name=product_name)
