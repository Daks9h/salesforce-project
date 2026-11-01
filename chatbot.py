#!/usr/bin/env python3
"""
AI-Powered Customer Support Chatbot - Electronics & Gadgets Global
Hybrid search: Keyword Matching -> RAG Vector Search -> Generative AI
With conversation memory and intelligent ticket escalation suggestions.
"""

import json
import re
import sys
from pathlib import Path
from rag_engine import RAGEngine
import os
import time
import urllib.request
import urllib.error
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
KB_FILE = Path(__file__).parent / "knowledge_base.json"

USE_LLM = os.environ.get("USE_LLM", "true").lower() == "true"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY") or ""

# External LLM Config (Optional)
LLM_API_URL = os.getenv("LLM_API_URL", "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.5")

# Initialize RAG Engine
rag_engine = RAGEngine(api_key=GOOGLE_API_KEY)

# ============================================================
# CONVERSATION MEMORY (per session, with TTL)
# ============================================================
conversation_history = defaultdict(list)  # session_id -> [(role, content)]
_session_timestamps = {}  # session_id -> last_active_timestamp
MAX_HISTORY = 5
SESSION_TTL = 1800  # 30 minutes in seconds

def _cleanup_expired_sessions():
    """Remove sessions that haven't been active within SESSION_TTL."""
    now = time.time()
    expired = [sid for sid, ts in _session_timestamps.items() if now - ts > SESSION_TTL]
    for sid in expired:
        conversation_history.pop(sid, None)
        _session_timestamps.pop(sid, None)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired conversation sessions")

def add_to_history(session_id, role, content):
    """Add message to conversation history with TTL tracking."""
    _cleanup_expired_sessions()
    _session_timestamps[session_id] = time.time()
    history = conversation_history[session_id]
    history.append((role, content))
    if len(history) > MAX_HISTORY * 2:
        conversation_history[session_id] = history[-MAX_HISTORY * 2:]

def get_history_context(session_id):
    """Get recent conversation context as string."""
    history = conversation_history.get(session_id, [])
    if not history:
        return ""
    lines = [f"{role}: {msg}" for role, msg in history[-MAX_HISTORY:]]
    return "\n".join(lines)

# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================
def load_kb():
    """Load knowledge base from JSON file."""
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Knowledge base file not found: {KB_FILE}")
        return {
            "categories": [],
            "escalation": {
                "message": "I could not find an answer in our knowledge base. Would you like to create a support ticket?",
                "hotline": "1-800-APEXTECH"
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in knowledge base: {e}")
        return {
            "categories": [],
            "escalation": {
                "message": "I could not find an answer in our knowledge base. Would you like to create a support ticket?",
                "hotline": "1-800-APEXTECH"
            }
        }

# ============================================================
# TOKENIZATION & KEYWORD MATCHING
# ============================================================
STOP_WORDS = set([
    # English
    "the", "is", "a", "an", "to", "and", "or", "of", "in", "for",
    "how", "what", "why", "can", "do", "does", "i", "my", "your", "it",
    "me", "you", "we", "they", "this", "that", "there", "with", "on", "at",
    "please", "help", "need", "want", "tell",
    # Thai
    "ครับ", "ค่ะ", "คะ", "นะ", "มั้ย", "ไหม", "ได้", "จะ", "ให้",
    "ที่", "ของ", "ใน", "เป็น", "มี", "ไป", "มา", "ทำ", "แล้ว",
    "และ", "หรือ", "แต่", "ไม่", "ก็", "ถ้า", "จาก", "กับ",
])

# For backward compatibility with existing tests
THAI_STOP_WORDS = STOP_WORDS

SYNONYM_MAP = {
    # Electronics & Gadgets
    "headphones": ["headset", "earphones", "earbuds", "audio", "listening", "can"],
    "earbuds": ["earbud", "buds", "earphones", "headphones", "airpods"],
    "smartwatch": ["watch", "fitness tracker", "band", "wearable", "clock"],
    "laptop": ["notebook", "computer", "pc", "macbook"],
    "speaker": ["bluetooth speaker", "audio", "soundbox", "boombox"],
    "bluetooth": ["bt", "pair", "pairing", "connect", "connection", "wireless"],
    "pairing": ["pair", "bluetooth", "connect", "sync", "link"],
    "battery": ["charge", "charging", "power", "dead", "drain", "draining"],
    "screen": ["display", "monitor", "flicker", "blank", "black"],
    "overheating": ["hot", "fan", "heat", "thermal", "warm"],
    "warranty": ["claim", "replacement", "repair", "defect", "rma", "guarantee"],
    "order": ["tracking", "shipment", "delivery", "shipping", "courier", "package"],
    "tracking": ["track", "order", "status", "delivery", "courier", "parcel"],
    # Retained Thai & general synonyms
    "wifi": ["ไวไฟ", "wi-fi", "วายฟาย", "อินเทอร์เน็ต", "internet", "เน็ต"],
    "ไวไฟ": ["wifi", "wi-fi", "internet"],
    "printer": ["ปริ้นเตอร์", "ปริ้นท์", "พิมพ์", "print"],
    "ปริ้นเตอร์": ["printer", "print"],
    "password": ["รหัสผ่าน", "พาสเวิร์ด", "pass", "pwd"],
    "รหัสผ่าน": ["password", "pass", "pwd"],
    "email": ["อีเมล", "เมล", "mail", "gmail", "outlook"],
    "อีเมล": ["email", "mail"],
    "error": ["ข้อผิดพลาด", "เออเร่อ", "พัง", "เสีย", "ใช้ไม่ได้"],
    "slow": ["ช้า", "อืด", "lag", "หน่วง"],
}

def tokenize_thai(text):
    """Tokenize query string into normalized lowercase tokens (English & multilingual)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\sก-๙]', ' ', text)
    tokens = []
    for word in text.split():
        if word and word not in STOP_WORDS:
            tokens.append(word)
    return tokens

def expand_with_synonyms(tokens):
    """Expand token set with synonyms for broader matching."""
    expanded = set(tokens)
    for token in tokens:
        if token in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[token])
    return expanded

def substring_match(user_input, target_text):
    """Check if any user token appears as a substring in target."""
    user_tokens = tokenize_thai(user_input)
    target_lower = target_text.lower()
    matches = 0
    for token in user_tokens:
        if len(token) >= 2 and token in target_lower:
            matches += 1
    return matches

def find_relevant_faqs(user_input, kb):
    """Find FAQs matching user input using keyword scoring."""
    user_tokens = tokenize_thai(user_input)
    user_expanded = expand_with_synonyms(user_tokens)
    user_set = set(user_tokens)
    scores = []

    for category in kb.get("categories", []):
        cat_keywords = set(kw.lower() for kw in category.get("keywords", []))

        for faq in category.get("faqs", []):
            score = 0

            # 1. Category keyword match
            for kw in cat_keywords:
                if kw in user_expanded:
                    score += 5
                for token in user_tokens:
                    if len(token) >= 2 and (kw in token or token in kw):
                        score += 3

            # 2. Exact word overlap with question
            q_tokens = set(tokenize_thai(faq["question"]))
            overlap = user_set & q_tokens
            score += len(overlap) * 4

            # 3. Expanded synonym overlap with question
            q_expanded = expand_with_synonyms(q_tokens)
            synonym_overlap = user_expanded & q_expanded
            score += len(synonym_overlap) * 2

            # 4. Substring match in question
            sub_matches = substring_match(user_input, faq["question"])
            score += sub_matches * 2

            # 5. Substring match in answer
            sub_answer = substring_match(user_input, faq["answer"])
            score += sub_answer * 1

            if score > 0:
                scores.append((score, faq, category["name"]))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:3]

# ============================================================
# LLM RESPONSE GENERATION
# ============================================================
def llm_answer(user_input, context_faqs, session_id="default"):
    """Generate concise, natural response using Gemini or configured LLM."""
    if not USE_LLM:
        return None

    context = "\n\n".join([
        f"FAQ: {faq.get('question', '')}\nAnswer: {faq.get('answer', '')}"
        for _, faq, _ in context_faqs
    ])
    history = get_history_context(session_id)

    # 1. Try Google Gemini first if key available
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""You are an expert Customer Support AI Assistant for ApexTech Electronics & Gadgets Global.
Provide a clear, helpful, and concise answer to the customer's question.
Rely primarily on the Knowledge Base context below.
If the solution is not clearly covered in the FAQs, politely recommend that the customer create a Support Ticket on this portal.

{f"Conversation Context:\n{history}\n" if history else ""}
Knowledge Base Context:
{context}

Customer Question: {user_input}

Helpful Answer:"""
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            logger.warning(f"Gemini call failed ({e}). Trying fallback endpoint.")

    # 2. Fallback to generic HTTP completion endpoint if configured
    if GOOGLE_API_KEY and LLM_API_URL:
        try:
            prompt = f"You are ApexTech Customer Support AI.\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            data = json.dumps({
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3
            }).encode("utf-8")

            req = urllib.request.Request(
                LLM_API_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GOOGLE_API_KEY}"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"External LLM endpoint failed: {e}")

    return None

# ============================================================
# CHATBOT RESPONSE PIPELINE
# ============================================================
def get_response(user_input, kb, session_id="default"):
    """
    Main chatbot inquiry resolver.
    Pipeline:
    1. Keyword scoring on Knowledge Base
    2. RAG vector similarity search
    3. Generative AI formulation when LLM is available
    4. Proactive support ticket escalation if unresolved
    """
    add_to_history(session_id, "user", user_input)

    # Escalation message with ticket recommendation
    escalation_text = (
        kb.get("escalation", {}).get(
            "message",
            "I could not find an exact resolution for your inquiry. Would you like to create a support ticket with our customer care team? You can use the 'Raise a Ticket' form below."
        )
    )

    # 1. Keyword Matching
    exact_matches = find_relevant_faqs(user_input, kb)
    if exact_matches and exact_matches[0][0] >= 6:
        best = exact_matches[0][1]
        response = f"**{best['question']}**\n\n{best['answer']}"
        add_to_history(session_id, "bot", response)
        return response, "Keyword"

    # 2. RAG Vector Search
    try:
        matches = rag_engine.query(user_input, n_results=3)
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        matches = []

    if not matches:
        if exact_matches and exact_matches[0][0] >= 3:
            best = exact_matches[0][1]
            response = f"You might find this helpful:\n\n**{best['question']}**\n\n{best['answer']}\n\n*If this does not resolve your problem, feel free to raise a support ticket below.*"
            add_to_history(session_id, "bot", response)
            return response, "Keyword (Suggested)"

        add_to_history(session_id, "bot", escalation_text)
        return escalation_text, "Escalated"

    best_match = matches[0]
    best_score = best_match.get("score", 0)

    keyword_bonus = 0
    if exact_matches:
        keyword_bonus = exact_matches[0][0] * 0.05
    combined_score = best_score + keyword_bonus

    # High confidence -> try LLM or return structured FAQ
    if combined_score >= 0.5:
        context_faqs = [(m.get("score", 0), m, m.get("category", "")) for m in matches]
        llm_resp = llm_answer(user_input, context_faqs, session_id)
        if llm_resp:
            add_to_history(session_id, "bot", llm_resp)
            return llm_resp, "Gemini AI"

        response = f"**{best_match['question']}**\n\n{best_match['answer']}"
        add_to_history(session_id, "bot", response)
        return response, "RAG"

    # Medium confidence -> suggest with escalation option
    if combined_score >= 0.25:
        response = (
            f"Here is the most relevant information for your query:\n\n"
            f"**{best_match['question']}**\n\n{best_match['answer']}\n\n"
            f"---\n*Need more help? You can submit a support ticket using the form below.*"
        )
        add_to_history(session_id, "bot", response)
        return response, "RAG (Suggested)"

    # Low confidence -> escalate
    add_to_history(session_id, "bot", escalation_text)
    return escalation_text, "Escalated"

# ============================================================
# CLI INTERACTION
# ============================================================
def main():
    kb = load_kb()
    print("=" * 60)
    print("  ApexTech - AI-Powered Customer Support & Service Automation")
    print("=" * 60)
    print("  Ask about Products, Order Tracking, Warranty, or General Support")
    print("  Type 'quit' or 'exit' to leave | Type 'help' for categories")
    print("=" * 60 + "\n")

    session_id = "terminal_session"
    while True:
        try:
            user_input = input("Customer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Thank you for choosing ApexTech. Goodbye!")
            break

        if user_input.lower() in ["help", "categories"]:
            print("\nSupport Categories:")
            for cat in kb.get("categories", []):
                print(f"  - {cat['name']} ({len(cat.get('faqs', []))} FAQs)")
            print(f"\nContact Support Hotline: {kb.get('escalation', {}).get('hotline', '1-800-APEXTECH')}\n")
            continue

        response, engine = get_response(user_input, kb, session_id)
        print(f"\nAssistant [{engine}]:\n{response}\n")

if __name__ == "__main__":
    main()
