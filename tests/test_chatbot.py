#!/usr/bin/env python3
"""
Unit Tests for AI Customer Support Chatbot - Electronics & Gadgets
==================================================================
Tests all core functions of chatbot.py:
- load_kb, tokenize_thai, expand_with_synonyms
- find_relevant_faqs, get_response, substring_match
- Conversation history helpers
- Edge cases: empty input, very long input, special characters

Run: pytest tests/ -v
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Mock RAGEngine before importing chatbot
_mock_rag_engine = MagicMock()
_mock_rag_engine.query.return_value = []

_mock_rag_cls = MagicMock(return_value=_mock_rag_engine)
sys.modules["rag_engine"] = MagicMock(RAGEngine=_mock_rag_cls)

from chatbot import (
    load_kb,
    tokenize_thai,
    expand_with_synonyms,
    find_relevant_faqs,
    get_response,
    substring_match,
    add_to_history,
    get_history_context,
    conversation_history,
    STOP_WORDS,
    SYNONYM_MAP,
    KB_FILE,
)


@pytest.fixture(scope="session")
def kb():
    """Load actual knowledge_base.json for tests."""
    kb_path = Path(__file__).resolve().parent.parent / "knowledge_base.json"
    assert kb_path.exists(), f"knowledge_base.json not found at {kb_path}"
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(autouse=True)
def _reset_rag_mock(monkeypatch):
    """Reset RAG mock and conversation history before each test."""
    import chatbot
    monkeypatch.setattr(chatbot.rag_engine, "query", _mock_rag_engine.query)
    _mock_rag_engine.query.reset_mock()
    _mock_rag_engine.query.return_value = []
    conversation_history.clear()
    yield


# ============================================================
# 1. load_kb()
# ============================================================

class TestLoadKB:
    """Test loading knowledge base."""

    def test_load_kb_returns_dict_with_categories_and_escalation(self):
        data = load_kb()
        assert isinstance(data, dict)
        assert "categories" in data
        assert "escalation" in data

    def test_load_kb_categories_is_list(self):
        data = load_kb()
        assert isinstance(data["categories"], list)
        assert len(data["categories"]) >= 4

    def test_load_kb_escalation_has_required_keys(self):
        data = load_kb()
        esc = data["escalation"]
        assert "message" in esc
        assert "hotline" in esc

    def test_load_kb_each_category_has_name_keywords_faqs(self, kb):
        for cat in kb["categories"]:
            assert "name" in cat, f"Missing 'name' in category: {cat}"
            assert "keywords" in cat, f"Missing 'keywords' in category: {cat.get('name')}"
            assert "faqs" in cat, f"Missing 'faqs' in category: {cat.get('name')}"

    def test_load_kb_fallback_on_missing_file(self, tmp_path):
        with patch("chatbot.KB_FILE", tmp_path / "nonexistent.json"):
            data = load_kb()
        assert data["categories"] == []
        assert "message" in data["escalation"]


# ============================================================
# 2. tokenize_thai()
# ============================================================

class TestTokenizeThai:
    """Test text tokenization and stop word stripping."""

    def test_basic_tokenization(self):
        tokens = tokenize_thai("wireless headphones pairing issue")
        assert "wireless" in tokens
        assert "headphones" in tokens
        assert "pairing" in tokens

    def test_stop_words_removed(self):
        tokens = tokenize_thai("how do i connect my smartwatch to the phone")
        for sw in ["how", "do", "i", "to", "the"]:
            assert sw not in tokens
        assert "smartwatch" in tokens
        assert "phone" in tokens

    def test_lowercased(self):
        tokens = tokenize_thai("BLUETOOTH Battery DRAIN")
        for t in tokens:
            assert t == t.lower()

    def test_empty_input_returns_empty_list(self):
        assert tokenize_thai("") == []
        assert tokenize_thai("   ") == []

    def test_punctuation_stripped(self):
        tokens = tokenize_thai("earbuds? broken! why, what?")
        assert all("?" not in t and "!" not in t and "," not in t for t in tokens)


# ============================================================
# 3. expand_with_synonyms()
# ============================================================

class TestExpandWithSynonyms:
    """Test synonym expansion."""

    def test_headphones_expands_to_audio_synonyms(self):
        expanded = expand_with_synonyms(["headphones"])
        assert "headset" in expanded or "earphones" in expanded

    def test_battery_expands_to_charging(self):
        expanded = expand_with_synonyms(["battery"])
        assert "charge" in expanded or "charging" in expanded

    def test_unknown_token_no_expansion(self):
        expanded = expand_with_synonyms(["xyzrandom123"])
        assert expanded == {"xyzrandom123"}

    def test_multiple_tokens_expand(self):
        expanded = expand_with_synonyms(["headphones", "warranty"])
        assert "claim" in expanded or "replacement" in expanded


# ============================================================
# 4. substring_match()
# ============================================================

class TestSubstringMatch:
    """Test substring matching."""

    def test_matching_substring_found(self):
        count = substring_match("headphones bluetooth", "How to pair Wireless Headphones via Bluetooth")
        assert count >= 1

    def test_no_match_returns_zero(self):
        count = substring_match("xyznotexist", "How to pair Wireless Headphones")
        assert count == 0

    def test_short_token_ignored(self):
        count = substring_match("a b c", "a b c d e")
        assert count == 0


# ============================================================
# 5. find_relevant_faqs()
# ============================================================

class TestFindRelevantFaqs:
    """Test Electronics & Gadgets FAQ keyword search."""

    def test_headphones_query_returns_product_issue_faqs(self, kb):
        results = find_relevant_faqs("pair wireless headphones bluetooth", kb)
        assert len(results) > 0
        categories = [r[2] for r in results]
        assert any("Product Issue" in c for c in categories)

    def test_order_tracking_query_returns_order_faqs(self, kb):
        results = find_relevant_faqs("track my order shipment courier", kb)
        assert len(results) > 0
        categories = [r[2] for r in results]
        assert any("Order Tracking" in c for c in categories)

    def test_warranty_query_returns_warranty_faqs(self, kb):
        results = find_relevant_faqs("warranty claim replacement defect", kb)
        assert len(results) > 0
        categories = [r[2] for r in results]
        assert any("Warranty Claim" in c for c in categories)

    def test_returns_at_most_3_results(self, kb):
        results = find_relevant_faqs("headphones smartwatch battery laptop", kb)
        assert len(results) <= 3

    def test_results_sorted_by_score_desc(self, kb):
        results = find_relevant_faqs("battery drain overheating", kb)
        if len(results) >= 2:
            scores = [r[0] for r in results]
            assert scores == sorted(scores, reverse=True)


# ============================================================
# 6. get_response()
# ============================================================

class TestGetResponse:
    """Test full response pipeline."""

    def test_returns_tuple_of_response_and_engine(self, kb):
        result = get_response("How do I pair my Wireless Headphones?", kb, session_id="test_sess")
        assert isinstance(result, tuple)
        assert len(result) == 2
        response, engine = result
        assert isinstance(response, str)
        assert isinstance(engine, str)

    def test_keyword_match_returns_keyword_engine(self, kb):
        response, engine = get_response("How do I pair my Wireless Headphones via Bluetooth?", kb, session_id="test_kw")
        assert engine in ("Keyword", "Keyword (Suggested)", "RAG", "RAG (Suggested)", "Gemini AI")

    def test_escalation_when_no_match(self, kb):
        _mock_rag_engine.query.return_value = []
        response, engine = get_response(
            "xyzrandomquerynothing12345nonexistent", kb, session_id="test_esc"
        )
        assert engine == "Escalated"
        assert "ticket" in response.lower() or "support" in response.lower()

    def test_rag_engine_called_on_low_keyword_score(self, kb):
        _mock_rag_engine.query.return_value = []
        get_response("xyzobscurequery123456", kb, session_id="test_rag_call")
        assert _mock_rag_engine.query.called

    def test_high_rag_score_returns_rag_engine(self, kb):
        _mock_rag_engine.query.return_value = [
            {"question": "How to fix smartwatch?", "answer": "Restart it.", "category": "Product Issue", "score": 0.85}
        ]
        response, engine = get_response("unusual question", kb, session_id="test_rag_high")
        assert engine in ("RAG", "Gemini AI")


# ============================================================
# 7. Conversation History
# ============================================================

class TestConversationHistory:
    """Test session memory."""

    def test_add_and_get_history(self):
        add_to_history("sess_1", "user", "Hello")
        add_to_history("sess_1", "bot", "Hi there!")
        ctx = get_history_context("sess_1")
        assert "user: Hello" in ctx
        assert "bot: Hi there!" in ctx

    def test_empty_session_returns_empty_string(self):
        assert get_history_context("nonexistent_session") == ""


# ============================================================
# 8. Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_very_long_input(self, kb):
        long_input = "headphones " * 500
        response, engine = get_response(long_input, kb, session_id="test_long")
        assert isinstance(response, str)

    def test_special_characters_input(self, kb):
        response, engine = get_response("!@#$%^&*()_+{}[]:;<>?,./~`", kb, session_id="test_special")
        assert isinstance(response, str)

    def test_numeric_only_input(self, kb):
        response, engine = get_response("1234567890", kb, session_id="test_nums")
        assert isinstance(response, str)

    def test_mixed_language_input(self, kb):
        response, engine = get_response("Wireless headphones ใช้งานยังไง", kb, session_id="test_mixed")
        assert isinstance(response, str)

    def test_emoji_input(self, kb):
        response, engine = get_response("🎧 ⌚ 💻 📦 🛡️", kb, session_id="test_emoji")
        assert isinstance(response, str)

    def test_tokenize_with_only_stop_words(self):
        assert tokenize_thai("is a the to and") == []

    def test_find_relevant_faqs_empty_input(self, kb):
        assert find_relevant_faqs("", kb) == []
