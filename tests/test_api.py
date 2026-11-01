#!/usr/bin/env python3
"""
Integration & API Tests for AI-Powered Customer Support and Service Automation
Tests:
- API Health & Metadata
- Chatbot Endpoint
- Customer Ticket Submission & AI Auto-Classification
- Public Ticket Status Lookup
- Admin Authentication & Authorization
- Ticket Management (Filtering, Search, Status Update, Resolution Notes)
- Dashboard Analytics
- AI Automation Service (Rule-based & Heuristics)
"""

import pytest
import json
import web_app
from ai_service import analyze_ticket, detect_product, rule_based_classify

@pytest.fixture
def client():
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as client:
        yield client

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "ApexTech" in data["company"]
    assert len(data["supported_products"]) == 5
    assert "Wireless Headphones" in data["supported_products"]
    assert data["faqs_count"] > 0

def test_chat_endpoint_valid_question(client):
    res = client.post("/api/chat", json={"message": "How do I pair my Wireless Headphones via Bluetooth?"})
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert "Bluetooth" in data["response"] or "Headphones" in data["response"]
    assert "engine" in data

def test_chat_endpoint_empty_message(client):
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 200
    data = res.get_json()
    assert "Please enter your question" in data["response"]

def test_ticket_submission_and_ai_classification(client):
    payload = {
        "customer_name": "Sarah Connor",
        "email": "sarah@example.com",
        "product_name": "Wireless Headphones",
        "order_id": "ORD-55441",
        "description": "My headphones are smoking and the battery is swelling dangerous emergency"
    }
    res = client.post("/api/tickets/new", json=payload)
    assert res.status_code == 201
    data = res.get_json()
    assert data["status"] == "success"
    assert "TKT-" in data["ticket_id"]
    assert data["category"] == "Product Issue"
    assert data["priority"] == "Urgent"
    assert "ai_insights" in data
    assert "suggested_response" in data["ai_insights"]
    assert "suggested_troubleshooting" in data["ai_insights"]

def test_public_ticket_lookup(client):
    # Create ticket
    payload = {
        "customer_name": "Tom Holland",
        "email": "tom@example.com",
        "product_name": "Smartwatch",
        "order_id": "ORD-11223",
        "description": "Smartwatch screen is blank and will not sync with iPhone."
    }
    create_res = client.post("/api/tickets/new", json=payload)
    ticket_id = create_res.get_json()["ticket_id"]

    # Public lookup
    lookup_res = client.get(f"/api/tickets/lookup/{ticket_id}")
    assert lookup_res.status_code == 200
    lookup_data = lookup_res.get_json()
    assert lookup_data["status"] == "success"
    assert lookup_data["ticket"]["ticket_id"] == ticket_id
    assert lookup_data["ticket"]["user_name"] == "Tom Holland"
    assert lookup_data["ticket"]["status"] == "Open"
    assert len(lookup_data["messages"]) >= 1

def test_public_ticket_lookup_nonexistent(client):
    res = client.get("/api/tickets/lookup/TKT-INVALID-9999")
    assert res.status_code == 404
    data = res.get_json()
    assert data["status"] == "not_found"

def test_admin_auth_and_protected_routes(client):
    # Unauthenticated access
    res = client.get("/api/tickets")
    assert res.status_code == 401

    # Login with wrong credentials
    bad_login = client.post("/api/admin/login", json={"username": "admin", "password": "wrongpassword"})
    assert bad_login.status_code == 401

    # Login with valid credentials
    good_login = client.post("/api/admin/login", json={"username": "admin", "password": "admin123"})
    assert good_login.status_code == 200
    assert good_login.get_json()["status"] == "ok"

    # Status check
    status_res = client.get("/api/admin/status")
    assert status_res.get_json()["logged_in"] is True

    # Authenticated access to tickets
    tickets_res = client.get("/api/tickets")
    assert tickets_res.status_code == 200
    assert "tickets" in tickets_res.get_json()

    # Authenticated access to analytics
    analytics_res = client.get("/api/analytics")
    assert analytics_res.status_code == 200
    adata = analytics_res.get_json()
    assert "total_tickets" in adata
    assert "open_tickets" in adata
    assert "in_progress_tickets" in adata
    assert "resolved_tickets" in adata

    # Logout
    logout_res = client.post("/api/admin/logout")
    assert logout_res.status_code == 200
    assert client.get("/api/tickets").status_code == 401

def test_admin_update_ticket(client):
    # Create ticket
    payload = {
        "customer_name": "Bruce Wayne",
        "email": "bruce@wayne.com",
        "product_name": "Laptop",
        "order_id": "ORD-99999",
        "description": "Laptop battery drains in 15 minutes."
    }
    create_res = client.post("/api/tickets/new", json=payload)
    ticket_id = create_res.get_json()["ticket_id"]

    # Admin Login
    client.post("/api/admin/login", json={"username": "admin", "password": "admin123"})

    # Update ticket status to In Progress with resolution notes
    update_res = client.post(f"/api/tickets/{ticket_id}/update", json={
        "status": "In Progress",
        "priority": "High",
        "resolution_notes": "We have reviewed the battery report and scheduled a technician."
    })
    assert update_res.status_code == 200
    assert update_res.get_json()["status"] == "success"

    # Verify updated values via lookup
    lookup_res = client.get(f"/api/tickets/lookup/{ticket_id}")
    t = lookup_res.get_json()["ticket"]
    assert t["status"] == "In Progress"
    assert t["priority"] == "High"
    assert "technician" in t["admin_reply"]

def test_ai_service_classification():
    # Order Tracking
    ot = rule_based_classify("Where is my package? FedEx delivery status is delayed", "Laptop")
    assert ot["category"] == "Order Tracking"

    # Warranty Claim
    wc = rule_based_classify("I need to file a warranty claim for replacement defective unit", "Smartwatch")
    assert wc["category"] == "Warranty Claim"

    # Product Issue
    pi = rule_based_classify("My wireless earbuds will not pair and the left side has no audio", "Wireless Earbuds")
    assert pi["category"] == "Product Issue"

    # General Inquiry
    gi = rule_based_classify("What are your store hours and return policy?", "Bluetooth Speaker")
    assert gi["category"] == "General Inquiry"

def test_ai_service_detect_product():
    assert detect_product("My headphones are stuttering") == "Wireless Headphones"
    assert detect_product("Smartwatch heart rate sensor") == "Smartwatch"
    assert detect_product("Laptop fan noise") == "Laptop"
    assert detect_product("Portable bluetooth speaker volume") == "Bluetooth Speaker"
    assert detect_product("Wireless earbuds charging case") == "Wireless Earbuds"

