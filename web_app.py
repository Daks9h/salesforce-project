#!/usr/bin/env python3
"""
AI-Powered Customer Support and Service Automation - Backend Web App
Features SQLite persistent storage, customer ticket submission, public ticket lookup,
AI automation (auto-classification, priority, suggested replies & troubleshooting),
and secure SaaS Admin Dashboard.
"""

import os
import json
import sqlite3
import hashlib
import hmac
import secrets
import time
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from flask import (
    Flask, request, jsonify, send_file,
    session, redirect, url_for, abort
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from chatbot import load_kb, get_response, conversation_history
from ai_service import analyze_ticket, SUPPORTED_PRODUCTS, CATEGORIES, PRIORITIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# APP CONFIG
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

# Session cookie security
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

# Rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["300 per minute"],
    storage_uri="memory://"
)

# Admin credentials
ADMIN_USER = os.getenv("ADMIN_USER") or "admin"
ADMIN_PASS = os.getenv("ADMIN_PASS") or "admin123"

DB_PATH = os.environ.get("SQLITE_PATH") or (
    "/tmp/tickets.db" if os.environ.get("VERCEL") else os.path.join(os.path.dirname(__file__), "tickets.db")
)

# Load Knowledge Base
kb = load_kb()

# ============================================================
# DATABASE SETUP & MIGRATION
# ============================================================
@contextmanager
def get_db():
    """Get database connection with row_factory as context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize and migrate database tables."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                user_name TEXT DEFAULT 'Customer',
                email TEXT DEFAULT '',
                product_name TEXT DEFAULT '',
                order_id TEXT DEFAULT '',
                category TEXT DEFAULT 'General Inquiry',
                priority TEXT DEFAULT 'Medium',
                user_message TEXT NOT NULL,
                bot_response TEXT,
                suggested_response TEXT,
                suggested_troubleshooting TEXT,
                status TEXT DEFAULT 'Open',
                admin_reply TEXT,
                admin_user TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ticket_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                rating INTEGER,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
            CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
            CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_ticket ON ticket_messages(ticket_id);
        """)

        # Perform schema migrations for existing databases
        cursor = conn.execute("PRAGMA table_info(tickets)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        migration_cols = {
            "email": "TEXT DEFAULT ''",
            "product_name": "TEXT DEFAULT ''",
            "order_id": "TEXT DEFAULT ''",
            "suggested_response": "TEXT DEFAULT ''",
            "suggested_troubleshooting": "TEXT DEFAULT ''",
            "resolution_notes": "TEXT DEFAULT ''"
        }
        for col_name, col_def in migration_cols.items():
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE tickets ADD COLUMN {col_name} {col_def}")
                    logger.info("Migrated tickets table: added column %s", col_name)
                except sqlite3.OperationalError:
                    pass

        # Normalize statuses in existing tables
        conn.execute("UPDATE tickets SET status = 'Open' WHERE LOWER(status) = 'open'")
        conn.execute("UPDATE tickets SET status = 'In Progress' WHERE LOWER(status) in ('in progress', 'replied')")
        conn.execute("UPDATE tickets SET status = 'Resolved' WHERE LOWER(status) in ('resolved', 'closed')")

        conn.commit()

    # Seed sample demonstration data if database is empty or has fewer than 3 tickets
    seed_sample_data()

def seed_sample_data():
    """Seed realistic sample electronics tickets for demonstration."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if count >= 4:
            return

        sample_cases = [
            {
                "ticket_id": "TKT-20261101-H101",
                "user_name": "Alexander Wright",
                "email": "alex.wright@example.com",
                "product_name": "Wireless Headphones",
                "order_id": "ORD-94821",
                "category": "Product Issue",
                "priority": "High",
                "user_message": "My Wireless Headphones experience severe audio stutter and disconnect every 2 minutes when paired with my laptop.",
                "status": "Open",
                "suggested_response": "Dear Alexander,\n\nThank you for reporting the Bluetooth stutter on your Wireless Headphones. Please try clearing the paired devices list by holding Power + Volume Down for 10 seconds, then re-pair. Our technical support agent will follow up shortly.",
                "suggested_troubleshooting": "1. Reset Bluetooth cache.\n2. Keep within 10 meters of laptop.\n3. Hold Power + Volume Down for 10 seconds to factory reset."
            },
            {
                "ticket_id": "TKT-20261101-W202",
                "user_name": "Sophia Martinez",
                "email": "sophia.m@example.com",
                "product_name": "Smartwatch",
                "order_id": "ORD-88319",
                "category": "Warranty Claim",
                "priority": "High",
                "user_message": "The heart rate sensor on the back of my Smartwatch stopped emitting green light and the screen is unresponsive after swimming.",
                "status": "In Progress",
                "suggested_response": "Dear Sophia,\n\nWe have initiated your warranty claim for the Smartwatch sensor malfunction. A prepaid return shipping label has been generated for hardware diagnostics.",
                "suggested_troubleshooting": "1. Inspect water seal.\n2. Verify 1-year warranty coverage.\n3. Generate prepaid RMA label for diagnostics."
            },
            {
                "ticket_id": "TKT-20261101-O303",
                "user_name": "Marcus Chen",
                "email": "mchen@example.com",
                "product_name": "Laptop",
                "order_id": "ORD-77402",
                "category": "Order Tracking",
                "priority": "Urgent",
                "user_message": "Tracking status shows 'Delivered by FedEx' yesterday at 3:15 PM, but the package was never received at my address or building reception.",
                "status": "Open",
                "suggested_response": "Dear Marcus,\n\nWe apologize for this delivery issue. We have opened a formal carrier trace with FedEx under case #TR-77402 and will notify you with the courier delivery GPS coordinates within 24 hours.",
                "suggested_troubleshooting": "1. Check perimeter and building management.\n2. Initiate FedEx courier tracer.\n3. Prepare replacement dispatch if parcel is lost."
            },
            {
                "ticket_id": "TKT-20261101-G404",
                "user_name": "Elena Rostova",
                "email": "elena.r@example.com",
                "product_name": "Bluetooth Speaker",
                "order_id": "ORD-62104",
                "category": "General Inquiry",
                "priority": "Low",
                "user_message": "Can I pair two of the ApexTech Bluetooth Speakers together for True Wireless Stereo (TWS) playback?",
                "status": "Resolved",
                "suggested_response": "Dear Elena,\n\nYes! You can press and hold the Bluetooth button on both speakers simultaneously for 3 seconds to link them into True Wireless Stereo mode (Left and Right channels).",
                "suggested_troubleshooting": "1. Turn both speakers on.\n2. Hold Bluetooth button on both for 3s.\n3. Wait for the 'Stereo Paired' voice chime."
            },
            {
                "ticket_id": "TKT-20261101-E505",
                "user_name": "David Kim",
                "email": "dkim@example.com",
                "product_name": "Wireless Earbuds",
                "order_id": "ORD-51923",
                "category": "Product Issue",
                "priority": "Medium",
                "user_message": "The left earbud does not illuminate or charge when placed inside the charging case.",
                "status": "In Progress",
                "suggested_response": "Dear David,\n\nWe recommend cleaning the charging contacts with a dry cotton swab. If the contact pin is depressed or oxidized, we will gladly arrange a replacement earbud.",
                "suggested_troubleshooting": "1. Clean charging contacts with alcohol swab.\n2. Press and hold case button for 15s to reset.\n3. Test with another USB-C cable."
            }
        ]

        for case in sample_cases:
            conn.execute("""
                INSERT OR IGNORE INTO tickets (
                    ticket_id, user_name, email, product_name, order_id,
                    category, priority, user_message, status,
                    suggested_response, suggested_troubleshooting
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case["ticket_id"], case["user_name"], case["email"],
                case["product_name"], case["order_id"], case["category"],
                case["priority"], case["user_message"], case["status"],
                case["suggested_response"], case["suggested_troubleshooting"]
            ))

            conn.execute("""
                INSERT OR IGNORE INTO ticket_messages (ticket_id, sender, message)
                VALUES (?, 'customer', ?)
            """, (case["ticket_id"], case["user_message"]))

        conn.commit()
        logger.info("Seeded sample electronics support tickets into tickets.db")

def generate_ticket_id():
    """Generate unique ticket ID like TKT-20261101-A1B2."""
    now = datetime.now()
    rand = secrets.token_hex(2).upper()
    return f"TKT-{now.strftime('%Y%m%d')}-{rand}"

# Initialize DB on startup
init_db()

# ============================================================
# AUTH MIDDLEWARE
# ============================================================
def admin_required(f):
    """Decorator to require admin login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("dashboard_view"))
        return f(*args, **kwargs)
    return decorated

# ============================================================
# PUBLIC UI ROUTES
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    """Serve the main customer support portal."""
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/dashboard")
def dashboard_view():
    """Serve admin dashboard."""
    return send_file(os.path.join(BASE_DIR, "dashboard.html"))

@app.route("/static/kb.js")
def serve_kb():
    """Serve the KB data as a JS file."""
    return send_file(os.path.join(BASE_DIR, "kb.js"), mimetype="application/javascript")

@app.route("/sw.js")
def serve_sw():
    """Service worker for offline PWA support."""
    return send_file(os.path.join(BASE_DIR, "sw.js"), mimetype="application/javascript")

# ============================================================
# ADMIN AUTH ROUTES
# ============================================================
@app.route("/api/admin/login", methods=["POST"])
@limiter.limit("15 per minute")
def admin_login():
    """Admin login endpoint."""
    data = request.json or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if hmac.compare_digest(username, ADMIN_USER) and hmac.compare_digest(password, ADMIN_PASS):
        session["admin_logged_in"] = True
        session["admin_user"] = username
        return jsonify({"status": "ok", "message": "Admin authenticated successfully"})
    return jsonify({"status": "error", "message": "Invalid administrator username or password"}), 401

@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    """Admin logout."""
    session.clear()
    return jsonify({"status": "ok", "message": "Logged out successfully"})

@app.route("/api/admin/status")
def admin_status():
    """Check if admin is currently logged in."""
    return jsonify({
        "logged_in": session.get("admin_logged_in", False),
        "admin_user": session.get("admin_user", "")
    })

# ============================================================
# CHAT API
# ============================================================
@app.route("/api/chat", methods=["POST"])
@limiter.limit("60 per minute")
def chat():
    """Customer Chatbot API with RAG & AI search."""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    user_input = data.get("message", "").strip()
    session_id = data.get("session_id", request.remote_addr)

    if not user_input:
        return jsonify({"response": "Please enter your question or describe the issue you are experiencing."})

    if len(user_input) > 2000:
        return jsonify({"error": "Message length exceeds maximum limit (2000 characters)"}), 400

    response, engine_name = get_response(user_input, kb, session_id)
    return jsonify({
        "response": response,
        "engine": engine_name,
        "session_id": session_id
    })

# ============================================================
# PUBLIC TICKET SUBMISSION & LOOKUP
# ============================================================
@app.route("/api/tickets/new", methods=["POST"])
@limiter.limit("30 per minute")
def create_ticket():
    """
    Public endpoint: Customer submits a support ticket.
    Triggers AI automation to classify category, estimate priority,
    generate suggested responses, and recommend troubleshooting instructions.
    """
    data = request.json or {}
    user_name = data.get("customer_name") or data.get("user_name") or "Valued Customer"
    email = data.get("email", "").strip()
    product_name = data.get("product_name", "").strip()
    order_id = data.get("order_id", "").strip()
    provided_category = data.get("category", "").strip()
    message = data.get("message") or data.get("description", "").strip()
    bot_response = data.get("bot_response", "")

    if not message:
        return jsonify({"error": "Problem description is required"}), 400

    if len(message) > 5000:
        return jsonify({"error": "Description too long"}), 400

    # Execute AI Automation
    ai_insights = analyze_ticket(message, product_name=product_name)

    # Use selected category if valid, otherwise use AI classification
    final_category = provided_category if provided_category in CATEGORIES else ai_insights["category"]
    final_priority = ai_insights.get("priority", "Medium")
    suggested_response = ai_insights.get("suggested_response", "")
    suggested_troubleshooting = ai_insights.get("suggested_troubleshooting", "")

    ticket_id = generate_ticket_id()

    with get_db() as conn:
        conn.execute("""
            INSERT INTO tickets (
                ticket_id, user_name, email, product_name, order_id,
                category, priority, user_message, bot_response,
                suggested_response, suggested_troubleshooting, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open')
        """, (
            ticket_id, user_name, email, product_name, order_id,
            final_category, final_priority, message, bot_response,
            suggested_response, suggested_troubleshooting
        ))

        # Add initial customer message to conversation thread
        conn.execute("""
            INSERT INTO ticket_messages (ticket_id, sender, message)
            VALUES (?, 'customer', ?)
        """, (ticket_id, message))

        conn.commit()

    return jsonify({
        "status": "success",
        "ticket_id": ticket_id,
        "category": final_category,
        "priority": final_priority,
        "message": f"Your support ticket {ticket_id} has been submitted successfully.",
        "ai_insights": {
            "suggested_response": suggested_response,
            "suggested_troubleshooting": suggested_troubleshooting,
            "method": ai_insights.get("method", "Rule-Based Intelligence")
        }
    }), 201

@app.route("/api/tickets/lookup/<ticket_id>", methods=["GET"])
@limiter.limit("60 per minute")
def lookup_ticket(ticket_id):
    """
    Public endpoint: Allows customers to check the real-time status
    and resolution notes of their submitted ticket using Ticket ID.
    """
    ticket_id = ticket_id.strip()
    with get_db() as conn:
        row = conn.execute("""
            SELECT ticket_id, user_name, email, product_name, order_id,
                   category, priority, user_message, status, admin_reply,
                   created_at, updated_at, closed_at
            FROM tickets
            WHERE ticket_id = ?
        """, (ticket_id,)).fetchone()

        if not row:
            return jsonify({"status": "not_found", "message": f"No ticket found matching ID '{ticket_id}'"}), 404

        messages = conn.execute("""
            SELECT sender, message, created_at
            FROM ticket_messages
            WHERE ticket_id = ?
            ORDER BY created_at ASC
        """, (ticket_id,)).fetchall()

    return jsonify({
        "status": "success",
        "ticket": dict(row),
        "messages": [dict(m) for m in messages]
    })

# ============================================================
# ADMIN TICKET MANAGEMENT & DASHBOARD API
# ============================================================
@app.route("/api/tickets", methods=["GET"])
@admin_required
def get_tickets():
    """Fetch tickets with status, category, priority, and search filters."""
    status = request.args.get("status")
    category = request.args.get("category")
    priority = request.args.get("priority")
    search = request.args.get("search", "").strip()

    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        limit, offset = 50, 0

    with get_db() as conn:
        query = "SELECT * FROM tickets WHERE 1=1"
        params = []

        if status and status != "all":
            query += " AND status = ?"
            params.append(status)

        if category and category != "all":
            query += " AND category = ?"
            params.append(category)

        if priority and priority != "all":
            query += " AND priority = ?"
            params.append(priority)

        if search:
            query += " AND (ticket_id LIKE ? OR user_name LIKE ? OR email LIKE ? OR product_name LIKE ? OR order_id LIKE ? OR user_message LIKE ?)"
            s = f"%{search}%"
            params.extend([s, s, s, s, s, s])

        count_query = query.replace("SELECT *", "SELECT COUNT(*)", 1)
        total = conn.execute(count_query, params).fetchone()[0]

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

    return jsonify({
        "tickets": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset
    })

@app.route("/api/tickets/<ticket_id>", methods=["GET"])
@admin_required
def get_ticket_detail(ticket_id):
    """Get single ticket details including AI suggestions and message history."""
    with get_db() as conn:
        ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        messages = conn.execute("""
            SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY created_at ASC
        """, (ticket_id,)).fetchall()

    return jsonify({
        "ticket": dict(ticket),
        "messages": [dict(m) for m in messages]
    })

@app.route("/api/tickets/<ticket_id>/update", methods=["POST"])
@admin_required
def update_ticket(ticket_id):
    """Update ticket status, priority, and add resolution notes."""
    data = request.json or {}
    status = data.get("status")
    priority = data.get("priority")
    notes = data.get("resolution_notes") or data.get("admin_reply")
    admin_user = session.get("admin_user", "admin")

    with get_db() as conn:
        ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)
            if status == "Resolved":
                updates.append("closed_at = CURRENT_TIMESTAMP")
            elif status in ("Open", "In Progress"):
                updates.append("closed_at = NULL")

        if priority:
            updates.append("priority = ?")
            params.append(priority)

        if notes is not None:
            notes_str = str(notes).strip()
            updates.append("admin_reply = ?")
            params.append(notes_str)
            updates.append("admin_user = ?")
            params.append(admin_user)

            if notes_str:
                conn.execute("""
                    INSERT INTO ticket_messages (ticket_id, sender, message)
                    VALUES (?, 'admin', ?)
                """, (ticket_id, notes_str))

        params.append(ticket_id)
        sql = f"UPDATE tickets SET {', '.join(updates)} WHERE ticket_id = ?"
        conn.execute(sql, params)
        conn.commit()

    return jsonify({"status": "success", "message": f"Ticket {ticket_id} updated successfully"})

@app.route("/api/tickets/<ticket_id>/reply", methods=["POST"])
@admin_required
def reply_ticket(ticket_id):
    """Admin sends a reply to the customer."""
    data = request.json or {}
    reply_text = data.get("reply", "").strip()
    admin_user = session.get("admin_user", "admin")

    if not reply_text:
        return jsonify({"error": "Reply message cannot be empty"}), 400

    with get_db() as conn:
        ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        conn.execute("""
            UPDATE tickets
            SET admin_reply = ?, admin_user = ?, status = 'In Progress',
                updated_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
        """, (reply_text, admin_user, ticket_id))

        conn.execute("""
            INSERT INTO ticket_messages (ticket_id, sender, message)
            VALUES (?, 'admin', ?)
        """, (ticket_id, reply_text))

        conn.commit()

    return jsonify({"status": "success", "message": "Reply recorded successfully"})

@app.route("/api/tickets/<ticket_id>/close", methods=["POST"])
@admin_required
def close_ticket(ticket_id):
    """Mark ticket as Resolved."""
    with get_db() as conn:
        ticket = conn.execute("SELECT 1 FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        conn.execute("""
            UPDATE tickets
            SET status = 'Resolved', closed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
        """, (ticket_id,))
        conn.commit()
    return jsonify({"status": "success", "message": "Ticket resolved"})

@app.route("/api/tickets/<ticket_id>/reopen", methods=["POST"])
@admin_required
def reopen_ticket(ticket_id):
    """Reopen a resolved ticket."""
    with get_db() as conn:
        ticket = conn.execute("SELECT 1 FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        conn.execute("""
            UPDATE tickets
            SET status = 'Open', closed_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
        """, (ticket_id,))
        conn.commit()
    return jsonify({"status": "success", "message": "Ticket reopened"})

# ============================================================
# ANALYTICS & DASHBOARD METRICS API
# ============================================================
@app.route("/api/analytics")
@admin_required
def analytics():
    """Get statistics for the SaaS Admin Dashboard."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        open_count = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Open'").fetchone()[0]
        in_progress_count = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'In Progress'").fetchone()[0]
        resolved_count = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Resolved'").fetchone()[0]

        # Category distribution
        categories = {}
        for row in conn.execute("SELECT category, COUNT(*) as cnt FROM tickets GROUP BY category"):
            categories[row["category"]] = row["cnt"]

        # Priority distribution
        priorities = {}
        for row in conn.execute("SELECT priority, COUNT(*) as cnt FROM tickets GROUP BY priority"):
            priorities[row["priority"]] = row["cnt"]

        # Recent complaints (last 5)
        recent = conn.execute("""
            SELECT ticket_id, user_name, product_name, category, priority, status, user_message, created_at
            FROM tickets
            ORDER BY created_at DESC
            LIMIT 5
        """).fetchall()

        # Resolution rate
        resolution_rate = round((resolved_count / total * 100), 1) if total > 0 else 0

    return jsonify({
        "total_tickets": total,
        "open_tickets": open_count,
        "in_progress_tickets": in_progress_count,
        "resolved_tickets": resolved_count,
        "resolution_rate": resolution_rate,
        "category_distribution": categories,
        "priority_distribution": priorities,
        "recent_tickets": [dict(r) for r in recent]
    })

# ============================================================
# HEALTH & METADATA
# ============================================================
@app.route("/api/health")
def health():
    """System health check and metadata."""
    with get_db() as conn:
        ticket_count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    faq_count = sum(len(c.get("faqs", [])) for c in kb.get("categories", []))

    return jsonify({
        "status": "ok",
        "app_title": "AI-Powered Customer Support and Service Automation",
        "company": "ApexTech Electronics & Gadgets Global",
        "faqs_count": faq_count,
        "tickets_count": ticket_count,
        "db_available": os.path.exists(DB_PATH),
        "ai_enabled": bool(os.getenv("GOOGLE_API_KEY")),
        "supported_products": SUPPORTED_PRODUCTS,
        "categories": CATEGORIES,
        "version": "3.0.0"
    })

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("  AI-Powered Customer Support & Service Automation")
    logger.info("  Customer Portal : http://localhost:5000")
    logger.info("  Admin Dashboard : http://localhost:5000/dashboard")
    logger.info("  Database Path   : %s", DB_PATH)
    logger.info("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
