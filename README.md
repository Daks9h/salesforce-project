# AI-Powered Customer Support and Service Automation

An academic full-stack project demonstrating intelligent customer support automation, retrieval-augmented generation (RAG), and service ticket management for an international Electronics & Gadgets enterprise.

---

## 📌 Project Overview

This application simulates the customer support operations of **ApexTech Electronics & Gadgets Global**, addressing customer inquiries across four primary domains:
1. **Product-Related Problems** (Wireless Headphones, Smartwatches, Laptops, Bluetooth Speakers, Wireless Earbuds)
2. **Order Tracking** (Courier tracking, shipping status, delayed delivery, address modification)
3. **Warranty Claims** (1-Year Manufacturer Warranty, defective unit replacements, RMA procedures)
4. **General Customer Inquiries** (30-day return policy, specifications, store hours, contact channels)

The core objective is to automate frontline customer support using an **AI Assistant** and streamline issue escalation via an integrated **Support Ticket Management System** and **SaaS Admin Dashboard**.

---

## 🏗️ Architecture & Technology Stack

```
                          ┌────────────────────────┐
                          │   Client Web Browser   │
                          │   (Customer & Admin)   │
                          └───────────┬────────────┘
                                      │ HTTP / JSON
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │            Flask Application Backend             │
             │           (web_app.py + Rate Limiter)            │
             └──────┬─────────────────┬─────────────────┬───────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
          ┌──────────────────┐ ┌──────────────┐ ┌───────────────┐
          │   Chatbot RAG    │ │  AI Service  │ │ SQLite Store  │
          │   (chatbot.py)   │ │(ai_service.py│ │ (tickets.db)  │
          └────────┬─────────┘ └──────┬───────┘ └───────────────┘
                   │                  │
                   ▼                  ▼
          ┌──────────────────┐ ┌──────────────┐
          │ ChromaDB Vector  │ │Google Gemini │
          │  (chroma_db)     │ │ / Rule Engine│
          └──────────────────┘ └──────────────┘
```

- **Frontend Interface**: Lightweight, responsive SaaS design with a light palette and navy blue/indigo accents. Built with HTML5, CSS3, and modern vanilla JavaScript. No heavy external frameworks required.
- **Backend Framework**: Python 3.12 with **Flask 3.1**, session-based admin authentication, and **Flask-Limiter** rate limiting.
- **Persistent Database**: SQLite (`tickets.db`) with Write-Ahead Logging (WAL), automatic schema migration, and demo sample data seeding.
- **AI Automation & RAG**:
  - **Vector Embeddings**: ChromaDB persistent store with fast, deterministic, zero-download feature-hashed embeddings and support for Google Generative AI embeddings.
  - **Hybrid Search**: Keyword scoring & synonym expansion $\rightarrow$ Vector search $\rightarrow$ Generative AI synthesis.
  - **Ticket Intelligence**: Automatic complaint classification, priority estimation, suggested customer responses, and recommended troubleshooting steps.
  - **Graceful Fallback**: 100% operational offline without any external API keys via rule-based heuristics and knowledge-base templates.
- **Deployment Compatibility**: Gunicorn (Linux/Containers), Waitress (Windows/Cross-platform), Docker, Render, Railway.

---

## 🚀 Key Features

### 1. Customer Support Chatbot
- Customers can describe electronics problems in natural language.
- Real-time answers retrieved from a curated Electronics & Gadgets knowledge base.
- If an inquiry cannot be resolved or requires hands-on assistance, the chatbot automatically recommends creating a support ticket with a one-click transition.

### 2. Support Ticket Management System
- Collects: **Customer Name**, **Email**, **Product Name**, **Order ID**, **Issue Category**, and **Problem Description**.
- Automatically generates a unique, human-readable Ticket ID (e.g., `TKT-20261101-A1B2`).
- Standardized ticket categories:
  - `Product Issue`
  - `Order Tracking`
  - `Warranty Claim`
  - `General Inquiry`
- Ticket life-cycle statuses: `Open`, `In Progress`, `Resolved`.

### 3. Public Ticket Status Lookup
- Customers can enter their Ticket ID to verify:
  - Live status badge (`Open`, `In Progress`, `Resolved`)
  - Assigned priority (`Low`, `Medium`, `High`, `Urgent`)
  - Recorded complaint summary
  - Official resolution notes and replies from support specialists

### 4. SaaS Administrator Dashboard
- Protected by secure session-based authentication (`/dashboard`).
- **Real-Time Analytics**:
  - Total tickets, Open tickets, In-Progress tickets, Resolved tickets, and Resolution Rate %.
  - Recent customer complaint stream with one-click management.
- **Interactive Ticket Table**:
  - Filter by Status, Category, and Priority.
  - Live full-text search by Ticket ID, customer name, email, or product.
- **Ticket Diagnostics & Resolution Modal**:
  - Complete customer submission details.
  - **AI Automation Insights Panel**: Displays AI-detected category, suggested priority, recommended troubleshooting steps, and an AI-drafted reply with a **"Use Suggested Response"** button.
  - Administrator can update status, change priority, and append official resolution notes.

---

## 📦 Electronics Sample Data

The application pre-seeds realistic demonstration cases for five core electronics products:
1. **Wireless Headphones**: Bluetooth pairing issues, audio stuttering, battery drain.
2. **Smartwatch**: Companion app sync failure, heart rate sensor diagnostics, battery optimization.
3. **Laptop**: Thermal management, fan noise, display flickering, driver troubleshooting.
4. **Bluetooth Speaker**: Charging issues, hard reset instructions, True Wireless Stereo (TWS) pairing.
5. **Wireless Earbuds**: Single earbud audio loss, charging contact maintenance, factory reset.

---

## 🛠️ Local Setup Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/romeototo/it-support-chatbot.git
cd it-support-chatbot
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
*(Optional)* Add your `GOOGLE_API_KEY` in `.env` if you want Google Gemini AI active. If omitted, the application runs seamlessly on built-in offline intelligence.

### 5. Initialize Knowledge Base & Vector Database
```bash
python init_rag.py
```

### 6. Run the Application
**Development Server:**
```bash
python web_app.py
```

**Production WSGI Server (Windows):**
```bash
waitress-serve --port=5000 web_app:app
```

**Production WSGI Server (Linux / macOS):**
```bash
gunicorn --bind 0.0.0.0:5000 web_app:app
```

### 7. Access Portals
- **Customer Portal**: [http://localhost:5000](http://localhost:5000)
- **Admin Dashboard**: [http://localhost:5000/dashboard](http://localhost:5000/dashboard)
  - Default Username: `admin`
  - Default Password: `admin123`

---

## 🧪 Testing

The test suite covers unit tests for the chatbot, RAG engine, AI classification service, frontend static contracts, and full API integration tests:

```bash
# Run all tests
pytest -v
```

Expected result:
```
======================= 49 passed in 2.03s =======================
```

---

## 🌐 Production Deployment Guide

### Option 1: Vercel (Easiest Serverless Deployment)
The repository is pre-configured with [vercel.json](file:///c:/Users/ADMIN/Desktop/salesforce-project/vercel.json) and [api/index.py](file:///c:/Users/ADMIN/Desktop/salesforce-project/api/index.py):
1. Push your repository to **GitHub**.
2. Go to [Vercel Dashboard](https://vercel.com/) and click **"Add New..."** $\rightarrow$ **"Project"**.
3. Import your GitHub repository.
4. Expand **Environment Variables** and optionally add:
   - `ADMIN_USER`: `admin`
   - `ADMIN_PASS`: `<your-admin-password>`
   - `FLASK_SECRET_KEY`: `<random-32-byte-hex>`
   - `GOOGLE_API_KEY`: *(Optional: your Gemini API key)*
5. Click **Deploy**. Vercel will build and assign your free live HTTPS URL (e.g., `https://your-project.vercel.app`).

### Option 2: Render (Recommended for Long-Term SQLite Persistence)
1. Fork or push this repository to your GitHub account.
2. Go to [Render Dashboard](https://dashboard.render.com/) $\rightarrow$ **New Web Service**.
3. Connect your GitHub repository.
4. Configure service settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt && python init_rag.py`
   - **Start Command**: `gunicorn web_app:app`
5. In **Environment Variables**, add:
   - `FLASK_ENV`: `production`
   - `ADMIN_USER`: `admin`
   - `ADMIN_PASS`: `<your-secure-password>`
   - `FLASK_SECRET_KEY`: `<random-32-byte-hex>`
   - `GOOGLE_API_KEY`: *(Optional: your Gemini API key)*
6. Click **Deploy Web Service**.

### Option 2: Docker Container
```bash
# Build Docker image
docker build -t apextech-support .

# Run container
docker run -p 5000:5000 -e ADMIN_PASS="secret123" apextech-support
```

---

## 📡 API Reference Summary

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/api/health` | Public | System health check and metadata |
| `POST` | `/api/chat` | Public | Chatbot inquiry endpoint (RAG / AI) |
| `POST` | `/api/tickets/new` | Public | Submit support ticket with AI auto-classification |
| `GET` | `/api/tickets/lookup/<ticket_id>` | Public | Track ticket status and resolution notes |
| `POST` | `/api/admin/login` | Public | Authenticate administrator session |
| `POST` | `/api/admin/logout` | Protected | Invalidate administrator session |
| `GET` | `/api/admin/status` | Public | Verify session authentication state |
| `GET` | `/api/tickets` | Protected | List tickets with filters (`status`, `category`, `search`) |
| `GET` | `/api/tickets/<ticket_id>` | Protected | Get single ticket detail + AI suggestions |
| `POST` | `/api/tickets/<ticket_id>/update` | Protected | Update ticket status, priority, and resolution notes |
| `GET` | `/api/analytics` | Protected | Dashboard metrics and recent complaints |

---

## 📄 License
This project is licensed under the MIT License.
