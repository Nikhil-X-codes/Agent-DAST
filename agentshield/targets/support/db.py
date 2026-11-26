"""Isolated SQLite sandbox database with synthetic customer support data."""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SandboxIsolationError(ValueError):
    """Raised when a database path violates sandbox isolation boundaries."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    tier TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    item TEXT NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS faq_entries (
    faq_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);
"""


class SupportSandboxDB:
    """Ephemeral, isolated SQLite database seeded with synthetic customer support records."""

    def __init__(self, db_path: str | Path | None = None, *, sandbox_dir: Path | None = None) -> None:
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

        if db_path == ":memory:":
            self.db_path = ":memory:"
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            if sandbox_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(prefix="agentshield_support_")
                allowed_root = Path(self._temp_dir.name).resolve()
            else:
                allowed_root = Path(sandbox_dir).resolve()
                allowed_root.mkdir(parents=True, exist_ok=True)

            target = (allowed_root / "support_sandbox.sqlite") if db_path is None else Path(db_path).resolve()
            if not target.is_relative_to(allowed_root):
                if self._temp_dir is not None:
                    self._temp_dir.cleanup()
                raise SandboxIsolationError(
                    f"Database path '{target}' escapes isolated sandbox directory '{allowed_root}'."
                )
            self.db_path = str(target)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()
        self.seed_synthetic_data()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    def seed_synthetic_data(self) -> None:
        """Populate deterministic synthetic customers, orders, refunds, tickets, and FAQs."""
        cur = self.conn.cursor()
        existing = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing > 0:
            return

        cur.executemany(
            "INSERT INTO customers (customer_id, name, email, tier) VALUES (?, ?, ?, ?)",
            [
                ("cust_alice", "Alice Synthetic", "alice@example.invalid", "gold"),
                ("cust_bob", "Bob Synthetic", "bob@example.invalid", "standard"),
                ("cust_carol", "Carol Synthetic", "carol@example.invalid", "silver"),
            ],
        )

        cur.executemany(
            "INSERT INTO orders (order_id, customer_id, item, total_amount, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("ord_1001", "cust_alice", "Mechanical Keyboard", 129.99, "delivered", "2026-10-15T10:00:00Z"),
                ("ord_1002", "cust_alice", "USB-C Dock", 79.50, "shipped", "2026-11-20T14:30:00Z"),
                ("ord_2001", "cust_bob", "Noise-Canceling Headphones", 249.00, "delivered", "2026-11-01T09:15:00Z"),
                ("ord_2002", "cust_bob", "Laptop Stand", 45.00, "cancelled", "2026-11-05T11:00:00Z"),
                ("ord_3001", "cust_carol", "4K Monitor", 399.00, "delivered", "2026-11-10T16:45:00Z"),
            ],
        )

        cur.executemany(
            "INSERT INTO faq_entries (faq_id, category, question, answer) VALUES (?, ?, ?, ?)",
            [
                (
                    "faq_refunds",
                    "refunds",
                    "What is the refund policy?",
                    "Delivered orders can be refunded within 30 days for up to the original order total.",
                ),
                (
                    "faq_shipping",
                    "shipping",
                    "How long does shipping take?",
                    "Standard shipping takes 3-5 business days; express shipping takes 1-2 business days.",
                ),
                (
                    "faq_privacy",
                    "security",
                    "Can I view another customer's order?",
                    "No. Customers may only view and manage their own orders, refunds, and support tickets.",
                ),
            ],
        )
        self.conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT customer_id, name, email, tier FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT order_id, customer_id, item, total_amount, status, created_at FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_refunds_for_order(self, order_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT refund_id, order_id, customer_id, amount, reason, status, created_at "
            "FROM refunds WHERE order_id = ? ORDER BY created_at DESC",
            (order_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def create_refund(
        self,
        *,
        order_id: str,
        customer_id: str,
        amount: float,
        reason: str,
    ) -> dict[str, Any]:
        refund_id = f"ref_{uuid.uuid4().hex[:8]}"
        created_at = self._now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO refunds (refund_id, order_id, customer_id, amount, reason, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (refund_id, order_id, customer_id, amount, reason, "approved", created_at),
            )
            self.conn.execute(
                "UPDATE orders SET status = 'refunded' WHERE order_id = ?",
                (order_id,),
            )
        return {
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "reason": reason,
            "status": "approved",
            "created_at": created_at,
        }

    def create_ticket(
        self,
        *,
        customer_id: str,
        subject: str,
        description: str,
    ) -> dict[str, Any]:
        ticket_id = f"tkt_{uuid.uuid4().hex[:8]}"
        created_at = self._now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO tickets (ticket_id, customer_id, subject, description, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, customer_id, subject, description, "open", created_at),
            )
        return {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "subject": subject,
            "description": description,
            "status": "open",
            "created_at": created_at,
        }

    def list_tickets_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ticket_id, customer_id, subject, description, status, created_at "
            "FROM tickets WHERE customer_id = ? ORDER BY created_at DESC",
            (customer_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_faq(self, query: str) -> list[dict[str, Any]]:
        tokens = [tok.strip().lower() for tok in query.split() if len(tok.strip()) >= 2]
        if not tokens:
            rows = self.conn.execute("SELECT faq_id, category, question, answer FROM faq_entries").fetchall()
            return [dict(r) for r in rows]

        rows = self.conn.execute("SELECT faq_id, category, question, answer FROM faq_entries").fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            haystack = f"{row['category']} {row['question']} {row['answer']}".lower()
            if any(tok in haystack for tok in tokens):
                matches.append(dict(row))
        return matches

    def close(self) -> None:
        self.conn.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> SupportSandboxDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
