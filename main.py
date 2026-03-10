#!/usr/bin/env python3
"""Serve the research site with authenticated global HTML editing."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
SITE_ROOT = ROOT / "site"
ADMIN_ROOT = SITE_ROOT / "admin"
DB_PATH = ADMIN_ROOT / "editor.db"

SESSION_COOKIE = "research_site_session"
DEFAULT_PASSWORD = "changeme"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_site_path(raw_path: str) -> str:
    path = unquote(raw_path.strip())
    if not path.startswith("/"):
        path = "/" + path
    if path.endswith("/"):
        path = path + "index.html"
    if Path(path).suffix == "":
        path = path.rstrip("/") + "/index.html"
    return path


def resolve_site_file(path: str) -> Path:
    safe_rel = path.lstrip("/")
    candidate = (SITE_ROOT / safe_rel).resolve()
    if SITE_ROOT not in candidate.parents and candidate != SITE_ROOT:
        raise ValueError("Path escapes site root")
    return candidate


def init_db() -> None:
    ADMIN_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content_overrides (
                path TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_override(path: str) -> str | None:
    with db_conn() as conn:
        row = conn.execute("SELECT html FROM content_overrides WHERE path = ?", (path,)).fetchone()
    return row["html"] if row else None


def list_html_pages() -> list[str]:
    paths: list[str] = []
    for file_path in SITE_ROOT.rglob("*.html"):
        if ADMIN_ROOT in file_path.parents:
            continue
        rel = "/" + file_path.relative_to(SITE_ROOT).as_posix()
        paths.append(rel)
    return sorted(paths)


class ResearchSiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET,POST,PUT,DELETE,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed)
            return

        target_path = normalize_site_path(parsed.path)
        if target_path.endswith(".html"):
            self.serve_html_with_override(target_path)
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/login":
            self.handle_login()
            return
        if parsed.path == "/api/logout":
            self.handle_logout()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/content":
            if not self.require_auth():
                return
            self.handle_put_content()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/content":
            if not self.require_auth():
                return
            self.handle_delete_content(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def parse_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def get_session_token(self) -> str | None:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else None

    def require_auth(self) -> bool:
        token = self.get_session_token()
        if not token:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "Authentication required"})
            return False
        with db_conn() as conn:
            row = conn.execute("SELECT token FROM sessions WHERE token = ?", (token,)).fetchone()
        if not row:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid session"})
            return False
        return True

    def handle_login(self) -> None:
        expected = os.environ.get("RESEARCH_SITE_ADMIN_PASSWORD", DEFAULT_PASSWORD)
        data = self.parse_json_body()
        provided = data.get("password", "")
        if not isinstance(provided, str) or provided != expected:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "Invalid password"})
            return

        token = secrets.token_urlsafe(32)
        with db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions(token, created_at) VALUES(?, ?)",
                (token, utc_now()),
            )
            conn.commit()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}={token}; HttpOnly; Path=/; SameSite=Lax",
        )
        payload = json.dumps({"ok": True}).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_logout(self) -> None:
        token = self.get_session_token()
        if token:
            with db_conn() as conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}=; HttpOnly; Path=/; Max-Age=0; SameSite=Lax",
        )
        payload = json.dumps({"ok": True}).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_api_get(self, parsed) -> None:
        if parsed.path == "/api/pages":
            if not self.require_auth():
                return
            self.send_json(HTTPStatus.OK, {"pages": list_html_pages()})
            return

        if parsed.path == "/api/content":
            if not self.require_auth():
                return
            query = parse_qs(parsed.query)
            requested = query.get("path", [""])[0]
            target_path = normalize_site_path(requested or "/index.html")
            try:
                file_path = resolve_site_file(target_path)
            except ValueError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid path"})
                return
            if not file_path.exists():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Page not found"})
                return

            base_html = file_path.read_text(encoding="utf-8")
            override_html = get_override(target_path)
            payload = {
                "path": target_path,
                "has_override": override_html is not None,
                "base_html": base_html,
                "effective_html": override_html if override_html is not None else base_html,
            }
            self.send_json(HTTPStatus.OK, payload)
            return

        if parsed.path == "/api/health":
            self.send_json(HTTPStatus.OK, {"ok": True, "utc": utc_now()})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_put_content(self) -> None:
        data = self.parse_json_body()
        raw_path = data.get("path")
        html = data.get("html")
        if not isinstance(raw_path, str) or not isinstance(html, str):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Expected string path and html"})
            return

        target_path = normalize_site_path(raw_path)
        try:
            file_path = resolve_site_file(target_path)
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid path"})
            return
        if not file_path.exists():
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Page not found"})
            return

        with db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO content_overrides(path, html, updated_at) VALUES(?, ?, ?)",
                (target_path, html, utc_now()),
            )
            conn.commit()
        self.send_json(HTTPStatus.OK, {"ok": True, "path": target_path})

    def handle_delete_content(self, parsed) -> None:
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        if not raw_path:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing path"})
            return
        target_path = normalize_site_path(raw_path)
        with db_conn() as conn:
            conn.execute("DELETE FROM content_overrides WHERE path = ?", (target_path,))
            conn.commit()
        self.send_json(HTTPStatus.OK, {"ok": True, "path": target_path})

    def serve_html_with_override(self, target_path: str) -> None:
        try:
            file_path = resolve_site_file(target_path)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        override_html = get_override(target_path)
        html = override_html if override_html is not None else file_path.read_text(encoding="utf-8")
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    init_db()
    host = os.environ.get("RESEARCH_SITE_HOST", "127.0.0.1")
    port = int(os.environ.get("RESEARCH_SITE_PORT", "8000"))
    password = os.environ.get("RESEARCH_SITE_ADMIN_PASSWORD")
    if not password:
        print(
            "WARNING: RESEARCH_SITE_ADMIN_PASSWORD is not set. "
            f"Using insecure default password: {DEFAULT_PASSWORD}"
        )
    server = ThreadingHTTPServer((host, port), ResearchSiteHandler)
    print(f"Serving research site on http://{host}:{port}")
    print("Admin editor: /admin/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
