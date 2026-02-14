#!/usr/bin/env python3
"""
Site Expiry Monitor

- Stores a registry of sites in a local SQLite DB under .skills-data/<skill-name>/data/sites.db
- Checks:
  - SSL certificate expiration (notAfter) for domain:443
  - Domain registration expiration via RDAP (best-effort; optional WHOIS fallback)
  - Optional HTTP responsiveness for a URL (expects HTTP 200 by default)
- Prints a compact table; can emit a macOS notification on alerts.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import socket
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IANA_RDAP_DNS_BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
PUBLIC_SUFFIX_LIST_URL = "https://publicsuffix.org/list/public_suffix_list.dat"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sites (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT,
  domain TEXT NOT NULL,
  url TEXT,
  check_http INTEGER NOT NULL DEFAULT 1,
  expected_status INTEGER NOT NULL DEFAULT 200,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_checked_at TEXT,
  last_http_status INTEGER,
  last_http_ok INTEGER,
  last_ssl_expiry TEXT,
  last_domain_expiry TEXT,
  last_error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sites_domain ON sites(domain);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def parse_skill_name_from_skill_md(skill_root: Path) -> Optional[str]:
    skill_md = skill_root / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def guess_project_root(skill_root: Path) -> Path:
    resolved = skill_root.resolve()
    if resolved.parent.name == "skills" and len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


def read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class Paths:
    skill_root: Path
    project_root: Path
    skill_name: str
    data_dir: Path
    db_path: Path
    rdap_bootstrap_cache: Path
    public_suffix_list_cache: Path


def resolve_paths() -> Paths:
    skill_root = Path(__file__).resolve().parents[1]
    skill_name = (
        os.environ.get("SKILL_NAME")
        or read_env_file(skill_root / ".env").get("SKILL_NAME")
        or parse_skill_name_from_skill_md(skill_root)
        or skill_root.name
    )

    data_dir_env = os.environ.get("SKILL_DATA_DIR") or read_env_file(skill_root / ".env").get("SKILL_DATA_DIR")
    if data_dir_env:
        data_dir = Path(data_dir_env).expanduser().resolve()
        project_root = data_dir.parent.parent if data_dir.parent.name == ".skills-data" else guess_project_root(skill_root)
    else:
        project_root = guess_project_root(skill_root)
        data_dir = (project_root / ".skills-data" / skill_name).resolve()

    db_path = data_dir / "data" / "sites.db"
    rdap_bootstrap_cache = data_dir / "cache" / "rdap_dns.json"
    public_suffix_list_cache = data_dir / "cache" / "public_suffix_list.dat"
    return Paths(
        skill_root=skill_root,
        project_root=project_root,
        skill_name=skill_name,
        data_dir=data_dir,
        db_path=db_path,
        rdap_bootstrap_cache=rdap_bootstrap_cache,
        public_suffix_list_cache=public_suffix_list_cache,
    )


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower().rstrip(".")
    if not domain:
        raise ValueError("Domain is empty")
    if "://" in domain or "/" in domain:
        raise ValueError(f"Domain must be a hostname, not a URL: {domain}")
    if any(c.isspace() for c in domain):
        raise ValueError(f"Domain contains whitespace: {domain}")
    return domain.encode("idna").decode("ascii")


def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"URL must be http/https and include a host: {url}")
    return url


def domain_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"URL missing host: {url}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL missing hostname: {url}")
    return normalize_domain(host)


def upsert_site(
    conn: sqlite3.Connection,
    *,
    domain: str,
    url: Optional[str],
    label: Optional[str],
    check_http: bool,
    expected_status: int,
) -> None:
    now = isoformat(utc_now())
    cur = conn.execute("SELECT id FROM sites WHERE domain = ?", (domain,))
    row = cur.fetchone()
    if row:
        conn.execute(
            """
            UPDATE sites
            SET label = ?, url = ?, check_http = ?, expected_status = ?, updated_at = ?
            WHERE domain = ?
            """,
            (label, url, int(check_http), int(expected_status), now, domain),
        )
    else:
        conn.execute(
            """
            INSERT INTO sites (label, domain, url, check_http, expected_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (label, domain, url, int(check_http), int(expected_status), now, now),
        )
    conn.commit()


def delete_site(conn: sqlite3.Connection, *, site_id: Optional[int], domain: Optional[str]) -> int:
    if site_id is not None:
        cur = conn.execute("DELETE FROM sites WHERE id = ?", (site_id,))
    elif domain:
        cur = conn.execute("DELETE FROM sites WHERE domain = ?", (domain,))
    else:
        raise ValueError("Provide --id or --domain")
    conn.commit()
    return int(cur.rowcount or 0)


def fetch_url_status(url: str, timeout_s: float) -> Tuple[Optional[int], Optional[str]]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "site-expiry-monitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.getcode()), None
    except urllib.error.HTTPError as e:
        return int(e.code), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_ssl_not_after(not_after: str) -> Optional[datetime]:
    s = re.sub(r"\s+", " ", not_after.strip())
    try:
        dt = datetime.strptime(s, "%b %d %H:%M:%S %Y %Z")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def fetch_ssl_expiry(domain: str, timeout_s: float) -> Tuple[Optional[datetime], Optional[str]]:
    addr = (domain, 443)

    # Prefer a verified connection so getpeercert() contains parsed fields like notAfter.
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        with socket.create_connection(addr, timeout=timeout_s) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert() or {}
        not_after = cert.get("notAfter")
        if not not_after:
            raise ValueError("SSL cert missing notAfter")
        dt = parse_ssl_not_after(str(not_after))
        if not dt:
            raise ValueError("SSL cert notAfter unparseable")
        return dt, None
    except Exception as verified_err:
        # Fallback: retrieve the peer cert without verification, then decode it locally.
        try:
            ctx2 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx2.check_hostname = False
            ctx2.verify_mode = ssl.CERT_NONE
            with socket.create_connection(addr, timeout=timeout_s) as sock:
                with ctx2.wrap_socket(sock, server_hostname=domain) as ssock:
                    der = ssock.getpeercert(binary_form=True)
            if not der:
                return None, f"SSL peer cert missing (verify_err={type(verified_err).__name__})"

            decoder = getattr(getattr(ssl, "_ssl", None), "_test_decode_cert", None)
            if not decoder:
                return None, f"Cannot decode cert locally (verify_err={type(verified_err).__name__})"

            pem = ssl.DER_cert_to_PEM_cert(der)
            paths = resolve_paths()
            tmp_dir = paths.data_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    delete=False,
                    dir=str(tmp_dir),
                    prefix="cert_",
                    suffix=".pem",
                    encoding="utf-8",
                ) as f:
                    f.write(pem)
                    tmp_path = f.name
                info = decoder(tmp_path)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            not_after2 = info.get("notAfter") if isinstance(info, dict) else None
            if not not_after2:
                return None, f"SSL cert missing notAfter (verify_err={type(verified_err).__name__})"
            dt2 = parse_ssl_not_after(str(not_after2))
            if not dt2:
                return None, f"SSL cert notAfter unparseable (verify_err={type(verified_err).__name__})"
            return dt2, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"


def load_public_suffix_list(
    cache_path: Path,
    *,
    max_age_s: int = 30 * 24 * 3600,
    timeout_s: float = 10.0,
) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= max_age_s:
                return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass

    req = urllib.request.Request(
        PUBLIC_SUFFIX_LIST_URL,
        headers={"Accept": "text/plain", "User-Agent": "site-expiry-monitor/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        cache_path.write_text(body, encoding="utf-8")
        return body
    except Exception:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        raise


def parse_public_suffix_list(text: str) -> Tuple[set[str], set[str], set[str]]:
    exact: set[str] = set()
    wildcard: set[str] = set()
    exceptions: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("!"):
            exceptions.add(line[1:])
        elif line.startswith("*."):
            wildcard.add(line[2:])
        else:
            exact.add(line)
    return exact, wildcard, exceptions


def load_public_suffix_rules(
    cache_path: Path,
    *,
    max_age_s: int = 30 * 24 * 3600,
    timeout_s: float = 10.0,
) -> Tuple[set[str], set[str], set[str]]:
    return parse_public_suffix_list(load_public_suffix_list(cache_path, max_age_s=max_age_s, timeout_s=timeout_s))


def public_suffix(domain: str, rules: Tuple[set[str], set[str], set[str]]) -> str:
    exact, wildcard, exceptions = rules
    labels = [p for p in domain.split(".") if p]
    if not labels:
        return domain

    best: Optional[str] = None
    best_len = 0
    for i in range(len(labels)):
        candidate = ".".join(labels[i:])
        if candidate in exceptions:
            return ".".join(labels[i + 1 :]) if (i + 1) < len(labels) else candidate

        if candidate in exact:
            n = len(labels) - i
            if n > best_len:
                best = candidate
                best_len = n

        if (i + 1) < len(labels):
            wc_base = ".".join(labels[i + 1 :])
            if wc_base in wildcard:
                n = len(labels) - i
                if n > best_len:
                    best = candidate
                    best_len = n

    if best:
        return best
    return labels[-1]


def registrable_domain(domain: str, *, rules: Optional[Tuple[set[str], set[str], set[str]]]) -> str:
    try:
        ipaddress.ip_address(domain)
        return domain
    except ValueError:
        pass

    labels = [p for p in domain.split(".") if p]
    if len(labels) < 2:
        return domain

    if not rules:
        return ".".join(labels[-2:])

    ps = public_suffix(domain, rules)
    ps_labels = [p for p in ps.split(".") if p]
    if len(labels) <= len(ps_labels):
        return domain

    return ".".join(labels[-(len(ps_labels) + 1) :])


def load_rdap_bootstrap(cache_path: Path, *, max_age_s: int = 7 * 24 * 3600, timeout_s: float = 10.0) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= max_age_s:
                return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    req = urllib.request.Request(IANA_RDAP_DNS_BOOTSTRAP_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    cache_path.write_text(body, encoding="utf-8")
    return json.loads(body)


def rdap_base_for_domain(bootstrap: Dict[str, Any], domain: str) -> Optional[str]:
    # IANA bootstrap: {"services": [ [ ["com","net"], ["https://.../"] ], ... ] }
    tld = domain.rsplit(".", 1)[-1].lower()
    services = bootstrap.get("services") or []
    for svc in services:
        if not isinstance(svc, list) or len(svc) < 2:
            continue
        tlds, urls = svc[0], svc[1]
        if isinstance(tlds, list) and tld in [str(x).lower() for x in tlds]:
            if isinstance(urls, list) and urls:
                return str(urls[0])
    return None


def parse_rdap_event_date(s: str) -> Optional[datetime]:
    # Typically ISO8601; sometimes ends with 'Z'
    s = s.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_domain_expiry_rdap(
    domain: str,
    *,
    bootstrap_cache_path: Path,
    timeout_s: float,
) -> Tuple[Optional[datetime], Optional[str]]:
    try:
        bootstrap = load_rdap_bootstrap(bootstrap_cache_path, timeout_s=timeout_s)
        base = rdap_base_for_domain(bootstrap, domain)
        if not base:
            return None, "RDAP bootstrap missing TLD service"
        base = base if base.endswith("/") else (base + "/")
        url = base + "domain/" + urllib.parse.quote(domain)
        req = urllib.request.Request(url, headers={"Accept": "application/rdap+json, application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        events = data.get("events") or []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            action = str(ev.get("eventAction") or "").lower()
            if action != "expiration":
                continue
            dt = parse_rdap_event_date(str(ev.get("eventDate") or ""))
            if dt:
                return dt, None
        return None, "RDAP response missing expiration event"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


WHOIS_EXPIRY_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"^Registrar Registration Expiration Date:\s*(?P<date>.+)$", re.IGNORECASE),
    re.compile(r"^Registry Expiry Date:\s*(?P<date>.+)$", re.IGNORECASE),
    re.compile(r"^Expiry Date:\s*(?P<date>.+)$", re.IGNORECASE),
    re.compile(r"^Expiration Date:\s*(?P<date>.+)$", re.IGNORECASE),
    re.compile(r"^paid-till:\s*(?P<date>.+)$", re.IGNORECASE),
    re.compile(r"^validity:\s*(?P<date>.+)$", re.IGNORECASE),
]


def parse_whois_date(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.replace("UTC", "Z")
    if raw.endswith("Z"):
        parsed = parse_rdap_event_date(raw)
        if parsed:
            return parsed
    # Common formats
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d-%b-%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def fetch_domain_expiry_whois(domain: str, *, timeout_s: float) -> Tuple[Optional[datetime], Optional[str]]:
    try:
        proc = subprocess.run(
            ["whois", domain],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return None, "whois not installed"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    text = proc.stdout or ""
    for line in text.splitlines():
        line = line.strip()
        for pat in WHOIS_EXPIRY_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            dt = parse_whois_date(m.group("date"))
            if dt:
                return dt, None
    return None, "WHOIS response missing expiry date"


def fetch_domain_expiry(
    domain: str,
    *,
    bootstrap_cache_path: Path,
    timeout_s: float,
    whois_fallback: bool,
) -> Tuple[Optional[datetime], Optional[str], str]:
    dt, err = fetch_domain_expiry_rdap(domain, bootstrap_cache_path=bootstrap_cache_path, timeout_s=timeout_s)
    if dt:
        return dt, None, "rdap"
    if not whois_fallback:
        return None, err, "rdap"
    dt2, err2 = fetch_domain_expiry_whois(domain, timeout_s=timeout_s)
    if dt2:
        return dt2, None, "whois"
    return None, err2 or err, "whois"


def days_left(expiry: Optional[datetime], now: datetime) -> Optional[int]:
    if not expiry:
        return None
    return (expiry.date() - now.date()).days


def format_dt_short(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def clip(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)] + "…"


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = ["id", "domain", "ssl_exp", "dom_exp", "http", "alerts", "last_check"]
    widths: Dict[str, int] = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    sep = "  "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    print(header)
    print(sep.join("-" * widths[c] for c in cols))
    for r in rows:
        line = sep.join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)


def macos_notify(title: str, message: str) -> None:
    if sys.platform != "darwin":
        return
    msg = message.replace("\\", "\\\\").replace('"', '\\"')
    ttl = title.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{msg}" with title "{ttl}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    except Exception:
        return


def select_sites(conn: sqlite3.Connection, *, site_id: Optional[int], domain: Optional[str]) -> List[sqlite3.Row]:
    if site_id is not None:
        cur = conn.execute("SELECT * FROM sites WHERE id = ? ORDER BY id", (site_id,))
    elif domain:
        cur = conn.execute("SELECT * FROM sites WHERE domain = ? ORDER BY id", (domain,))
    else:
        cur = conn.execute("SELECT * FROM sites ORDER BY id")
    return list(cur.fetchall())


def update_check_result(
    conn: sqlite3.Connection,
    *,
    site_id: int,
    checked_at: str,
    http_status: Optional[int],
    http_ok: Optional[bool],
    ssl_expiry: Optional[datetime],
    domain_expiry: Optional[datetime],
    error: Optional[str],
) -> None:
    conn.execute(
        """
        UPDATE sites
        SET last_checked_at = ?,
            last_http_status = ?,
            last_http_ok = ?,
            last_ssl_expiry = ?,
            last_domain_expiry = ?,
            last_error = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            checked_at,
            http_status,
            None if http_ok is None else int(http_ok),
            isoformat(ssl_expiry),
            isoformat(domain_expiry),
            error,
            checked_at,
            site_id,
        ),
    )


def run_check(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    threshold_days: int,
    timeout_s: float,
    whois_fallback: bool,
    paths: Paths,
    psl_rules: Optional[Tuple[set[str], set[str], set[str]]],
) -> Tuple[Dict[str, Any], List[str]]:
    now = utc_now()
    checked_at = isoformat(now) or ""
    domain = str(row["domain"])
    label = str(row["label"] or "")
    url = str(row["url"] or f"https://{domain}/")
    check_http = bool(int(row["check_http"]))
    expected_status = int(row["expected_status"])

    alerts: List[str] = []
    errors: List[str] = []

    ssl_exp, ssl_err = fetch_ssl_expiry(domain, timeout_s=timeout_s)
    if ssl_err:
        errors.append(f"ssl: {ssl_err}")
        alerts.append("SSL ERR")
    ssl_days = days_left(ssl_exp, now)
    if ssl_days is not None and ssl_days < threshold_days:
        alerts.append(f"SSL {ssl_days}d")

    reg_domain = registrable_domain(domain, rules=psl_rules)
    dom_exp, dom_err, dom_src = fetch_domain_expiry(
        reg_domain,
        bootstrap_cache_path=paths.rdap_bootstrap_cache,
        timeout_s=timeout_s,
        whois_fallback=whois_fallback,
    )
    if dom_err:
        errors.append(f"domain({reg_domain},{dom_src}): {dom_err}")
    dom_days = days_left(dom_exp, now)
    if dom_days is not None and dom_days < threshold_days:
        alerts.append(f"DOMAIN {dom_days}d")

    http_status: Optional[int] = None
    http_ok: Optional[bool] = None
    http_err: Optional[str] = None
    if check_http:
        http_status, http_err = fetch_url_status(url, timeout_s=timeout_s)
        http_ok = http_status == expected_status
        if http_err:
            errors.append(f"http: {http_err}")
        if http_status is None:
            alerts.append("HTTP ERR")
        elif not http_ok:
            alerts.append(f"HTTP {http_status}")

    error_str = "; ".join(errors) if errors else None
    update_check_result(
        conn,
        site_id=int(row["id"]),
        checked_at=checked_at,
        http_status=http_status,
        http_ok=http_ok,
        ssl_expiry=ssl_exp,
        domain_expiry=dom_exp,
        error=error_str,
    )
    conn.commit()

    out = {
        "id": int(row["id"]),
        "label": clip(label or "-", 24),
        "domain": domain,
        "ssl_exp": format_dt_short(ssl_exp),
        "dom_exp": format_dt_short(dom_exp),
        "http": "-" if not check_http else ("ERR" if http_status is None else str(http_status)),
        "alerts": ",".join(alerts) if alerts else "-",
        "last_check": checked_at.split("T", 1)[0] if checked_at else "-",
    }
    return out, alerts


def cmd_init(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    with connect_db(paths.db_path) as conn:
        init_db(conn)
    print(f"DB ready: {paths.db_path}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    domain: Optional[str] = args.domain
    url: Optional[str] = args.url

    if url and not domain:
        domain = domain_from_url(normalize_url(url))
    if domain:
        domain = normalize_domain(domain)
    else:
        raise SystemExit("Provide --domain or --url")

    if url:
        url = normalize_url(url)

    with connect_db(paths.db_path) as conn:
        init_db(conn)
        upsert_site(
            conn,
            domain=domain,
            url=url,
            label=args.label,
            check_http=not args.no_http,
            expected_status=int(args.expected_status),
        )
    print(f"Upserted: {domain}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    with connect_db(paths.db_path) as conn:
        init_db(conn)
        n = delete_site(conn, site_id=args.id, domain=normalize_domain(args.domain) if args.domain else None)
    print(f"Removed: {n}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    with connect_db(paths.db_path) as conn:
        init_db(conn)
        rows = select_sites(conn, site_id=args.id, domain=normalize_domain(args.domain) if args.domain else None)

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        out_rows.append(
            {
                "id": int(r["id"]),
                "label": clip(str(r["label"] or "-"), 24),
                "domain": str(r["domain"]),
                "ssl_exp": (str(r["last_ssl_expiry"] or "-")[:10]) if r["last_ssl_expiry"] else "-",
                "dom_exp": (str(r["last_domain_expiry"] or "-")[:10]) if r["last_domain_expiry"] else "-",
                "http": "-"
                if not bool(int(r["check_http"]))
                else ("ERR" if r["last_http_status"] is None else str(int(r["last_http_status"]))),
                "alerts": "-",
                "last_check": (str(r["last_checked_at"] or "-").split("T", 1)[0]),
            }
        )

    if args.json:
        print(json.dumps(out_rows, indent=2))
    else:
        print_table(out_rows)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    paths = resolve_paths()
    threshold_days = int(args.threshold_days)
    timeout_s = float(args.timeout_s)
    whois_fallback = bool(args.whois_fallback)
    try:
        psl_rules = load_public_suffix_rules(paths.public_suffix_list_cache, timeout_s=timeout_s)
    except Exception:
        psl_rules = None

    with connect_db(paths.db_path) as conn:
        init_db(conn)
        rows = select_sites(conn, site_id=args.id, domain=normalize_domain(args.domain) if args.domain else None)
        if not rows:
            print("No sites found. Add one with: python3 scripts/site_expiry_monitor.py add --domain example.com")
            return 2

        table_rows: List[Dict[str, Any]] = []
        all_alerts: List[Tuple[str, List[str]]] = []
        for r in rows:
            tr, alerts = run_check(
                conn,
                r,
                threshold_days=threshold_days,
                timeout_s=timeout_s,
                whois_fallback=whois_fallback,
                paths=paths,
                psl_rules=psl_rules,
            )
            table_rows.append(tr)
            if alerts:
                all_alerts.append((str(r["domain"]), alerts))

    if args.json:
        print(json.dumps(table_rows, indent=2))
    else:
        print_table(table_rows)

    if args.notify and all_alerts:
        parts: List[str] = []
        for dom, alerts in all_alerts[:5]:
            parts.append(f"{dom}: {'/'.join(alerts)}")
        msg = "; ".join(parts)
        if len(all_alerts) > 5:
            msg += f"; +{len(all_alerts) - 5} more"
        macos_notify("Site Expiry Monitor", clip(msg, 220))

    return 1 if all_alerts else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="site_expiry_monitor.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_init = sub.add_parser("init", help="Create/upgrade the local SQLite database")
    s_init.set_defaults(fn=cmd_init)

    s_add = sub.add_parser("add", help="Add or update a tracked site")
    s_add.add_argument("--domain", default="", help="Hostname only (example.com)")
    s_add.add_argument("--url", default="", help="Optional URL (https://example.com/)")
    s_add.add_argument("--label", default="", help="Optional display label")
    s_add.add_argument("--no-http", action="store_true", help="Disable HTTP status check for this site")
    s_add.add_argument("--expected-status", default="200", help="Expected HTTP status (default: 200)")
    s_add.set_defaults(fn=cmd_add)

    s_rm = sub.add_parser("remove", help="Remove a tracked site")
    s_rm.add_argument("--id", type=int, default=None, help="Site id")
    s_rm.add_argument("--domain", default="", help="Domain")
    s_rm.set_defaults(fn=cmd_remove)

    s_ls = sub.add_parser("list", help="List tracked sites (table view)")
    s_ls.add_argument("--id", type=int, default=None, help="Site id")
    s_ls.add_argument("--domain", default="", help="Domain")
    s_ls.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    s_ls.set_defaults(fn=cmd_list)

    s_ck = sub.add_parser("check", help="Run checks and update the DB")
    s_ck.add_argument("--id", type=int, default=None, help="Site id")
    s_ck.add_argument("--domain", default="", help="Domain")
    s_ck.add_argument("--threshold-days", default="10", help="Alert threshold in days (default: 10)")
    s_ck.add_argument("--timeout-s", default="12", help="Per-check timeout in seconds (default: 12)")
    s_ck.add_argument("--whois-fallback", action="store_true", help="Try `whois` if RDAP cannot find domain expiry")
    s_ck.add_argument("--notify", action="store_true", help="Send a macOS notification if alerts exist")
    s_ck.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    s_ck.set_defaults(fn=cmd_check)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
