-- SQLite schema for site-expiry-monitor
-- Location: <project_root>/.skills-data/site-expiry-monitor/data/sites.db

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

