#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SKILL_NAME = "goals-habits-tracker"
SCHEMA_VERSION = 2


def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _print_json(obj: Any) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _die(message: str, *, code: int = 1) -> None:
    _print_json({"ok": False, "error": message})
    raise SystemExit(code)


def _project_root_from_script() -> Path:
    # .../.codex/skills/<skill>/scripts/goals_cli.py -> project root is 4 parents up.
    return Path(__file__).resolve().parents[4]


def _skill_root() -> Path:
    # .../.codex/skills/<skill>/scripts/goals_cli.py -> skill root is 2 parents up.
    return Path(__file__).resolve().parents[1]


def _skill_data_dir() -> Path:
    project_root = _project_root_from_script()
    return project_root / ".skills-data" / SKILL_NAME


def _default_db_path() -> Path:
    return _skill_data_dir() / "goals.sqlite3"


def _db_path() -> Path:
    raw = os.environ.get("GOALS_DB")
    return Path(raw).expanduser() if raw else _default_db_path()


def _ensure_dirs() -> None:
    d = _skill_data_dir()
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "cache" / "pycache").mkdir(parents=True, exist_ok=True)
    (d / "tmp").mkdir(parents=True, exist_ok=True)


def _env_path() -> Path:
    return _skill_data_dir() / ".env"


def _config_path() -> Path:
    return _skill_data_dir() / "config.json"


def _read_env() -> dict[str, str]:
    """
    Parse a very small subset of dotenv:
    - KEY=value
    - KEY="value"
    Lines that don't match are ignored. This is intentionally minimal.
    """
    p = _env_path()
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        out[k] = v
    return out


def _quote_env_value(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _upsert_env(updates: dict[str, str]) -> None:
    """
    Update (or append) KEY="value" entries in .env while preserving unknown lines.
    If .env doesn't exist yet, write a minimal file (setup.sh normally creates it).
    """
    p = _env_path()
    existing_lines: list[str] = []
    if p.exists():
        existing_lines = p.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out_lines: list[str] = []

    for raw in existing_lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", raw.strip())
        if not m:
            out_lines.append(raw)
            continue
        key = m.group(1)
        if key in updates:
            out_lines.append(f"{key}={_quote_env_value(updates[key])}")
            seen.add(key)
        else:
            out_lines.append(raw)

    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")

    for k in sorted(updates.keys()):
        if k in seen:
            continue
        out_lines.append(f"{k}={_quote_env_value(updates[k])}")

    if not out_lines:
        for k in sorted(updates.keys()):
            out_lines.append(f"{k}={_quote_env_value(updates[k])}")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out_lines).rstrip("\n") + "\n", encoding="utf-8")


def _read_config() -> dict[str, Any]:
    p = _config_path()
    if not p.exists():
        return {"areas": [], "onboarded": False, "onboarded_at": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"areas": [], "onboarded": False, "onboarded_at": None}
        return data
    except json.JSONDecodeError:
        return {"areas": [], "onboarded": False, "onboarded_at": None}


def _write_config(cfg: dict[str, Any]) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    _ensure_dirs()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA user_version;")
    user_version = int(cur.fetchone()[0])
    if user_version == SCHEMA_VERSION:
        return
    if user_version not in (0, 1):
        _die(
            f"Unsupported DB schema version: {user_version} (expected {SCHEMA_VERSION})"
        )

    if user_version == 0:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS goals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT NOT NULL,
              horizon TEXT NOT NULL CHECK (horizon IN ('year','quarter')),
              start_date TEXT NOT NULL,
              end_date TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','completed','abandoned')),
              parent_goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
              area TEXT,
              focus TEXT,
              why TEXT,
              success_criteria TEXT,
              notes TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS goals_horizon_idx ON goals(horizon);
            CREATE INDEX IF NOT EXISTS goals_status_idx ON goals(status);
            CREATE INDEX IF NOT EXISTS goals_parent_idx ON goals(parent_goal_id);

            CREATE TABLE IF NOT EXISTS goal_updates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
              date TEXT NOT NULL,
              progress_note TEXT,
              progress_value REAL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS goal_updates_goal_idx ON goal_updates(goal_id);
            CREATE INDEX IF NOT EXISTS goal_updates_date_idx ON goal_updates(date);

            CREATE TABLE IF NOT EXISTS habits (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              kind TEXT NOT NULL CHECK (kind IN ('build','quit')),
              cadence TEXT NOT NULL CHECK (cadence IN ('daily','weekly')),
              target INTEGER NOT NULL,
              unit TEXT NOT NULL DEFAULT 'times',
              start_date TEXT NOT NULL,
              end_date TEXT,
              status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','archived')),
              area TEXT,
              focus TEXT,
              notes TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS habits_status_idx ON habits(status);

            CREATE TABLE IF NOT EXISTS habit_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              habit_id INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
              date TEXT NOT NULL,
              value INTEGER NOT NULL,
              note TEXT,
              created_at TEXT NOT NULL,
              UNIQUE(habit_id, date)
            );
            CREATE INDEX IF NOT EXISTS habit_logs_date_idx ON habit_logs(date);

            CREATE TABLE IF NOT EXISTS goal_habits (
              goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
              habit_id INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
              PRIMARY KEY (goal_id, habit_id)
            );

            CREATE TABLE IF NOT EXISTS checkins (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              period TEXT NOT NULL CHECK (period IN ('week','quarter','year')),
              start_date TEXT NOT NULL,
              end_date TEXT NOT NULL,
              summary TEXT,
              data_json TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS checkins_period_start_idx ON checkins(period, start_date);
            """
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
        conn.commit()
        return

    # v1 -> v2 migration: add optional area/focus columns
    conn.execute("ALTER TABLE goals ADD COLUMN area TEXT;")
    conn.execute("ALTER TABLE goals ADD COLUMN focus TEXT;")
    conn.execute("ALTER TABLE habits ADD COLUMN area TEXT;")
    conn.execute("ALTER TABLE habits ADD COLUMN focus TEXT;")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


_REL_DAY_RE = re.compile(r"^(?P<sign>[+-])(?P<num>\d+)d$")


def _parse_date(raw: str) -> date:
    raw = raw.strip().lower()
    today = date.today()
    if raw == "today":
        return today
    if raw == "yesterday":
        return today - timedelta(days=1)
    if raw == "tomorrow":
        return today + timedelta(days=1)
    m = _REL_DAY_RE.match(raw)
    if m:
        n = int(m.group("num"))
        if m.group("sign") == "-":
            n = -n
        return today + timedelta(days=n)
    try:
        return date.fromisoformat(raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid date '{raw}'. Use YYYY-MM-DD, today, yesterday, tomorrow, or +/-Nd."
        ) from e


def _iso(d: date) -> str:
    return d.isoformat()


def _week_start(d: date) -> date:
    env = _read_env()
    ws = (env.get("WEEK_START") or "mon").strip().lower()
    if ws == "sun":
        # Python weekday(): Monday=0..Sunday=6. Want Sunday-start.
        offset = (d.weekday() + 1) % 7
        return d - timedelta(days=offset)
    return d - timedelta(days=d.weekday())  # Monday-start


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(d: date, months: int) -> date:
    if months == 0:
        return d
    m0 = (d.month - 1) + months
    year = d.year + (m0 // 12)
    month = (m0 % 12) + 1
    day = min(d.day, _last_day_of_month(year, month))
    return date(year, month, day)


def _get_goal(conn: sqlite3.Connection, goal_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()


def _get_habit(conn: sqlite3.Connection, habit_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM habits WHERE id = ?", (habit_id,)).fetchone()


def _cmd_init(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)
    cur = conn.execute("PRAGMA user_version;")
    version = int(cur.fetchone()[0])
    _print_json({"ok": True, "db_path": str(db_path), "schema_version": version})


@dataclass(frozen=True)
class _Period:
    start: date
    end: date


def _habit_range_summary(
    conn: sqlite3.Connection, start: date, end: date, *, status: str = "active"
) -> dict[str, Any]:
    """
    Summarize habits over an inclusive date range.
    Note: habits are logged per day; for weekly habits we treat the target as per-week and
    multiply by the number of distinct weeks touched by the range (based on WEEK_START).
    """

    habits_sql = "SELECT * FROM habits"
    params: list[Any] = []
    if status != "all":
        habits_sql += " WHERE status = ?"
        params.append(status)
    habits_sql += " ORDER BY name COLLATE NOCASE ASC, id ASC"
    habits = _rows_to_dicts(conn.execute(habits_sql, params).fetchall())

    habit_ids = [h["id"] for h in habits]
    logs_by_habit: dict[int, list[dict[str, Any]]] = {hid: [] for hid in habit_ids}
    if habit_ids:
        q_marks = ",".join(["?"] * len(habit_ids))
        rows = conn.execute(
            f"""
            SELECT * FROM habit_logs
            WHERE habit_id IN ({q_marks})
              AND date >= ?
              AND date <= ?
            ORDER BY date ASC, id ASC
            """,
            [*habit_ids, _iso(start), _iso(end)],
        ).fetchall()
        for r in rows:
            logs_by_habit[int(r["habit_id"])].append(dict(r))

    # Count weeks touched by the range (for weekly target scaling).
    # Use the current WEEK_START preference to define week boundaries.
    wk = _week_start(start)
    week_starts: list[date] = []
    while wk <= end:
        week_starts.append(wk)
        wk = wk + timedelta(days=7)
    weeks_touched = max(1, len(week_starts))

    days_in_range = (end - start).days + 1

    summaries: list[dict[str, Any]] = []
    for h in habits:
        hid = int(h["id"])
        logs = logs_by_habit.get(hid, [])
        kind = h["kind"]
        cadence = h["cadence"]
        target = int(h["target"])

        if kind == "build":
            if cadence == "daily":
                target_total = target * days_in_range
            else:
                target_total = target * weeks_touched
            actual = sum(int(x["value"]) for x in logs)
            pct = (
                None if target_total <= 0 else round((actual / target_total) * 100.0, 1)
            )
            summaries.append(
                {
                    "habit": h,
                    "period": {"start": _iso(start), "end": _iso(end)},
                    "target_total": target_total,
                    "actual_total": actual,
                    "completion_pct": pct,
                    "logs": logs,
                }
            )
        else:
            if cadence == "daily":
                target_total = target * days_in_range
            else:
                target_total = target * weeks_touched
            occurrences = sum(int(x["value"]) for x in logs)
            days_logged = len(logs)
            days_slipped = sum(1 for x in logs if int(x["value"]) > 0)
            summaries.append(
                {
                    "habit": h,
                    "period": {"start": _iso(start), "end": _iso(end)},
                    "target_total": target_total,
                    "occurrences_logged": occurrences,
                    "days_logged": days_logged,
                    "days_slipped": days_slipped,
                    "within_target_assuming_missing_zero": occurrences <= target_total,
                    "logs": logs,
                }
            )

    return {
        "period": {"start": _iso(start), "end": _iso(end)},
        "days_in_period": days_in_range,
        "weeks_touched": weeks_touched,
        "habits": summaries,
    }


def _habit_week_summary(
    conn: sqlite3.Connection, week_start: date, *, status: str = "active"
) -> dict[str, Any]:
    week_end = week_start + timedelta(days=6)
    summary = _habit_range_summary(conn, week_start, week_end, status=status)
    # Keep backward-compatible shape for existing callers.
    return {
        "week": {"start": _iso(week_start), "end": _iso(week_end)},
        "habits": summary["habits"],
    }


def _autolink_orphan_quarter_goals(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Ensure legacy quarter goals without a parent get linked (or get a created year parent),
    so rollups work consistently over time.
    """
    orphan_rows = _rows_to_dicts(
        conn.execute(
            """
            SELECT id, title, start_date, area, focus
            FROM goals
            WHERE horizon = 'quarter' AND parent_goal_id IS NULL
            ORDER BY start_date ASC, id ASC
            """
        ).fetchall()
    )
    if not orphan_rows:
        return {"checked": 0, "updated": 0}

    ts = _now_utc_iso()
    stats: dict[str, Any] = {
        "checked": len(orphan_rows),
        "updated": 0,
        "by_resolution": {},
        "created_parent_ids": [],
    }

    for row in orphan_rows:
        qid = int(row["id"])
        qtitle = str(row.get("title") or "").strip()
        qstart = _parse_date(str(row.get("start_date") or ""))
        qarea = row.get("area")
        qfocus = row.get("focus")

        parent_id, resolution, _candidates = _ensure_year_parent_for_quarter_goal(
            conn,
            quarter_start=qstart,
            area=str(qarea) if qarea is not None else None,
            focus=str(qfocus) if qfocus is not None else None,
            quarter_title=qtitle,
        )
        conn.execute(
            "UPDATE goals SET parent_goal_id = ?, updated_at = ? WHERE id = ?",
            (int(parent_id), ts, qid),
        )
        stats["updated"] += 1
        stats["by_resolution"][resolution] = (
            int(stats["by_resolution"].get(resolution, 0)) + 1
        )
        if resolution in ("created_missing", "created_ambiguous"):
            stats["created_parent_ids"].append(int(parent_id))

    conn.commit()
    return stats


def _cmd_dashboard(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    autolink_stats = _autolink_orphan_quarter_goals(conn)

    today = date.today()
    wk_start = _week_start(today)
    wk_end = wk_start + timedelta(days=6)

    goals = _rows_to_dicts(
        conn.execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY end_date ASC, id ASC"
        ).fetchall()
    )
    habits = _rows_to_dicts(
        conn.execute(
            "SELECT * FROM habits WHERE status = 'active' ORDER BY name COLLATE NOCASE ASC, id ASC"
        ).fetchall()
    )
    links = _rows_to_dicts(
        conn.execute(
            "SELECT * FROM goal_habits ORDER BY goal_id ASC, habit_id ASC"
        ).fetchall()
    )

    latest_checkins: dict[str, Any] = {}
    for period in ("week", "quarter", "year"):
        row = conn.execute(
            "SELECT * FROM checkins WHERE period = ? ORDER BY start_date DESC, id DESC LIMIT 1",
            (period,),
        ).fetchone()
        latest_checkins[period] = _row_to_dict(row)

    habit_week = _habit_week_summary(conn, wk_start, status="active")

    _print_json(
        {
            "ok": True,
            "db_path": str(db_path),
            "today": _iso(today),
            "week": {"start": _iso(wk_start), "end": _iso(wk_end)},
            "quarter_parent_autolink": autolink_stats,
            "goals": goals,
            "habits": habits,
            "goal_habits": links,
            "habit_week": habit_week,
            "latest_checkins": latest_checkins,
        }
    )


def _cmd_bootstrap(args: argparse.Namespace) -> None:
    """
    Ensure the DB schema exists and return a small "status" object that the agent
    can use to decide whether to run onboarding questions in chat.
    """
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    cfg = _read_config()
    changed_cfg = False
    if not isinstance(cfg.get("areas"), list):
        cfg["areas"] = []
        changed_cfg = True
    if not isinstance(cfg.get("onboarded"), bool):
        cfg["onboarded"] = False
        changed_cfg = True
    if "onboarded_at" not in cfg:
        cfg["onboarded_at"] = None
        changed_cfg = True
    if changed_cfg or not _config_path().exists():
        _write_config(cfg)

    env = _read_env()
    # If setup.sh didn't run yet (or .env is missing keys), ensure baseline keys exist.
    project_root = _project_root_from_script()
    baseline: dict[str, str] = {
        "SKILL_ROOT": str(_skill_root()),
        "SKILL_NAME": SKILL_NAME,
        "SKILL_DATA_ROOT": str(project_root / ".skills-data"),
        "SKILL_DATA_DIR": str(_skill_data_dir()),
        "GOALS_DB": str(db_path),
        "WEEK_START": "mon",
        "TIMEZONE": "",
        "QUIT_HABIT_TRACKING": "occurrences",
        "WEEKLY_CHECKIN_DAY": "",
        "WEEKLY_CHECKIN_TIME": "",
    }
    missing = {k: v for k, v in baseline.items() if k not in env}
    if missing:
        _upsert_env(missing)
        env = _read_env()

    goal_count = int(conn.execute("SELECT COUNT(1) FROM goals").fetchone()[0])
    habit_count = int(conn.execute("SELECT COUNT(1) FROM habits").fetchone()[0])
    checkin_count = int(conn.execute("SELECT COUNT(1) FROM checkins").fetchone()[0])
    is_empty = goal_count == 0 and habit_count == 0 and checkin_count == 0
    needs_onboarding = (not bool(cfg.get("onboarded", False))) and is_empty

    _print_json(
        {
            "ok": True,
            "db_path": str(db_path),
            "schema_version": SCHEMA_VERSION,
            "counts": {
                "goals": goal_count,
                "habits": habit_count,
                "checkins": checkin_count,
            },
            "is_empty": is_empty,
            "needs_onboarding": needs_onboarding,
            "prefs": {
                "week_start": (env.get("WEEK_START") or "mon"),
                "timezone": (env.get("TIMEZONE") or ""),
                "quit_habit_tracking": (
                    env.get("QUIT_HABIT_TRACKING") or "occurrences"
                ),
                "weekly_checkin_day": (env.get("WEEKLY_CHECKIN_DAY") or ""),
                "weekly_checkin_time": (env.get("WEEKLY_CHECKIN_TIME") or ""),
            },
            "areas": cfg.get("areas", []),
            "onboarded": bool(cfg.get("onboarded", False)),
            "onboarded_at": cfg.get("onboarded_at", None),
        }
    )


def _cmd_prefs_get(args: argparse.Namespace) -> None:
    env = _read_env()
    _print_json(
        {
            "ok": True,
            "prefs": {
                "week_start": (env.get("WEEK_START") or "mon"),
                "timezone": (env.get("TIMEZONE") or ""),
                "quit_habit_tracking": (
                    env.get("QUIT_HABIT_TRACKING") or "occurrences"
                ),
                "weekly_checkin_day": (env.get("WEEKLY_CHECKIN_DAY") or ""),
                "weekly_checkin_time": (env.get("WEEKLY_CHECKIN_TIME") or ""),
            },
        }
    )


def _cmd_prefs_set(args: argparse.Namespace) -> None:
    updates: dict[str, str] = {}
    if args.week_start:
        updates["WEEK_START"] = args.week_start
    if args.timezone is not None:
        updates["TIMEZONE"] = args.timezone
    if args.quit_habit_tracking:
        updates["QUIT_HABIT_TRACKING"] = args.quit_habit_tracking
    if args.weekly_checkin_day is not None:
        updates["WEEKLY_CHECKIN_DAY"] = args.weekly_checkin_day
    if args.weekly_checkin_time is not None:
        updates["WEEKLY_CHECKIN_TIME"] = args.weekly_checkin_time

    if not updates:
        _die("No preferences provided.")

    _upsert_env(updates)
    _cmd_prefs_get(args)


def _cmd_areas_get(args: argparse.Namespace) -> None:
    cfg = _read_config()
    areas = cfg.get("areas", [])
    if not isinstance(areas, list):
        areas = []
    _print_json({"ok": True, "areas": areas})


def _cmd_areas_set(args: argparse.Namespace) -> None:
    areas_raw: list[str] = []
    for a in args.area:
        a = (a or "").strip()
        if not a:
            continue
        areas_raw.append(a)

    # de-dupe while preserving order (case-insensitive)
    seen: set[str] = set()
    areas: list[str] = []
    for a in areas_raw:
        key = a.casefold()
        if key in seen:
            continue
        seen.add(key)
        areas.append(a)

    cfg = _read_config()
    cfg["areas"] = areas
    if not isinstance(cfg.get("onboarded"), bool):
        cfg["onboarded"] = False
    if "onboarded_at" not in cfg:
        cfg["onboarded_at"] = None
    _write_config(cfg)
    _print_json({"ok": True, "areas": areas})


def _cmd_onboarding_complete(args: argparse.Namespace) -> None:
    cfg = _read_config()
    cfg["onboarded"] = True
    cfg["onboarded_at"] = _now_utc_iso()
    if not isinstance(cfg.get("areas"), list):
        cfg["areas"] = []
    _write_config(cfg)
    _print_json({"ok": True, "config": cfg})


def _compute_goal_dates(args: argparse.Namespace) -> tuple[date, date]:
    horizon = args.horizon
    if horizon == "year":
        if args.year is not None:
            y = int(args.year)
            return date(y, 1, 1), date(y, 12, 31)
        if not args.start or not args.end:
            _die("For --horizon year, provide --year YYYY or both --start and --end.")
        return _parse_date(args.start), _parse_date(args.end)

    # quarter
    if not args.start:
        _die("For --horizon quarter, provide --start (e.g. today or YYYY-MM-DD).")
    start = _parse_date(args.start)
    if args.end:
        end = _parse_date(args.end)
        return start, end
    months = int(args.months)
    if months <= 0:
        _die("--months must be > 0")
    end = _add_months(start, months) - timedelta(days=1)
    return start, end


def _tokenize_title(s: str) -> set[str]:
    # Heuristic: split on non-alnum, drop common stop-words, keep short-but-meaningful tokens.
    # This is intentionally small and biased toward practical matching (e.g. "gym", "apps").
    stop = {
        "a",
        "an",
        "and",
        "at",
        "by",
        "doing",
        "for",
        "from",
        "get",
        "go",
        "in",
        "into",
        "keep",
        "make",
        "month",
        "of",
        "on",
        "per",
        "plan",
        "reach",
        "save",
        "start",
        "stop",
        "the",
        "time",
        "times",
        "to",
        "week",
        "with",
        "year",
    }
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", (s or "").casefold()) if t]
    out: set[str] = set()
    for t in toks:
        if t in stop:
            continue
        if any(ch.isdigit() for ch in t) and len(t) >= 2:
            out.add(t)
            continue
        if len(t) >= 3:
            out.add(t)
    return out


def _ensure_year_parent_for_quarter_goal(
    conn: sqlite3.Connection,
    *,
    quarter_start: date,
    area: str | None,
    focus: str | None,
    quarter_title: str,
) -> tuple[int, str, list[dict[str, Any]]]:
    """
    Ensure a quarter goal always has a year parent:
    - Prefer linking to an existing year goal in the same year (matching area/focus when provided).
    - If none match strictly, fall back to area-only matching + keyword matching before creating.
    - If still ambiguous, create a stable rollup year goal and use it.

    Returns: (parent_goal_id, resolution, candidates)
      resolution:
        - "linked_unique": strict match found exactly one parent
        - "linked_area_only": strict match missing, but area-only match found exactly one parent
        - "linked_keyword": keyword match picked a unique parent from multiple candidates
        - "created_missing": no reasonable candidates; created a new year parent
        - "created_ambiguous": multiple plausible candidates; created/used a rollup year parent
      candidates: list of candidate year goals considered (may be empty).
    """
    year = int(quarter_start.year)
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)

    area_norm = area.strip() if isinstance(area, str) and area.strip() else None
    focus_norm = focus.strip() if isinstance(focus, str) and focus.strip() else None

    def _load_candidates(*, match_focus: bool) -> list[dict[str, Any]]:
        wheres: list[str] = [
            "horizon = 'year'",
            "start_date = ?",
            "end_date = ?",
            "status = 'active'",
        ]
        params: list[Any] = [_iso(year_start), _iso(year_end)]
        if area_norm is not None:
            wheres.append("area = ?")
            params.append(area_norm)
        if match_focus and focus_norm is not None:
            wheres.append("focus = ?")
            params.append(focus_norm)

        sql = "SELECT * FROM goals WHERE " + " AND ".join(wheres) + " ORDER BY id ASC"
        return _rows_to_dicts(conn.execute(sql, params).fetchall())

    # 1) Strict match: (year + area [+ focus]).
    strict = _load_candidates(match_focus=True)
    candidates = strict

    if len(strict) == 1:
        return int(candidates[0]["id"]), "linked_unique", candidates

    # 2) If strict is empty, try area-only candidates (same year + area), then keyword match.
    # This helps when focus labels differ between year and quarter plans ("Training" vs "Fitness").
    if len(strict) == 0 and (area_norm is not None or focus_norm is not None):
        area_only = _load_candidates(match_focus=False)
        candidates = area_only
        if len(area_only) == 1:
            return int(area_only[0]["id"]), "linked_area_only", area_only
        if len(area_only) > 1:
            q_toks = _tokenize_title(" ".join([quarter_title or "", focus_norm or ""]))
            if q_toks:
                scored: list[tuple[int, dict[str, Any]]] = []
                for c in area_only:
                    hay = " ".join(
                        [str(c.get("title") or ""), str(c.get("focus") or "")]
                    )
                    c_toks = _tokenize_title(hay)
                    score = len(q_toks.intersection(c_toks))
                    scored.append((score, c))
                scored.sort(key=lambda x: (-x[0], int(x[1]["id"])))
                if scored and scored[0][0] > 0:
                    top_score = scored[0][0]
                    tops = [c for s, c in scored if s == top_score]
                    if len(tops) == 1:
                        return int(tops[0]["id"]), "linked_keyword", area_only

    # 3) Multiple strict candidates: try keyword match on title.
    if len(strict) > 1:
        q_toks = _tokenize_title(" ".join([quarter_title or "", focus_norm or ""]))
        if q_toks:
            scored: list[tuple[int, dict[str, Any]]] = []
            for c in strict:
                hay = " ".join([str(c.get("title") or ""), str(c.get("focus") or "")])
                c_toks = _tokenize_title(hay)
                score = len(q_toks.intersection(c_toks))
                scored.append((score, c))
            scored.sort(key=lambda x: (-x[0], int(x[1]["id"])))
            if scored and scored[0][0] > 0:
                top_score = scored[0][0]
                tops = [c for s, c in scored if s == top_score]
                if len(tops) == 1:
                    return int(tops[0]["id"]), "linked_keyword", strict

    # 4) If no candidates at all, create a new year parent.
    if len(candidates) == 0:
        ts = _now_utc_iso()
        title = f"{area_norm or 'Uncategorized'}: {focus_norm or quarter_title}".strip()
        notes = "Auto-created year goal as parent for quarter goals."
        cur = conn.execute(
            """
            INSERT INTO goals (title, horizon, start_date, end_date, status, parent_goal_id, area, focus, why, success_criteria, notes, created_at, updated_at)
            VALUES (?, 'year', ?, ?, 'active', NULL, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                title,
                _iso(year_start),
                _iso(year_end),
                area_norm,
                focus_norm,
                notes,
                ts,
                ts,
            ),
        )
        return int(cur.lastrowid), "created_missing", []

    # Still ambiguous: create a stable "rollup" parent for (year, area, focus).
    rollup_title = (
        f"{area_norm or 'Uncategorized'}: {focus_norm or 'Quarter Rollup'} (Rollup)"
    )
    existing_rollup = conn.execute(
        """
        SELECT * FROM goals
        WHERE horizon = 'year'
          AND start_date = ?
          AND end_date = ?
          AND (area IS ? OR area = ?)
          AND (focus IS ? OR focus = ?)
          AND title = ?
        LIMIT 1
        """,
        (
            _iso(year_start),
            _iso(year_end),
            area_norm,
            area_norm,
            focus_norm,
            focus_norm,
            rollup_title,
        ),
    ).fetchone()
    if existing_rollup is not None:
        return int(existing_rollup["id"]), "created_ambiguous", candidates

    ts = _now_utc_iso()
    notes = (
        "Auto-created rollup year goal as parent for an ambiguous quarter goal. "
        "Consider relinking to one of the existing year goals."
    )
    cur = conn.execute(
        """
        INSERT INTO goals (title, horizon, start_date, end_date, status, parent_goal_id, area, focus, why, success_criteria, notes, created_at, updated_at)
        VALUES (?, 'year', ?, ?, 'active', NULL, ?, ?, NULL, NULL, ?, ?, ?)
        """,
        (
            rollup_title,
            _iso(year_start),
            _iso(year_end),
            area_norm,
            focus_norm,
            notes,
            ts,
            ts,
        ),
    )
    return int(cur.lastrowid), "created_ambiguous", candidates


def _cmd_add_goal(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    start, end = _compute_goal_dates(args)
    if end < start:
        _die("--end must be on/after --start.")

    if args.parent_goal_id is not None:
        parent = _get_goal(conn, int(args.parent_goal_id))
        if parent is None:
            _die(f"parent goal not found: {args.parent_goal_id}")
        if parent["horizon"] != "year" and args.horizon == "quarter":
            _die("For quarter goals, --parent-goal-id must refer to a year goal.")

    ts = _now_utc_iso()
    parent_goal_id = args.parent_goal_id
    parent_resolution = None
    parent_candidates: list[dict[str, Any]] = []

    # Ensure quarter goals always have a year parent (link or create).
    if args.horizon == "quarter" and parent_goal_id is None:
        parent_goal_id, parent_resolution, parent_candidates = (
            _ensure_year_parent_for_quarter_goal(
                conn,
                quarter_start=start,
                area=args.area,
                focus=args.focus,
                quarter_title=args.title.strip(),
            )
        )

    cur = conn.execute(
        """
        INSERT INTO goals (title, horizon, start_date, end_date, status, parent_goal_id, area, focus, why, success_criteria, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            args.title.strip(),
            args.horizon,
            _iso(start),
            _iso(end),
            parent_goal_id,
            args.area,
            args.focus,
            args.why,
            args.success,
            args.notes,
            ts,
            ts,
        ),
    )
    goal_id = int(cur.lastrowid)
    conn.commit()
    row = _get_goal(conn, goal_id)
    out: dict[str, Any] = {"ok": True, "goal": _row_to_dict(row)}
    if args.horizon == "quarter" and parent_resolution is not None:
        out["parent_resolution"] = parent_resolution
        out["parent_candidates"] = parent_candidates
        out["parent_goal"] = _row_to_dict(_get_goal(conn, int(parent_goal_id)))
    _print_json(out)


def _cmd_list_goals(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    wheres: list[str] = []
    params: list[Any] = []
    if args.horizon:
        wheres.append("horizon = ?")
        params.append(args.horizon)
    if not args.all:
        wheres.append("status = ?")
        params.append(args.status)
    else:
        if args.status:
            wheres.append("status = ?")
            params.append(args.status)

    sql = "SELECT * FROM goals"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY end_date ASC, id ASC"

    goals = _rows_to_dicts(conn.execute(sql, params).fetchall())
    _print_json({"ok": True, "goals": goals})


def _cmd_add_goal_update(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    goal_id = int(args.goal_id)
    goal = _get_goal(conn, goal_id)
    if goal is None:
        _die(f"goal not found: {goal_id}")

    d = _parse_date(args.date) if args.date else date.today()
    note = args.note
    value = float(args.value) if args.value is not None else None
    if note is None and value is None:
        _die("Provide --note and/or --value.")

    ts = _now_utc_iso()
    cur = conn.execute(
        """
        INSERT INTO goal_updates (goal_id, date, progress_note, progress_value, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (goal_id, _iso(d), note, value, ts),
    )
    update_id = int(cur.lastrowid)
    conn.execute(
        "UPDATE goals SET updated_at = ? WHERE id = ?", (_now_utc_iso(), goal_id)
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM goal_updates WHERE id = ?", (update_id,)
    ).fetchone()
    _print_json({"ok": True, "goal": _row_to_dict(goal), "update": _row_to_dict(row)})


def _cmd_list_goal_updates(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    wheres: list[str] = []
    params: list[Any] = []
    if args.goal_id is not None:
        wheres.append("goal_id = ?")
        params.append(int(args.goal_id))

    sql = "SELECT * FROM goal_updates"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY date DESC, id DESC LIMIT ?"
    params.append(int(args.limit))

    rows = _rows_to_dicts(conn.execute(sql, params).fetchall())
    _print_json({"ok": True, "goal_updates": rows})


def _cmd_update_goal(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    goal_id = int(args.goal_id)
    existing = _get_goal(conn, goal_id)
    if existing is None:
        _die(f"goal not found: {goal_id}")

    fields: list[str] = []
    params: list[Any] = []

    def set_field(col: str, val: Any) -> None:
        fields.append(f"{col} = ?")
        params.append(val)

    if args.title is not None:
        set_field("title", args.title.strip())
    if args.status is not None:
        set_field("status", args.status)
    if args.start is not None:
        set_field("start_date", _iso(_parse_date(args.start)))
    if args.end is not None:
        set_field("end_date", _iso(_parse_date(args.end)))
    if args.parent_goal_id is not None:
        if args.parent_goal_id == 0:
            # Keep invariants: quarter goals must always have a year parent.
            if existing["horizon"] == "quarter":
                _die(
                    "Quarter goals must always be linked to a year goal; provide a year --parent-goal-id instead of 0."
                )
            set_field("parent_goal_id", None)
        else:
            parent = _get_goal(conn, int(args.parent_goal_id))
            if parent is None:
                _die(f"parent goal not found: {args.parent_goal_id}")
            if existing["horizon"] == "quarter" and parent["horizon"] != "year":
                _die("For quarter goals, --parent-goal-id must refer to a year goal.")
            set_field("parent_goal_id", int(args.parent_goal_id))
    if args.why is not None:
        set_field("why", args.why)
    if args.area is not None:
        set_field("area", args.area)
    if args.focus is not None:
        set_field("focus", args.focus)
    if args.success is not None:
        set_field("success_criteria", args.success)
    if args.notes is not None:
        set_field("notes", args.notes)

    if not fields:
        _die("No updates provided.")

    set_field("updated_at", _now_utc_iso())
    params.append(goal_id)

    conn.execute(f"UPDATE goals SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    row = _get_goal(conn, goal_id)
    _print_json({"ok": True, "goal": _row_to_dict(row)})


def _cmd_add_habit(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    start = _parse_date(args.start) if args.start else date.today()
    end = _parse_date(args.end) if args.end else None
    if end is not None and end < start:
        _die("--end must be on/after --start.")

    goal_ids = [int(x) for x in (args.goal_id or [])]
    for gid in goal_ids:
        if _get_goal(conn, gid) is None:
            _die(f"goal not found: {gid}")

    ts = _now_utc_iso()
    cur = conn.execute(
        """
        INSERT INTO habits (name, kind, cadence, target, unit, start_date, end_date, status, area, focus, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
        """,
        (
            args.name.strip(),
            args.kind,
            args.cadence,
            int(args.target),
            args.unit,
            _iso(start),
            _iso(end) if end else None,
            args.area,
            args.focus,
            args.notes,
            ts,
            ts,
        ),
    )
    habit_id = int(cur.lastrowid)

    for gid in goal_ids:
        conn.execute(
            "INSERT OR IGNORE INTO goal_habits (goal_id, habit_id) VALUES (?, ?)",
            (gid, habit_id),
        )

    conn.commit()
    row = _get_habit(conn, habit_id)
    _print_json({"ok": True, "habit": _row_to_dict(row), "linked_goal_ids": goal_ids})


def _cmd_list_habits(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    wheres: list[str] = []
    params: list[Any] = []
    if not args.all:
        wheres.append("status = ?")
        params.append(args.status)
    else:
        if args.status:
            wheres.append("status = ?")
            params.append(args.status)

    sql = "SELECT * FROM habits"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY name COLLATE NOCASE ASC, id ASC"

    habits = _rows_to_dicts(conn.execute(sql, params).fetchall())
    _print_json({"ok": True, "habits": habits})


def _cmd_update_habit(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    habit_id = int(args.habit_id)
    existing = _get_habit(conn, habit_id)
    if existing is None:
        _die(f"habit not found: {habit_id}")

    fields: list[str] = []
    params: list[Any] = []

    def set_field(col: str, val: Any) -> None:
        fields.append(f"{col} = ?")
        params.append(val)

    if args.name is not None:
        set_field("name", args.name.strip())
    if args.status is not None:
        set_field("status", args.status)
    if args.cadence is not None:
        set_field("cadence", args.cadence)
    if args.target is not None:
        set_field("target", int(args.target))
    if args.unit is not None:
        set_field("unit", args.unit)
    if args.area is not None:
        set_field("area", args.area)
    if args.focus is not None:
        set_field("focus", args.focus)
    if args.start is not None:
        set_field("start_date", _iso(_parse_date(args.start)))
    if args.end is not None:
        if args.end == "0":
            set_field("end_date", None)
        else:
            set_field("end_date", _iso(_parse_date(args.end)))
    if args.notes is not None:
        set_field("notes", args.notes)

    if not fields:
        _die("No updates provided.")

    set_field("updated_at", _now_utc_iso())
    params.append(habit_id)

    conn.execute(f"UPDATE habits SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    row = _get_habit(conn, habit_id)
    _print_json({"ok": True, "habit": _row_to_dict(row)})


def _cmd_link_habit_goal(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    habit_id = int(args.habit_id)
    goal_id = int(args.goal_id)
    if _get_habit(conn, habit_id) is None:
        _die(f"habit not found: {habit_id}")
    if _get_goal(conn, goal_id) is None:
        _die(f"goal not found: {goal_id}")

    conn.execute(
        "INSERT OR IGNORE INTO goal_habits (goal_id, habit_id) VALUES (?, ?)",
        (goal_id, habit_id),
    )
    conn.commit()
    _print_json({"ok": True, "goal_id": goal_id, "habit_id": habit_id})


def _cmd_log_habit(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    habit_id = int(args.habit_id)
    habit = _get_habit(conn, habit_id)
    if habit is None:
        _die(f"habit not found: {habit_id}")

    d = _parse_date(args.date) if args.date else date.today()
    # If the user backfills earlier dates, auto-backdate the habit start date so summaries feel consistent.
    try:
        habit_start = _parse_date(str(habit["start_date"]))
    except Exception:
        habit_start = d
    if d < habit_start:
        conn.execute(
            "UPDATE habits SET start_date = ? WHERE id = ?", (_iso(d), habit_id)
        )

    if args.value is None:
        value = 1 if habit["kind"] == "build" else 0
    else:
        value = int(args.value)
    if value < 0:
        _die("--value must be >= 0")

    ts = _now_utc_iso()
    existing = conn.execute(
        "SELECT * FROM habit_logs WHERE habit_id = ? AND date = ?",
        (habit_id, _iso(d)),
    ).fetchone()

    if existing is None:
        conn.execute(
            "INSERT INTO habit_logs (habit_id, date, value, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (habit_id, _iso(d), value, args.note, ts),
        )
    else:
        if args.mode == "add":
            new_value = int(existing["value"]) + value
        else:
            new_value = value
        new_note = args.note if args.note is not None else existing["note"]
        conn.execute(
            "UPDATE habit_logs SET value = ?, note = ? WHERE id = ?",
            (new_value, new_note, int(existing["id"])),
        )

    conn.execute(
        "UPDATE habits SET updated_at = ? WHERE id = ?", (_now_utc_iso(), habit_id)
    )
    conn.commit()

    habit = _get_habit(conn, habit_id)
    row = conn.execute(
        "SELECT * FROM habit_logs WHERE habit_id = ? AND date = ?",
        (habit_id, _iso(d)),
    ).fetchone()
    _print_json({"ok": True, "habit": _row_to_dict(habit), "log": _row_to_dict(row)})


def _cmd_habit_week(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    ws = _parse_date(args.week_start) if args.week_start else _week_start(date.today())
    ws = _week_start(ws)
    summary = _habit_week_summary(conn, ws, status=args.status)
    _print_json({"ok": True, **summary})


def _month_period_from_arg(raw: str | None) -> _Period:
    if raw is None or not str(raw).strip():
        today = date.today()
        y, m = today.year, today.month
    else:
        s = str(raw).strip()
        mobj = re.match(r"^(\d{4})-(\d{2})$", s)
        if not mobj:
            raise ValueError("--month must be YYYY-MM (e.g. 2026-02)")
        y = int(mobj.group(1))
        m = int(mobj.group(2))
        if m < 1 or m > 12:
            raise ValueError("--month must be YYYY-MM (month 01..12)")
    start = date(y, m, 1)
    end = date(y, m, calendar.monthrange(y, m)[1])
    return _Period(start=start, end=end)


def _cmd_habit_month(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    try:
        period = _month_period_from_arg(args.month)
    except ValueError as e:
        _die(str(e))

    summary = _habit_range_summary(conn, period.start, period.end, status=args.status)
    _print_json(
        {
            "ok": True,
            "month": {"start": _iso(period.start), "end": _iso(period.end)},
            **summary,
        }
    )


def _read_json_payload(path_or_dash: str) -> tuple[Any, str]:
    if path_or_dash == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path_or_dash).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON payload: {e}") from e
    canonical = json.dumps(
        parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return parsed, canonical


def _cmd_add_checkin(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        _die("--end must be on/after --start.")

    data_json: str | None = None
    parsed: Any | None = None
    if args.json is not None:
        try:
            parsed, data_json = _read_json_payload(args.json)
        except ValueError as e:
            _die(str(e))

    ts = _now_utc_iso()
    cur = conn.execute(
        """
        INSERT INTO checkins (period, start_date, end_date, summary, data_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (args.period, _iso(start), _iso(end), args.summary, data_json, ts),
    )
    checkin_id = int(cur.lastrowid)
    conn.commit()
    row = conn.execute("SELECT * FROM checkins WHERE id = ?", (checkin_id,)).fetchone()
    _print_json({"ok": True, "checkin": _row_to_dict(row), "data": parsed})


def _cmd_list_checkins(args: argparse.Namespace) -> None:
    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_schema(conn)

    wheres: list[str] = []
    params: list[Any] = []
    if args.period:
        wheres.append("period = ?")
        params.append(args.period)
    sql = "SELECT * FROM checkins"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY start_date DESC, id DESC LIMIT ?"
    params.append(int(args.limit))

    rows = _rows_to_dicts(conn.execute(sql, params).fetchall())
    _print_json({"ok": True, "checkins": rows})


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="goals", description="Goals & habits tracker (SQLite)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize the SQLite DB (idempotent)").set_defaults(
        func=_cmd_init
    )

    sub.add_parser(
        "dashboard", help="Get active goals/habits + current week summary"
    ).set_defaults(func=_cmd_dashboard)

    # "Chat-first" onboarding helpers.
    # These let the agent initialize preferences and categories without exposing shell commands to the user.
    b = sub.add_parser(
        "bootstrap",
        help="Ensure DB + config exist and report whether onboarding is needed",
    )
    b.set_defaults(func=_cmd_bootstrap)

    pg = sub.add_parser("prefs-get", help="Read onboarding preferences from .env")
    pg.set_defaults(func=_cmd_prefs_get)

    ps = sub.add_parser("prefs-set", help="Set onboarding preferences in .env")
    ps.add_argument("--week-start", choices=["mon", "sun"])
    ps.add_argument("--timezone", help="IANA timezone, e.g. Europe/Madrid")
    ps.add_argument("--quit-habit-tracking", choices=["occurrences", "boolean"])
    ps.add_argument("--weekly-checkin-day", help="e.g. Sun, Monday, Thu")
    ps.add_argument("--weekly-checkin-time", help="HH:MM (24h), e.g. 18:00")
    ps.set_defaults(func=_cmd_prefs_set)

    ag = sub.add_parser(
        "areas-get", help="Get the list of areas/categories from config.json"
    )
    ag.set_defaults(func=_cmd_areas_get)

    aset = sub.add_parser(
        "areas-set", help="Replace the list of areas/categories in config.json"
    )
    aset.add_argument(
        "--area", action="append", default=[], help="Repeatable, e.g. --area Business"
    )
    aset.set_defaults(func=_cmd_areas_set)

    oc = sub.add_parser(
        "onboarding-complete", help="Mark onboarding complete in config.json"
    )
    oc.set_defaults(func=_cmd_onboarding_complete)

    g_add = sub.add_parser("add-goal", help="Add a goal")
    g_add.add_argument("--horizon", required=True, choices=["year", "quarter"])
    g_add.add_argument("--title", required=True)
    g_add.add_argument(
        "--year",
        type=int,
        help="Convenience for yearly goals (sets start/end to Jan 1..Dec 31)",
    )
    g_add.add_argument(
        "--start", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd"
    )
    g_add.add_argument("--end", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd")
    g_add.add_argument(
        "--months",
        type=int,
        default=3,
        help="For quarter goals without --end (default: 3)",
    )
    g_add.add_argument("--parent-goal-id", type=int)
    g_add.add_argument("--area", help="Optional category like Business/Health/Finance")
    g_add.add_argument("--focus", help="Optional focus/project label")
    g_add.add_argument("--why")
    g_add.add_argument("--success", dest="success")
    g_add.add_argument("--notes")
    g_add.set_defaults(func=_cmd_add_goal)

    g_list = sub.add_parser("list-goals", help="List goals")
    g_list.add_argument("--horizon", choices=["year", "quarter"])
    g_list.add_argument(
        "--status", choices=["active", "completed", "abandoned"], default="active"
    )
    g_list.add_argument(
        "--all", action="store_true", help="Do not default-filter to status=active"
    )
    g_list.set_defaults(func=_cmd_list_goals)

    gu_add = sub.add_parser("add-goal-update", help="Add a progress update for a goal")
    gu_add.add_argument("--goal-id", dest="goal_id", required=True, type=int)
    gu_add.add_argument(
        "--date",
        default="today",
        help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd",
    )
    gu_add.add_argument("--note")
    gu_add.add_argument("--value", type=float)
    gu_add.set_defaults(func=_cmd_add_goal_update)

    gu_list = sub.add_parser(
        "list-goal-updates", help="List goal updates (optionally filtered by goal)"
    )
    gu_list.add_argument("--goal-id", dest="goal_id", type=int)
    gu_list.add_argument("--limit", type=int, default=20)
    gu_list.set_defaults(func=_cmd_list_goal_updates)

    g_up = sub.add_parser("update-goal", help="Update goal fields")
    g_up.add_argument("--goal-id", required=True, type=int)
    g_up.add_argument("--title")
    g_up.add_argument("--status", choices=["active", "completed", "abandoned"])
    g_up.add_argument("--start", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd")
    g_up.add_argument("--end", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd")
    g_up.add_argument("--parent-goal-id", type=int, help="Set to 0 to clear")
    g_up.add_argument("--area", help="Optional category like Business/Health/Finance")
    g_up.add_argument("--focus", help="Optional focus/project label")
    g_up.add_argument("--why")
    g_up.add_argument("--success")
    g_up.add_argument("--notes")
    g_up.set_defaults(func=_cmd_update_goal)

    h_add = sub.add_parser("add-habit", help="Add a habit")
    h_add.add_argument("--name", required=True)
    h_add.add_argument("--kind", required=True, choices=["build", "quit"])
    h_add.add_argument("--cadence", required=True, choices=["daily", "weekly"])
    h_add.add_argument("--target", required=True, type=int)
    h_add.add_argument("--unit", default="times")
    h_add.add_argument(
        "--start",
        default="today",
        help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd",
    )
    h_add.add_argument("--end", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd")
    h_add.add_argument("--area", help="Optional category like Business/Health/Finance")
    h_add.add_argument("--focus", help="Optional focus/project label")
    h_add.add_argument("--notes")
    h_add.add_argument(
        "--goal-id",
        action="append",
        help="Repeatable; link habit to an existing goal id",
    )
    h_add.set_defaults(func=_cmd_add_habit)

    h_list = sub.add_parser("list-habits", help="List habits")
    h_list.add_argument(
        "--status", choices=["active", "paused", "archived"], default="active"
    )
    h_list.add_argument(
        "--all", action="store_true", help="Do not default-filter to status=active"
    )
    h_list.set_defaults(func=_cmd_list_habits)

    h_up = sub.add_parser("update-habit", help="Update habit fields")
    h_up.add_argument("--habit-id", required=True, type=int)
    h_up.add_argument("--name")
    h_up.add_argument("--status", choices=["active", "paused", "archived"])
    h_up.add_argument("--cadence", choices=["daily", "weekly"])
    h_up.add_argument("--target", type=int)
    h_up.add_argument("--unit")
    h_up.add_argument("--area", help="Optional category like Business/Health/Finance")
    h_up.add_argument("--focus", help="Optional focus/project label")
    h_up.add_argument("--start", help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd")
    h_up.add_argument(
        "--end",
        help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd (set to 0 to clear)",
    )
    h_up.add_argument("--notes")
    h_up.set_defaults(func=_cmd_update_habit)

    link = sub.add_parser(
        "link-habit-goal", help="Link an existing habit to an existing goal"
    )
    link.add_argument("--habit-id", required=True, type=int)
    link.add_argument("--goal-id", required=True, type=int)
    link.set_defaults(func=_cmd_link_habit_goal)

    log = sub.add_parser("log-habit", help="Upsert a habit log for a specific date")
    log.add_argument("--habit-id", required=True, type=int)
    log.add_argument(
        "--date",
        default="today",
        help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd",
    )
    log.add_argument("--value", type=int, help="If omitted: build->1, quit->0")
    log.add_argument("--note")
    log.add_argument("--mode", choices=["set", "add"], default="set")
    log.set_defaults(func=_cmd_log_habit)

    hw = sub.add_parser(
        "habit-week", help="Summarize habits for a week (Monday..Sunday)"
    )
    hw.add_argument(
        "--week-start", help="YYYY-MM-DD or today/...; will be normalized to Monday"
    )
    hw.add_argument(
        "--status", choices=["active", "paused", "archived", "all"], default="active"
    )
    hw.set_defaults(func=_cmd_habit_week)

    hm = sub.add_parser("habit-month", help="Summarize habits for a calendar month")
    hm.add_argument("--month", help="YYYY-MM (defaults to current month)")
    hm.add_argument(
        "--status", choices=["active", "paused", "archived", "all"], default="active"
    )
    hm.set_defaults(func=_cmd_habit_month)

    c_add = sub.add_parser(
        "add-checkin", help="Add a weekly/quarterly/yearly check-in or planning session"
    )
    c_add.add_argument("--period", required=True, choices=["week", "quarter", "year"])
    c_add.add_argument(
        "--start", required=True, help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd"
    )
    c_add.add_argument(
        "--end", required=True, help="YYYY-MM-DD or today/yesterday/tomorrow or +/-Nd"
    )
    c_add.add_argument("--summary")
    c_add.add_argument(
        "--json",
        help="Path to JSON file or '-' to read JSON from stdin (stored as text in DB)",
    )
    c_add.set_defaults(func=_cmd_add_checkin)

    c_list = sub.add_parser("list-checkins", help="List check-ins / planning sessions")
    c_list.add_argument("--period", choices=["week", "quarter", "year"])
    c_list.add_argument("--limit", type=int, default=20)
    c_list.set_defaults(func=_cmd_list_checkins)

    return p


def main(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:
        # Allow piping to head/jq without stack traces.
        raise SystemExit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
