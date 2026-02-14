# SQLite schema (goals-habits-tracker)

This skill stores state in a local SQLite DB (default: `.skills-data/goals-habits-tracker/goals.sqlite3`).

## Tables (v2)

### `goals`
- `id` integer primary key
- `title` text (required)
- `horizon` text: `year` | `quarter`
- `start_date` text (`YYYY-MM-DD`)
- `end_date` text (`YYYY-MM-DD`)
- `status` text: `active` | `completed` | `abandoned`
- `parent_goal_id` integer nullable
  - Intended use: quarter goals under a year goal.
  - Skill behavior: when creating a quarter goal, the CLI will always link it to an existing year goal (best match by year + area/focus/title) or create a new year goal parent if needed.
- `area` text nullable (e.g., Business, Health)
- `focus` text nullable (e.g., Income, Training)
- `why` text nullable
- `success_criteria` text nullable
- `notes` text nullable
- `created_at` text (ISO-8601 UTC)
- `updated_at` text (ISO-8601 UTC)

### `goal_updates`
Optional progress notes over time.
- `id` integer primary key
- `goal_id` integer (FK -> `goals.id`)
- `date` text (`YYYY-MM-DD`)
- `progress_note` text nullable
- `progress_value` real nullable (percent or arbitrary numeric)
- `created_at` text (ISO-8601 UTC)

### `habits`
- `id` integer primary key
- `name` text (required)
- `kind` text: `build` | `quit`
- `cadence` text: `daily` | `weekly`
- `target` integer (required)
  - For `build`: target completions per cadence period
  - For `quit`: target occurrences per cadence period (usually `0`)
- `unit` text (default `times`)
- `start_date` text (`YYYY-MM-DD`)
- `end_date` text nullable (`YYYY-MM-DD`)
- `status` text: `active` | `paused` | `archived`
- `area` text nullable (e.g., Health)
- `focus` text nullable (e.g., Weight)
- `notes` text nullable
- `created_at` text (ISO-8601 UTC)
- `updated_at` text (ISO-8601 UTC)

### `habit_logs`
One row per habit per day (upserted by `log-habit`).
- `id` integer primary key
- `habit_id` integer (FK -> `habits.id`)
- `date` text (`YYYY-MM-DD`)
- `value` integer (non-negative)
- `note` text nullable
- `created_at` text (ISO-8601 UTC)
- Unique: (`habit_id`, `date`)

### `goal_habits`
Many-to-many link between goals and habits.
- `goal_id` integer (FK -> `goals.id`)
- `habit_id` integer (FK -> `habits.id`)
- Primary key: (`goal_id`, `habit_id`)

### `checkins`
Stores weekly check-ins and planning sessions (quarter/year).
- `id` integer primary key
- `period` text: `week` | `quarter` | `year`
- `start_date` text (`YYYY-MM-DD`)
- `end_date` text (`YYYY-MM-DD`)
- `summary` text nullable
- `data_json` text nullable (arbitrary JSON payload as text)
- `created_at` text (ISO-8601 UTC)

## Recommended `data_json` shapes

### Year / quarter planning session
```json
{
  "themes": ["Health", "Craft"],
  "goals": [
    {
      "title": "Ship v1",
      "success_criteria": "10 paying users by May",
      "key_habits": ["Ship 1 feature/week", "Talk to 3 users/week"]
    }
  ],
  "constraints": ["No work weekends"],
  "risks": ["Scope creep"],
  "next_actions": ["Schedule user interviews", "Write v1 spec"]
}
```

### Weekly check-in
```json
{
  "wins": ["..."],
  "misses": ["..."],
  "lessons": ["..."],
  "next_week_focus": ["..."],
  "goal_updates": [{"goal_id": 1, "progress_note": "..."}],
  "habit_notes": [{"habit_id": 2, "note": "..."}]
}
```
