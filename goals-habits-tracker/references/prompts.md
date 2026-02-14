# Prompts & session templates

Use these when the user asks for planning/check-ins. Keep it short: ask the minimum questions needed to produce a usable plan and save it.

## Habit tracking setup (first run)
Ask:
- Which 1-3 habits do you want to start tracking first?
- Which 1 habit do you want to stop doing (if any)?
- How do you want to check in weekly (day/time)?
- For quit habits, do you want to log occurrences (0/1/2...) or boolean (done/not done)?

Persist:
- Save check-in preferences via `prefs-set`.
- Create habits via `add-habit`.

## Onboarding script (one question at a time)
Run `bootstrap` first. If `needs_onboarding == true`, ask the following in order, one per message, and persist immediately:
- Week start: Mon-Sun (recommended) vs Sun-Sat -> `prefs-set --week-start ...`
- Timezone: IANA (or skip) -> `prefs-set --timezone ...`
- Quit-habit tracking: occurrences vs boolean -> `prefs-set --quit-habit-tracking ...`
- Areas/categories: comma-separated (or skip) -> `areas-set --area ...`
- Weekly check-in day/time: (or skip) -> `prefs-set --weekly-checkin-day ... --weekly-checkin-time ...`
Then run `onboarding-complete`.

## Yearly planning (60-90 minutes)

Ask:
- What year are we planning for?
- What are 1-3 "themes" for the year?
- What are the top 1-3 goals? (Each must have a measurable success criteria.)
- What are the biggest constraints (time, health, money) that must be respected?
- What are 1-3 "lead measures" (habits) that would make the goals inevitable?

Output (in chat):
- A short "Year Plan" with themes, 1-3 goals, success criteria, and 1-5 habits.
- A default weekly cadence (pick a weekly check-in day).

If the user plans as a table (Area / Project-Focus / Task):
- Map each row to a goal:
  - goal.area = Area (e.g., Business)
  - goal.focus = Project/Focus (e.g., Income)
  - goal.title = Task/Goal (e.g., Reach 25-30k net monthly income)
- Store the full table, then ask the user to pick:
  - Top 3 priorities for weekly focus
  - 3-5 habits to track weekly

Persist:
- Create each goal via `add-goal --horizon year --year <YYYY>`.
- Create supporting habits via `add-habit` and link to goals via `link-habit-goal`.
- Save a session via `add-checkin --period year --start YYYY-01-01 --end YYYY-12-31 --summary ... --json -`.

If the user provides KPI targets (like a quarterly KPI table):
- Store them in the yearly planning session `data_json` under a key like `kpis` (and keep the original table text in `notes` if helpful).

## Quarterly / "next 3 months" planning (30-60 minutes)

Ask:
- When does the 3-month window start? (Default: today.)
- What is the single most important outcome? (Optionally 2-3 total goals.)
- What does "done" look like (success criteria)?
- What 1-3 habits will drive it each week?
- What should be explicitly deprioritized this quarter?

Persist:
- Create 1-3 goals via `add-goal --horizon quarter --start <date> --months 3 ...`.
- Create/link habits.
- Save a session via `add-checkin --period quarter --start <start> --end <end> ...`.

If the user provides a 3-month table, store each row as a goal with `area` and `focus` as above.

If the user provides an "Apps releases" table/list:
- Either store each app as a quarter goal (area Business, focus Own Apps) or store the whole list under quarter planning session `data_json.apps`.

## Weekly check-in (10-20 minutes)

Start:
- Run `dashboard` (or `habit-week`) to ground the conversation.

Ask:
- Wins: what went well?
- Misses: what didn't happen (and why)?
- Lessons: what pattern should we keep/change?
- Next week: what are 1-3 focus items?
- Habit tweaks: which habit needs to be easier/more specific?

Persist:
- Save the check-in via `add-checkin --period week --start <week_start> --end <week_end> --summary ... --json -`.

## New habit (5-10 minutes)

Ask:
- Is this a "build" habit (do more) or "quit" habit (do less/none)?
- Daily or weekly cadence?
- What's the target number?
- What's the smallest version you can do even on bad days?

Persist:
- `add-habit ...`

## "Stop doing {something}" (quit habit)

Ask:
- What exactly are we stopping? (make it observable)
- When/where does it usually happen? (trigger)
- What's the replacement behavior?
- How should we track it?

Persist:
- Create a quit habit with `--kind quit`.
- For tracking, recommend:
  - Log `0` on success days.
  - Log `1+` on slip days (occurrences).
