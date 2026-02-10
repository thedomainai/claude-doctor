# claude-config-doctor

A diagnostic tool that analyzes the health of your [Claude Code](https://docs.anthropic.com/en/docs/claude-code) configuration as a whole system.

Claude Code uses a layered config hierarchy (`CLAUDE.md`, `rules/`, `skills/`, `memory/`). As projects grow, configs accumulate redundancy, scope misplacements, stale references, and coverage gaps. This tool detects those problems with a single command.

## Quick Start

```bash
# No installation needed — just clone and run
git clone https://github.com/<your-username>/claude-config-doctor.git
cd claude-config-doctor

python3 src/config_doctor.py
```

> **Requirements:** Python 3.9+ (no external dependencies, stdlib only)

## Usage

```bash
# Human-readable report (default)
python3 src/config_doctor.py

# JSON output (for scripting or Claude Code skill integration)
python3 src/config_doctor.py --json

# Disable colors (e.g. for piping)
python3 src/config_doctor.py --no-color

# Custom context window size (default: 200,000 tokens)
python3 src/config_doctor.py --context-window 1000000

# Specify a custom .claude directory
python3 src/config_doctor.py --claude-home /path/to/.claude
```

### Example Output

```
  ╭────────────────────────────────────────────────────╮
  │  Claude Config Doctor                              │
  ╰────────────────────────────────────────────────────╯

  Score   ████████████████████████  95/100  A

  Tokens  ~1,200 / 200,000 (0.6%)
  Files   6 config files across 3 projects
  Issues  0 errors, 1 warnings, 2 infos

  ─── Dimensions ──────────────────────────────────────────

    Budget         ████████████████████  100  pass
    Redundancy     ████████████████████  100  pass
    Scope Fitness  ████████████████████  100  pass
    Freshness      ████████████████████  100  pass
    Conflicts      ████████████████████  100  pass
    Coverage       ██████████████████░░   90  pass
    Effectiveness  ███████████████████░   95  pass
```

## What It Checks

The tool evaluates your config across **7 dimensions**:

| Dimension | What it checks |
|-----------|---------------|
| **Budget** | Total token consumption vs. context window. Flags individual files over 1,000 tokens and MEMORY.md approaching the 200-line limit. |
| **Redundancy** | Duplicated directives across config layers (e.g. same rule in global CLAUDE.md and a project CLAUDE.md). |
| **Scope Fitness** | Directives placed at the wrong level (e.g. project-specific instructions in global config, or rule-like content in MEMORY.md). |
| **Freshness** | Stale path references that no longer exist on disk. Files not updated in 90+ days. |
| **Conflicts** | Contradictory directives across layers (e.g. "Respond in English" in one file vs. "Respond in Japanese" in another). |
| **Coverage** | Missing global CLAUDE.md, missing language/style preferences, projects without CLAUDE.md, under-specified configs for multi-project setups. |
| **Effectiveness** | Excessive negative directives ("don't", "never", "avoid") and vague expressions ("as needed", "if appropriate"). |

### Context-Dependent Evaluation

Not all checks apply to every setup. The tool automatically detects your environment:

- **Config maturity** — `sparse` (<500 tokens), `moderate` (500-3000), `rich` (3000+)
- **Setup type** — `single_project` or `multi_project`

Checks like Redundancy, Conflicts, and Effectiveness are skipped for sparse configs where they would produce false positives.

## Scoring

Each dimension receives a score from 0 to 100. The overall score is a weighted average:

| Dimension | Weight |
|-----------|--------|
| Freshness | 1.5x |
| Conflicts | 1.5x |
| Budget | 1.0x |
| Scope Fitness | 1.0x |
| Coverage | 1.0x |
| Redundancy | 0.5x |
| Effectiveness | 0.5x |

Rating scale: **A** (90+), **B** (70-89), **C** (50-69), **D** (30-49), **F** (0-29)

## How It Works

The tool scans the standard Claude Code config locations:

```
~/.claude/
├── CLAUDE.md                          # Global instructions
├── rules/*.md                         # Path-scoped rules
├── skills/*/SKILL.md                  # Custom skills
├── projects/<path>/memory/MEMORY.md   # Per-workspace memory (200 line limit)
└── settings.json                      # Permission & plugin settings

<your-project>/
└── CLAUDE.md                          # Project-specific instructions
```

Projects are discovered automatically by reverse-mapping `~/.claude/projects/` entries and scanning for project markers (`.git`, `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, etc.).

## Claude Code Skill Integration

You can integrate this tool as a Claude Code [skill](https://docs.anthropic.com/en/docs/claude-code/skills) for on-demand diagnosis within your sessions.

Create `~/.claude/skills/config-doctor/SKILL.md`:

```markdown
---
description: "Diagnose Claude Code config health"
user-invocable: true
argument-hint: "[--quick]"
---

# Config Doctor

Run the static analysis CLI:

\`\`\`bash
python3 /path/to/claude-config-doctor/src/config_doctor.py --json
\`\`\`

If `--quick` is passed, report the JSON results directly.
Otherwise, use the JSON output as a basis for deeper semantic analysis
(detecting synonymous directives, contextual contradictions, etc.).
```

Then invoke it with `/config-doctor` or `/config-doctor --quick` in any Claude Code session.

> **Note:** Update `/path/to/claude-config-doctor/` to the actual path where you cloned this repository.

## JSON Output Schema

The `--json` flag outputs structured data:

```json
{
  "overall": {
    "score": 0.93,
    "rating": "A",
    "score_percent": 93
  },
  "context": {
    "project_count": 3,
    "total_config_tokens": 2500,
    "context_window": 200000,
    "config_maturity": "moderate",
    "setup_type": "multi_project",
    "has_custom_skills": true,
    "has_custom_rules": true,
    "has_project_memory": true,
    "memory_line_count": 45,
    "rule_count": 2,
    "skill_count": 3
  },
  "checks": [
    {
      "check_name": "budget",
      "display_name": "Budget",
      "score": 1.0,
      "applicable": true,
      "findings": []
    }
  ],
  "files": { }
}
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

All tests run with mock data and temporary directories. No real `~/.claude/` is accessed during testing.

## License

MIT
