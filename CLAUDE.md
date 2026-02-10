# claude-doctor

A diagnostic tool for Claude Code configuration files (CLAUDE.md, rules, skills, memory).

## Development

```bash
# Run
python3 src/config_doctor.py
python3 src/config_doctor.py --json

# Test
python3 -m pytest tests/ -v
```

## Tech Stack

- Python 3.9+ (stdlib only, no external dependencies)

## Directory Structure

```
src/config_doctor.py       # Single-file CLI
tests/test_config_doctor.py
```
