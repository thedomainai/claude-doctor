#!/usr/bin/env python3
"""claude-doctor: Diagnose the health of your Claude Code configuration.

Analyzes the entire config hierarchy (CLAUDE.md, rules, skills, memory, settings)
as a system to detect redundancy, scope misplacement, budget overruns,
staleness, conflicts, coverage gaps, and effectiveness issues.

Works with any Claude Code environment.
"""

import argparse
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# ANSI Color Support
# =============================================================================

class Style:
    """ANSI escape codes for terminal styling."""

    _enabled = True

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    _defaults: dict = {}

    @classmethod
    def disable(cls) -> None:
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")
        cls._enabled = False

    @classmethod
    def reset(cls) -> None:
        """Restore all style attributes to their original values."""
        for attr, value in cls._defaults.items():
            setattr(cls, attr, value)
        cls._enabled = True

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled


# Store default Style values for reset()
Style._defaults = {
    attr: getattr(Style, attr)
    for attr in dir(Style)
    if attr.isupper() and not attr.startswith("_")
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FileMetrics:
    """Static metrics for a single config file."""
    path: str
    category: str  # global_claude_md, rule, skill, memory, project_claude_md, settings
    exists: bool
    size_bytes: int
    line_count: int
    estimated_tokens: int
    last_modified: Optional[str]
    sections: List[str]
    referenced_paths: List[str]
    directive_count: int
    negation_count: int
    content: str


@dataclass
class UserContext:
    """User's config environment context."""
    project_count: int
    total_config_tokens: int
    config_maturity: str   # sparse, moderate, rich
    setup_type: str        # single_project, multi_project
    has_custom_skills: bool
    has_custom_rules: bool
    has_project_memory: bool
    memory_line_count: int
    rule_count: int
    skill_count: int
    project_dirs: List[str]  # discovered project directories


@dataclass
class Finding:
    """A single diagnostic finding."""
    severity: str  # info, warning, error
    dimension: str
    message: str
    file_path: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class CheckResult:
    """Result of a single check."""
    check_name: str
    display_name: str
    findings: List[Finding]
    score: float  # 0.0 - 1.0
    applicable: bool


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count based on ASCII ratio.

    English text: ~4 chars/token
    CJK text: ~1.5 chars/token
    """
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii_chars = len(text) - ascii_chars
    en_tokens = ascii_chars / 4.0
    ja_tokens = non_ascii_chars / 1.5
    return max(1, int(en_tokens + ja_tokens))


# =============================================================================
# File Analysis
# =============================================================================

def extract_sections(content: str) -> List[str]:
    """Extract Markdown headings."""
    return [
        line.strip().lstrip("#").strip()
        for line in content.splitlines()
        if line.strip().startswith("#")
    ]


_CJK_RE = re.compile(r'[\u3000-\u9fff\uff00-\uffef]')
_PLACEHOLDER_RE = re.compile(r'YYYY|UUID|xxxx', re.IGNORECASE)
_PATH_PREFIX_RE = re.compile(
    r'^/(Users|home|var|etc|opt|tmp|usr|Library|root|srv|mnt)/'
)


def _looks_like_filesystem_path(p: str) -> bool:
    """Determine if a string is a real filesystem path.

    Rejects CJK slash expressions, template placeholders, tool name slashes, etc.
    """
    # Template placeholders
    if '<' in p or '>' in p:
        return False
    # URL
    if p.startswith('///') or p.startswith('//'):
        return False
    # CJK characters (slash-separated text, e.g. 高/中/低)
    if _CJK_RE.search(p):
        return False
    # Placeholder patterns
    if _PLACEHOLDER_RE.search(p):
        return False
    if len(p) < 5:
        return False
    # ~/ paths are always valid
    if p.startswith('~'):
        return True
    # Common absolute path prefixes (macOS + Linux)
    if _PATH_PREFIX_RE.match(p):
        return True
    return False


def extract_referenced_paths(content: str) -> List[str]:
    """Extract filesystem paths referenced in config content."""
    paths = []
    seen = set()

    def _add(p: str) -> None:
        if p not in seen and _looks_like_filesystem_path(p):
            seen.add(p)
            paths.append(p)

    for match in re.finditer(r'(?:~|/)[/\w._-]+(?:/[\w._-]+)+', content):
        end = match.end()
        trailing = content[end:end + 3]
        if re.search(r'[/<].*<', trailing) or trailing.startswith('<'):
            continue
        _add(match.group())

    for match in re.finditer(r'`([~./][^`\s]+)`', content):
        candidate = match.group(1)
        if '/' in candidate:
            _add(candidate)

    return paths


def count_directives(content: str) -> int:
    """Count bullet-point directive lines."""
    count = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            count += 1
    return count


_RULE_INDICATORS = [
    re.compile(r'\b(must|always|should|shall)\b', re.IGNORECASE),
    re.compile(r'(してください|すること|必ず|常に)'),
]

_VAGUE_PATTERNS = [
    (re.compile(r'適切に'), "\"appropriately\""),
    (re.compile(r'必要に応じて'), "\"as needed\""),
    (re.compile(r'適宜'), "\"as appropriate\""),
    (re.compile(r'なるべく'), "\"preferably\""),
    (re.compile(r'できれば'), "\"if possible\""),
    (re.compile(r'\bif (?:necessary|needed|appropriate)\b', re.IGNORECASE),
     "\"if necessary\""),
    (re.compile(r'\bas (?:needed|appropriate)\b', re.IGNORECASE),
     "\"as needed\""),
    (re.compile(r'\bwhen (?:possible|appropriate)\b', re.IGNORECASE),
     "\"when possible\""),
]

_NEGATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'しない', r'使わない', r'避ける', r'禁止',
        r'不要', r'使用しない', r'行わない', r'含めない',
        r"\bdon'?t\b", r'\bnever\b', r'\bavoid\b',
        r'\bdo not\b', r'\bnot use\b', r'\bprohibit',
    ]
]


def count_negations(content: str) -> int:
    """Count negative directive lines (English and Japanese)."""
    count = 0
    for line in content.splitlines():
        for pattern in _NEGATION_PATTERNS:
            if pattern.search(line):
                count += 1
                break
    return count


def analyze_file(path: Path, category: str) -> FileMetrics:
    """Collect metrics for a single file."""
    if not path.exists():
        return FileMetrics(
            path=str(path), category=category, exists=False,
            size_bytes=0, line_count=0, estimated_tokens=0,
            last_modified=None, sections=[], referenced_paths=[],
            directive_count=0, negation_count=0, content="",
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except (PermissionError, OSError):
        return FileMetrics(
            path=str(path), category=category, exists=True,
            size_bytes=0, line_count=0, estimated_tokens=0,
            last_modified=None, sections=[], referenced_paths=[],
            directive_count=0, negation_count=0, content="",
        )

    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return FileMetrics(
        path=str(path), category=category, exists=True,
        size_bytes=stat.st_size,
        line_count=len(content.splitlines()),
        estimated_tokens=estimate_tokens(content),
        last_modified=mtime.isoformat(),
        sections=extract_sections(content),
        referenced_paths=extract_referenced_paths(content),
        directive_count=count_directives(content),
        negation_count=count_negations(content),
        content=content,
    )


# =============================================================================
# Project Discovery
# =============================================================================

def path_from_project_dir_name(dir_name: str) -> str:
    """Convert a ~/.claude/projects/ subdirectory name back to a real path.

    Example: -Users-alice-projects -> /Users/alice/projects
    """
    return "/" + dir_name.lstrip("-").replace("-", "/")


def discover_projects(claude_home: Path) -> List[str]:
    """Discover project directories from ~/.claude/projects/ entries.

    Reverse-maps each subdirectory name to a real path, then checks for
    project markers (.git, package.json, etc.) in the directory and its
    immediate children.

    Returns: sorted list of existing project directory paths.
    """
    projects_dir = claude_home / "projects"
    if not projects_dir.exists():
        return []

    found = set()

    for entry in projects_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        real_path = path_from_project_dir_name(entry.name)
        real_dir = Path(real_path)

        if not real_dir.exists():
            continue

        # Check if this directory itself is a project
        project_markers = [
            "CLAUDE.md", ".git", "package.json", "pyproject.toml",
            "Cargo.toml", "go.mod", "Makefile", "setup.py", "pom.xml",
        ]
        is_project = any((real_dir / m).exists() for m in project_markers)

        if is_project:
            found.add(str(real_dir))

        # Check immediate children for project markers
        try:
            for child in real_dir.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if any((child / m).exists() for m in project_markers):
                    found.add(str(child))
        except PermissionError:
            pass

    return sorted(found)


# =============================================================================
# File Collection
# =============================================================================

def collect_config_files(claude_home: Optional[Path] = None) -> Dict[str, List[FileMetrics]]:
    """Collect all config files under ~/.claude/ and compute metrics."""
    if claude_home is None:
        claude_home = Path.home() / ".claude"

    files: Dict[str, List[FileMetrics]] = {
        "global_claude_md": [],
        "rules": [],
        "skills": [],
        "memories": [],
        "project_claude_mds": [],
        "settings": [],
    }

    # Global CLAUDE.md
    files["global_claude_md"].append(
        analyze_file(claude_home / "CLAUDE.md", "global_claude_md")
    )

    # Global rules
    rules_dir = claude_home / "rules"
    if rules_dir.exists():
        for f in sorted(rules_dir.glob("*.md")):
            files["rules"].append(analyze_file(f, "rule"))

    # Global skills
    skills_dir = claude_home / "skills"
    if skills_dir.exists():
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            files["skills"].append(analyze_file(skill_md, "skill"))

    # Project memories
    projects_dir = claude_home / "projects"
    if projects_dir.exists():
        for mem in sorted(projects_dir.glob("*/memory/MEMORY.md")):
            files["memories"].append(analyze_file(mem, "memory"))

    # Project CLAUDE.md files
    seen: set = set()

    def _add_project_md(md_path: Path) -> None:
        try:
            key = str(md_path.resolve())
        except OSError:
            key = str(md_path)
        if key not in seen and md_path.exists():
            seen.add(key)
            files["project_claude_mds"].append(
                analyze_file(md_path, "project_claude_md")
            )

    # Reverse-map ~/.claude/projects/ entries to find CLAUDE.md files
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir() or proj_dir.name.startswith("."):
                continue
            real_path = path_from_project_dir_name(proj_dir.name)
            real_dir = Path(real_path)
            if real_dir.exists():
                _add_project_md(real_dir / "CLAUDE.md")
                try:
                    for sub in real_dir.iterdir():
                        if sub.is_dir() and not sub.name.startswith("."):
                            _add_project_md(sub / "CLAUDE.md")
                except PermissionError:
                    pass

    # Also check discovered project directories
    for proj_dir in discover_projects(claude_home):
        _add_project_md(Path(proj_dir) / "CLAUDE.md")

    # Settings
    for name in ["settings.json", "settings.local.json"]:
        settings_file = claude_home / name
        if settings_file.exists():
            files["settings"].append(analyze_file(settings_file, "settings"))

    return files


# =============================================================================
# Context Detection
# =============================================================================

def classify_maturity(total_tokens: int) -> str:
    """Classify config maturity based on total token count."""
    if total_tokens < MATURITY_SPARSE_LIMIT:
        return "sparse"
    elif total_tokens < MATURITY_MODERATE_LIMIT:
        return "moderate"
    else:
        return "rich"


CONTENT_CATEGORIES = [
    "global_claude_md", "rules", "skills", "memories", "project_claude_mds",
]


def iter_content_files(
    files: Dict[str, List[FileMetrics]],
    categories: Optional[List[str]] = None,
) -> List[FileMetrics]:
    """Collect FileMetrics from specified categories (default: all content categories)."""
    if categories is None:
        categories = CONTENT_CATEGORIES
    return [fm for cat in categories for fm in files.get(cat, [])]

def detect_context(
    files: Dict[str, List[FileMetrics]],
    claude_home: Optional[Path] = None,
) -> UserContext:
    """Determine user context from collected files."""
    if claude_home is None:
        claude_home = Path.home() / ".claude"

    project_dirs = discover_projects(claude_home)
    project_count = max(len(project_dirs), len(files["project_claude_mds"]))

    # Total tokens (excluding settings)
    total_tokens = 0
    for category, file_list in files.items():
        if category == "settings":
            continue
        for f in file_list:
            total_tokens += f.estimated_tokens

    config_maturity = classify_maturity(total_tokens)

    setup_type = "multi_project" if project_count >= 2 else "single_project"

    memory_line_count = 0
    for mem in files["memories"]:
        memory_line_count = max(memory_line_count, mem.line_count)

    return UserContext(
        project_count=project_count,
        total_config_tokens=total_tokens,
        config_maturity=config_maturity,
        setup_type=setup_type,
        has_custom_skills=len(files["skills"]) > 0,
        has_custom_rules=len(files["rules"]) > 0,
        has_project_memory=len(files["memories"]) > 0,
        memory_line_count=memory_line_count,
        rule_count=len(files["rules"]),
        skill_count=len(files["skills"]),
        project_dirs=project_dirs,
    )


# =============================================================================
# Check Interface
# =============================================================================

DEFAULT_CONTEXT_WINDOW = 200_000
MEMORY_LINE_LIMIT = 200

# --- Threshold Constants ---
BUDGET_ERROR_RATIO = 0.05
BUDGET_WARNING_RATIO = 0.01
BUDGET_FILE_TOKEN_LIMIT = 1000
MEMORY_WARNING_LINES = 180
MEMORY_INFO_LINES = 150
FRESHNESS_WARNING_DAYS = 180
FRESHNESS_INFO_DAYS = 90
NEGATION_WARNING_COUNT = 5
NEGATION_INFO_COUNT = 3
MATURITY_SPARSE_LIMIT = 500
MATURITY_MODERATE_LIMIT = 3000
PENALTY_ERROR = 0.3
PENALTY_WARNING = 0.15
PENALTY_INFO = 0.05


class Check(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def display_name(self) -> str:
        ...

    @abstractmethod
    def is_applicable(self, ctx: UserContext) -> bool:
        ...

    @abstractmethod
    def run(
        self,
        files: Dict[str, List[FileMetrics]],
        ctx: UserContext,
        context_window: int,
    ) -> CheckResult:
        ...

    def _make_result(
        self, findings: List[Finding], applicable: bool = True
    ) -> CheckResult:
        if not applicable:
            return CheckResult(self.name(), self.display_name(), [], 1.0, False)
        if not findings:
            return CheckResult(self.name(), self.display_name(), [], 1.0, True)

        penalty = 0.0
        for f in findings:
            if f.severity == "error":
                penalty += PENALTY_ERROR
            elif f.severity == "warning":
                penalty += PENALTY_WARNING
            elif f.severity == "info":
                penalty += PENALTY_INFO
        score = max(0.0, 1.0 - penalty)
        return CheckResult(self.name(), self.display_name(), findings, score, True)


# =============================================================================
# Check Implementations
# =============================================================================

class BudgetCheck(Check):
    """Evaluate total token budget of all config files."""

    def name(self) -> str:
        return "budget"

    def display_name(self) -> str:
        return "Budget"

    def is_applicable(self, ctx: UserContext) -> bool:
        return True

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []
        total = ctx.total_config_tokens
        ratio = total / context_window

        if ratio > BUDGET_ERROR_RATIO:
            findings.append(Finding(
                severity="error", dimension="budget",
                message="Config tokens exceed %.0f%% of context window "
                        "(%d tokens, %.1f%%)"
                        % (BUDGET_ERROR_RATIO * 100, total, ratio * 100),
                file_path="(total)",
                suggestion="Reduce config size by removing unnecessary content",
            ))
        elif ratio > BUDGET_WARNING_RATIO:
            findings.append(Finding(
                severity="warning", dimension="budget",
                message="Config tokens exceed %.0f%% of context window "
                        "(%d tokens, %.1f%%)"
                        % (BUDGET_WARNING_RATIO * 100, total, ratio * 100),
                file_path="(total)",
                suggestion="Config is growing. Consider simplifying",
            ))

        for f in iter_content_files(files):
            if f.estimated_tokens > BUDGET_FILE_TOKEN_LIMIT:
                findings.append(Finding(
                    severity="info", dimension="budget",
                    message="Single file exceeds %d tokens (%d tokens)"
                            % (BUDGET_FILE_TOKEN_LIMIT, f.estimated_tokens),
                    file_path=f.path,
                    suggestion="Consider splitting or simplifying this file",
                ))

        for mem in files.get("memories", []):
            if mem.line_count > MEMORY_WARNING_LINES:
                findings.append(Finding(
                    severity="warning", dimension="budget",
                    message="MEMORY.md approaching limit (%d/%d lines)"
                            % (mem.line_count, MEMORY_LINE_LIMIT),
                    file_path=mem.path,
                    suggestion="Clean up outdated information",
                ))
            elif mem.line_count > MEMORY_INFO_LINES:
                findings.append(Finding(
                    severity="info", dimension="budget",
                    message="MEMORY.md at %d/%d lines"
                            % (mem.line_count, MEMORY_LINE_LIMIT),
                    file_path=mem.path,
                ))

        return self._make_result(findings)


class RedundancyCheck(Check):
    """Detect duplicated directives across config layers."""

    def name(self) -> str:
        return "redundancy"

    def display_name(self) -> str:
        return "Redundancy"

    def is_applicable(self, ctx: UserContext) -> bool:
        return ctx.config_maturity != "sparse"

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []

        directives_by_norm: Dict[str, List[Tuple[str, str, int]]] = {}

        for fm in iter_content_files(
            files, ["global_claude_md", "rules", "memories", "project_claude_mds"]
        ):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                stripped = line.strip()
                if not stripped.startswith(("- ", "* ")):
                    continue
                normalized = stripped.lstrip("-* ").strip()
                normalized = re.sub(r'[\s\u3000\u3001\u3002.,]+', '', normalized)
                normalized = normalized.lower()
                if len(normalized) < 5:
                    continue
                if normalized not in directives_by_norm:
                    directives_by_norm[normalized] = []
                directives_by_norm[normalized].append((fm.path, stripped, i))

        for normalized, occurrences in directives_by_norm.items():
            unique_files = set(occ[0] for occ in occurrences)
            if len(unique_files) >= 2:
                locations = ["%s:%d" % (occ[0], occ[2]) for occ in occurrences]
                original_text = occurrences[0][1]
                findings.append(Finding(
                    severity="warning", dimension="redundancy",
                    message="Duplicated directive across %d files: \"%s\""
                            % (len(unique_files), original_text),
                    file_path=locations[0],
                    suggestion="Keep in the most appropriate level. "
                               "Locations: %s" % ", ".join(locations),
                ))

        return self._make_result(findings)


class ScopeFitnessCheck(Check):
    """Check whether directives are placed at the appropriate scope level."""

    def name(self) -> str:
        return "scope_fitness"

    def display_name(self) -> str:
        return "Scope Fitness"

    def is_applicable(self, ctx: UserContext) -> bool:
        return ctx.config_maturity != "sparse"

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []

        project_names = set()
        for d in ctx.project_dirs:
            name = Path(d).name
            if len(name) > 3:
                project_names.add(name)

        # Build word-boundary patterns for project names
        project_pats = {
            pname: re.compile(r'\b' + re.escape(pname) + r'\b')
            for pname in project_names
        }

        # Check global CLAUDE.md for project-specific keywords
        for fm in files.get("global_claude_md", []):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                for pname, pat in project_pats.items():
                    if pat.search(line):
                        findings.append(Finding(
                            severity="warning", dimension="scope_fitness",
                            message="Global CLAUDE.md contains project-specific "
                                    "reference \"%s\" (project: %s)"
                                    % (line.strip(), pname),
                            file_path=fm.path, line_number=i,
                            suggestion="Move to project-level CLAUDE.md",
                        ))

        # Check rules/ for project-specific references
        for fm in files.get("rules", []):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                for pname, pat in project_pats.items():
                    if pat.search(line):
                        findings.append(Finding(
                            severity="info", dimension="scope_fitness",
                            message="Global rule references project: \"%s\""
                                    % line.strip(),
                            file_path=fm.path, line_number=i,
                        ))

        # Check MEMORY.md for rule-like directives that belong elsewhere
        for fm in files.get("memories", []):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                stripped = line.strip()
                if not stripped.startswith(("- ", "* ")):
                    continue
                if stripped.startswith(("#", "`")):
                    continue
                for pattern in _RULE_INDICATORS:
                    if pattern.search(stripped):
                        findings.append(Finding(
                            severity="info", dimension="scope_fitness",
                            message="MEMORY.md contains rule-like directive: "
                                    "\"%s\"" % stripped,
                            file_path=fm.path, line_number=i,
                            suggestion="Rules belong in CLAUDE.md or rules/. "
                                       "MEMORY.md should store facts only",
                        ))
                        break

        return self._make_result(findings)


class FreshnessCheck(Check):
    """Check that config references are still valid and up-to-date."""

    def name(self) -> str:
        return "freshness"

    def display_name(self) -> str:
        return "Freshness"

    def is_applicable(self, ctx: UserContext) -> bool:
        return True

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []
        now = datetime.now(tz=timezone.utc)

        for fm in iter_content_files(files):
            if not fm.exists:
                continue

            for ref_path in fm.referenced_paths:
                expanded = os.path.expanduser(ref_path)
                if not os.path.exists(expanded):
                    if '*' in ref_path or '{' in ref_path:
                        continue
                    if not ref_path.startswith(('/', '~')):
                        continue
                    findings.append(Finding(
                        severity="warning", dimension="freshness",
                        message="Referenced path does not exist: %s" % ref_path,
                        file_path=fm.path,
                        suggestion="Verify the path or remove the stale reference",
                    ))

            if fm.last_modified:
                mtime = datetime.fromisoformat(fm.last_modified)
                days_old = (now - mtime).days
                if days_old > FRESHNESS_WARNING_DAYS:
                    findings.append(Finding(
                        severity="warning", dimension="freshness",
                        message="Not updated in %d days" % days_old,
                        file_path=fm.path,
                        suggestion="Review whether content is still accurate",
                    ))
                elif days_old > FRESHNESS_INFO_DAYS:
                    findings.append(Finding(
                        severity="info", dimension="freshness",
                        message="Not updated in %d days" % days_old,
                        file_path=fm.path,
                    ))

        for mem in files.get("memories", []):
            if not mem.exists:
                continue
            for ref_path in mem.referenced_paths:
                expanded = os.path.expanduser(ref_path)
                if "/projects/" in ref_path and not os.path.exists(expanded):
                    if '*' not in ref_path:
                        findings.append(Finding(
                            severity="warning", dimension="freshness",
                            message="Project in MEMORY.md no longer exists: %s"
                                    % ref_path,
                            file_path=mem.path,
                            suggestion="Remove stale project entries from MEMORY.md",
                        ))

        return self._make_result(findings)


class ConflictsCheck(Check):
    """Detect contradictory directives across config layers."""

    def name(self) -> str:
        return "conflicts"

    def display_name(self) -> str:
        return "Conflicts"

    def is_applicable(self, ctx: UserContext) -> bool:
        return ctx.config_maturity != "sparse"

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []

        topic_patterns = {
            "response_language": [
                (r'日本語で(?:回答|応答|返答)', "Japanese"),
                (r'(?:英語|English)で(?:回答|応答|返答)', "English"),
                (r'[Rr]espond in [Jj]apanese', "Japanese"),
                (r'[Rr]espond in [Ee]nglish', "English"),
                (r'[Aa]nswer in [Jj]apanese', "Japanese"),
                (r'[Aa]nswer in [Ee]nglish', "English"),
            ],
            "writing_style": [
                (r'です・ます', "desu-masu"),
                (r'だ・である', "da-dearu"),
                (r'である調', "dearu"),
            ],
            "commit_format": [
                (r'[Cc]onventional [Cc]ommits?', "Conventional Commits"),
                (r'(?:gitmoji|Gitmoji)', "Gitmoji"),
                (r'[Aa]ngular.*commit', "Angular"),
            ],
        }

        topic_values: Dict[str, List[Tuple[str, str, int]]] = {}
        for fm in iter_content_files(
            files, ["global_claude_md", "rules", "memories", "project_claude_mds"]
        ):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                for topic, patterns in topic_patterns.items():
                    for pattern, value in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            if topic not in topic_values:
                                topic_values[topic] = []
                            topic_values[topic].append((value, fm.path, i))

        for topic, values in topic_values.items():
            unique_values = set(v[0] for v in values)
            if len(unique_values) >= 2:
                locations = ["%s:%d (%s)" % (v[1], v[2], v[0]) for v in values]
                findings.append(Finding(
                    severity="error", dimension="conflicts",
                    message="Conflicting directives for \"%s\": %s"
                            % (topic, ", ".join(sorted(unique_values))),
                    file_path=values[0][1],
                    suggestion="Unify to a single value. Locations: %s"
                               % "; ".join(locations),
                ))

        return self._make_result(findings)


class CoverageCheck(Check):
    """Detect missing or under-specified configuration."""

    def name(self) -> str:
        return "coverage"

    def display_name(self) -> str:
        return "Coverage"

    def is_applicable(self, ctx: UserContext) -> bool:
        return True

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []

        # Check for global CLAUDE.md existence
        global_mds = files.get("global_claude_md", [])
        if not global_mds or not global_mds[0].exists:
            findings.append(Finding(
                severity="error", dimension="coverage",
                message="Global CLAUDE.md does not exist",
                file_path="~/.claude/CLAUDE.md",
                suggestion="Create CLAUDE.md with response language, "
                           "style, and basic rules",
            ))
        else:
            gmd = global_mds[0]
            has_language = bool(re.search(
                r'日本語|英語|[Jj]apanese|[Ee]nglish|[Ll]anguage|[Ll]ang',
                gmd.content
            ))
            has_style = bool(re.search(
                r'です・ます|だ・である|文体|[Tt]one|[Ss]tyle|[Ff]ormal|[Cc]asual',
                gmd.content
            ))
            has_commit = bool(re.search(
                r'[Cc]ommit|コミット', gmd.content
            ))

            if not has_language:
                findings.append(Finding(
                    severity="warning", dimension="coverage",
                    message="No response language specified",
                    file_path=gmd.path,
                    suggestion="Add a language directive "
                               "(e.g., \"Respond in English\")",
                ))
            if not has_style:
                findings.append(Finding(
                    severity="info", dimension="coverage",
                    message="No writing style specified",
                    file_path=gmd.path,
                    suggestion="Consider specifying a tone or style preference",
                ))
            if not has_commit:
                findings.append(Finding(
                    severity="info", dimension="coverage",
                    message="No commit message format specified",
                    file_path=gmd.path,
                ))

        # Config density relative to project count
        if ctx.project_count >= 3 and ctx.total_config_tokens < 50 * ctx.project_count:
            findings.append(Finding(
                severity="warning", dimension="coverage",
                message="Config may be under-specified for %d projects "
                        "(%d tokens, %.0f tokens/project)"
                        % (ctx.project_count, ctx.total_config_tokens,
                           ctx.total_config_tokens / ctx.project_count),
                file_path="(total)",
                suggestion="Consider adding project-level CLAUDE.md files "
                           "with build/test instructions",
            ))

        # Detect projects without CLAUDE.md
        projects_with_md = set()
        for fm in files.get("project_claude_mds", []):
            if fm.exists:
                projects_with_md.add(str(Path(fm.path).parent))

        for proj_dir in ctx.project_dirs:
            if proj_dir not in projects_with_md:
                findings.append(Finding(
                    severity="info", dimension="coverage",
                    message="No CLAUDE.md found",
                    file_path=proj_dir,
                    suggestion="Add CLAUDE.md with build/test commands "
                               "to speed up session startup",
                ))

        # No custom skills or rules at all
        if (not ctx.has_custom_skills and not ctx.has_custom_rules
                and ctx.project_count >= 2):
            findings.append(Finding(
                severity="info", dimension="coverage",
                message="No custom skills or rules defined",
                file_path="~/.claude/",
                suggestion="Extract recurring workflows into skills/ "
                           "or file-specific patterns into rules/",
            ))

        return self._make_result(findings)


class EffectivenessProxyCheck(Check):
    """Evaluate config effectiveness via proxy signals."""

    def name(self) -> str:
        return "effectiveness_proxy"

    def display_name(self) -> str:
        return "Effectiveness"

    def is_applicable(self, ctx: UserContext) -> bool:
        return ctx.config_maturity != "sparse"

    def run(self, files: Dict[str, List[FileMetrics]], ctx: UserContext,
            context_window: int) -> CheckResult:
        findings = []

        total_negations = 0
        negation_locations = []
        for fm in iter_content_files(
            files, ["global_claude_md", "rules", "memories", "project_claude_mds"]
        ):
            if fm.negation_count > 0:
                total_negations += fm.negation_count
                negation_locations.append(
                    "%s (%d)" % (fm.path, fm.negation_count)
                )

        if total_negations > NEGATION_WARNING_COUNT:
            findings.append(Finding(
                severity="warning", dimension="effectiveness_proxy",
                message="High number of negative directives (%d)"
                        % total_negations,
                file_path="(total)",
                suggestion="Rephrase negatives as positives for clarity. "
                           "Locations: %s" % ", ".join(negation_locations),
            ))
        elif total_negations > NEGATION_INFO_COUNT:
            findings.append(Finding(
                severity="info", dimension="effectiveness_proxy",
                message="%d negative directives found" % total_negations,
                file_path="(total)",
            ))

        vague_count = 0
        for fm in iter_content_files(
            files, ["global_claude_md", "rules", "project_claude_mds"]
        ):
            if not fm.exists:
                continue
            for i, line in enumerate(fm.content.splitlines(), 1):
                for pattern, label in _VAGUE_PATTERNS:
                    if pattern.search(line):
                        vague_count += 1
                        if vague_count <= 3:
                            findings.append(Finding(
                                severity="info",
                                dimension="effectiveness_proxy",
                                message="Vague expression %s: \"%s\""
                                        % (label, line.strip()),
                                file_path=fm.path, line_number=i,
                                suggestion="Replace with specific criteria "
                                           "for more reliable behavior",
                            ))
                        break

        if vague_count > 3:
            findings.append(Finding(
                severity="info", dimension="effectiveness_proxy",
                message="%d more vague expressions found"
                        % (vague_count - 3),
                file_path="(total)",
            ))

        return self._make_result(findings)


# =============================================================================
# Check Registry & Scoring
# =============================================================================

ALL_CHECKS: List[Check] = [
    BudgetCheck(),
    RedundancyCheck(),
    ScopeFitnessCheck(),
    FreshnessCheck(),
    ConflictsCheck(),
    CoverageCheck(),
    EffectivenessProxyCheck(),
]

WEIGHTS: Dict[str, float] = {
    "budget": 1.0,
    "redundancy": 0.5,
    "scope_fitness": 1.0,
    "freshness": 1.5,
    "conflicts": 1.5,
    "coverage": 1.0,
    "effectiveness_proxy": 0.5,
}


def calculate_overall(results: List[CheckResult]) -> Tuple[float, str]:
    """Calculate weighted overall score and rating."""
    applicable = [
        (r, WEIGHTS.get(r.check_name, 1.0))
        for r in results if r.applicable
    ]
    if not applicable:
        return 1.0, "A"

    weighted_sum = sum(r.score * w for r, w in applicable)
    total_weight = sum(w for _, w in applicable)
    score = weighted_sum / total_weight

    if score >= 0.90:
        rating = "A"
    elif score >= 0.70:
        rating = "B"
    elif score >= 0.50:
        rating = "C"
    elif score >= 0.30:
        rating = "D"
    else:
        rating = "F"

    return score, rating


# =============================================================================
# CLI Output Formatting
# =============================================================================

BOX_TL = "\u256d"  # ╭
BOX_TR = "\u256e"  # ╮
BOX_BL = "\u2570"  # ╰
BOX_BR = "\u2571"  # (using ╯ below)
BOX_BR = "\u256f"  # ╯
BOX_H = "\u2500"   # ─
BOX_V = "\u2502"   # │
BAR_FULL = "\u2588"   # █
BAR_EMPTY = "\u2591"  # ░


def _score_bar(score: float, width: int = 20) -> str:
    """Render score as a visual bar."""
    filled = int(score * width)
    empty = width - filled

    if score >= 0.9:
        color = Style.GREEN
    elif score >= 0.7:
        color = Style.YELLOW
    else:
        color = Style.RED

    return color + BAR_FULL * filled + Style.DIM + BAR_EMPTY * empty + Style.RESET


def _score_indicator(score: float) -> str:
    """Status label based on score."""
    if score >= 0.9:
        return Style.GREEN + "pass" + Style.RESET
    elif score >= 0.7:
        return Style.YELLOW + "warn" + Style.RESET
    else:
        return Style.RED + "FAIL" + Style.RESET


def _severity_icon(severity: str) -> str:
    """Severity icon."""
    if severity == "error":
        return Style.RED + Style.BOLD + "ERR " + Style.RESET
    elif severity == "warning":
        return Style.YELLOW + "WARN" + Style.RESET
    else:
        return Style.CYAN + "info" + Style.RESET


def _rating_style(rating: str) -> str:
    """Colorize rating letter."""
    colors = {"A": Style.GREEN, "B": Style.GREEN, "C": Style.YELLOW,
              "D": Style.RED, "F": Style.RED}
    return colors.get(rating, "") + Style.BOLD + rating + Style.RESET


def _shorten_path(path: str) -> str:
    """Shorten path by replacing home directory with ~."""
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def format_report(
    results: List[CheckResult],
    files: Dict[str, List[FileMetrics]],
    ctx: UserContext,
    overall_score: float,
    overall_rating: str,
    context_window: int,
) -> str:
    """Generate a formatted CLI report."""
    s = Style
    out = []

    total_tokens = ctx.total_config_tokens
    token_pct = total_tokens / context_window * 100

    total_files = sum(
        len(fl) for cat, fl in files.items() if cat != "settings"
    )
    errors = sum(1 for r in results for f in r.findings if f.severity == "error")
    warnings = sum(1 for r in results for f in r.findings if f.severity == "warning")
    infos = sum(1 for r in results for f in r.findings if f.severity == "info")
    total_findings = errors + warnings + infos

    # ── Header ──────────────────────────────────────────────
    w = 52
    out.append("")
    out.append(s.DIM + "  " + BOX_TL + BOX_H * w + BOX_TR + s.RESET)
    out.append(s.DIM + "  " + BOX_V + s.RESET
               + s.BOLD + "  Claude Doctor"
               + " " * (w - 14) + s.RESET
               + s.DIM + BOX_V + s.RESET)
    out.append(s.DIM + "  " + BOX_BL + BOX_H * w + BOX_BR + s.RESET)
    out.append("")

    # ── Score Overview ──────────────────────────────────────
    score_pct = int(overall_score * 100)
    out.append("  " + s.BOLD + "Score" + s.RESET
               + "   " + _score_bar(overall_score, 24)
               + "  " + s.BOLD + "%d" % score_pct + s.RESET
               + "/100  " + _rating_style(overall_rating))
    out.append("")

    out.append("  " + s.DIM + "Tokens" + s.RESET
               + "  ~%s / %s (%.1f%%)"
               % ("{:,}".format(total_tokens),
                  "{:,}".format(context_window),
                  token_pct))

    out.append("  " + s.DIM + "Files " + s.RESET
               + "  %d config files across %d projects"
               % (total_files, ctx.project_count))

    if total_findings > 0:
        parts = []
        if errors:
            parts.append(s.RED + "%d errors" % errors + s.RESET)
        if warnings:
            parts.append(s.YELLOW + "%d warnings" % warnings + s.RESET)
        if infos:
            parts.append(s.CYAN + "%d infos" % infos + s.RESET)
        out.append("  " + s.DIM + "Issues" + s.RESET + "  " + ", ".join(parts))
    out.append("")

    # ── Context ─────────────────────────────────────────────
    out.append(s.DIM + "  " + BOX_H * 3 + " Context "
               + BOX_H * (w - 8) + s.RESET)
    out.append("")

    density_colors = {"sparse": s.YELLOW, "moderate": s.GREEN, "rich": s.CYAN}
    density_color = density_colors.get(ctx.config_maturity, "")
    out.append("    Setup     %s (%d projects)"
               % (ctx.setup_type.replace("_", " "), ctx.project_count))
    out.append("    Density   %s%s%s"
               % (density_color, ctx.config_maturity, s.RESET))
    out.append("    Skills    %d    Rules  %d"
               % (ctx.skill_count, ctx.rule_count))
    if ctx.has_project_memory:
        mem_pct = ctx.memory_line_count / MEMORY_LINE_LIMIT * 100
        out.append("    Memory    %d/%d lines (%.0f%%)"
                   % (ctx.memory_line_count, MEMORY_LINE_LIMIT, mem_pct))
    out.append("")

    # ── Dimension Scores ────────────────────────────────────
    out.append(s.DIM + "  " + BOX_H * 3 + " Dimensions "
               + BOX_H * (w - 10) + s.RESET)
    out.append("")

    name_width = max(len(r.display_name) for r in results) + 1
    for r in results:
        name_padded = r.display_name.ljust(name_width)
        if r.applicable:
            score_str = "%3d" % int(r.score * 100)
            out.append("    %s %s  %s  %s"
                       % (name_padded,
                          _score_bar(r.score, 20),
                          score_str,
                          _score_indicator(r.score)))
        else:
            out.append("    %s %s  %s  %s"
                       % (name_padded,
                          s.DIM + BAR_EMPTY * 20 + s.RESET,
                          s.DIM + "  -" + s.RESET,
                          s.DIM + "skip" + s.RESET))
    out.append("")

    # ── Findings ────────────────────────────────────────────
    if total_findings > 0:
        out.append(s.DIM + "  " + BOX_H * 3 + " Findings "
                   + BOX_H * (w - 9) + s.RESET)
        out.append("")

        # Sort: errors first, then warnings, then infos
        all_findings = []
        for r in results:
            for f in r.findings:
                all_findings.append(f)

        severity_order = {"error": 0, "warning": 1, "info": 2}
        all_findings.sort(key=lambda f: severity_order.get(f.severity, 9))

        for f in all_findings:
            icon = _severity_icon(f.severity)
            out.append("    %s  %s" % (icon, f.message))
            shortened = _shorten_path(f.file_path)
            out.append("    " + s.DIM + "      " + shortened + s.RESET
                       + ((":%d" % f.line_number) if f.line_number else ""))
            if f.suggestion:
                out.append("    " + s.DIM + "      -> "
                           + f.suggestion + s.RESET)
            out.append("")

    # ── Footer ──────────────────────────────────────────────
    out.append(s.DIM + "  " + BOX_H * (w + 2) + s.RESET)

    # Files inventory
    file_categories = [
        ("global_claude_md", "CLAUDE.md"),
        ("rules", "rules/"),
        ("skills", "skills/"),
        ("memories", "memory/"),
        ("project_claude_mds", "project CLAUDE.md"),
    ]
    inventory_parts = []
    for cat, label in file_categories:
        count = len([f for f in files.get(cat, []) if f.exists])
        if count > 0:
            inventory_parts.append("%d %s" % (count, label))

    if inventory_parts:
        out.append(s.DIM + "  Scanned: " + ", ".join(inventory_parts) + s.RESET)

    out.append("")
    return "\n".join(out)


def format_json(
    results: List[CheckResult],
    files: Dict[str, List[FileMetrics]],
    ctx: UserContext,
    overall_score: float,
    overall_rating: str,
    context_window: int,
) -> str:
    """Output all diagnostic data in JSON format."""

    def finding_to_dict(f: Finding) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "severity": f.severity,
            "dimension": f.dimension,
            "message": f.message,
            "file_path": f.file_path,
        }
        if f.line_number is not None:
            d["line_number"] = f.line_number
        if f.suggestion is not None:
            d["suggestion"] = f.suggestion
        return d

    def result_to_dict(r: CheckResult) -> Dict[str, Any]:
        return {
            "check_name": r.check_name,
            "display_name": r.display_name,
            "score": round(r.score, 3),
            "applicable": r.applicable,
            "findings": [finding_to_dict(f) for f in r.findings],
        }

    def file_to_dict(fm: FileMetrics) -> Dict[str, Any]:
        return {
            "path": fm.path,
            "category": fm.category,
            "exists": fm.exists,
            "size_bytes": fm.size_bytes,
            "line_count": fm.line_count,
            "estimated_tokens": fm.estimated_tokens,
            "last_modified": fm.last_modified,
            "sections": fm.sections,
            "directive_count": fm.directive_count,
            "negation_count": fm.negation_count,
        }

    file_metrics = {}
    for category, file_list in files.items():
        file_metrics[category] = [file_to_dict(fm) for fm in file_list]

    output = {
        "overall": {
            "score": round(overall_score, 3),
            "rating": overall_rating,
            "score_percent": int(overall_score * 100),
        },
        "context": {
            "project_count": ctx.project_count,
            "total_config_tokens": ctx.total_config_tokens,
            "context_window": context_window,
            "config_maturity": ctx.config_maturity,
            "setup_type": ctx.setup_type,
            "has_custom_skills": ctx.has_custom_skills,
            "has_custom_rules": ctx.has_custom_rules,
            "has_project_memory": ctx.has_project_memory,
            "memory_line_count": ctx.memory_line_count,
            "rule_count": ctx.rule_count,
            "skill_count": ctx.skill_count,
        },
        "checks": [result_to_dict(r) for r in results],
        "files": file_metrics,
    }

    return json.dumps(output, ensure_ascii=False, indent=2)


# =============================================================================
# Main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="claude-doctor",
        description="Diagnose the health of your Claude Code configuration",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--context-window", type=int, default=200_000,
        help="Context window size in tokens (default: 200000)",
    )
    parser.add_argument(
        "--claude-home", type=str, default=None,
        help="Path to .claude directory (default: ~/.claude)",
    )
    return parser.parse_args(argv)


def run_diagnosis(
    claude_home: Optional[Path] = None,
    context_window: int = 200_000,
) -> Tuple[
    List[CheckResult],
    Dict[str, List[FileMetrics]],
    UserContext,
    float,
    str,
]:
    """Run the full diagnosis and return results (API for tests and integrations)."""
    files = collect_config_files(claude_home)
    ctx = detect_context(files, claude_home)

    results = []
    for check in ALL_CHECKS:
        if check.is_applicable(ctx):
            results.append(check.run(files, ctx, context_window))
        else:
            results.append(CheckResult(
                check.name(), check.display_name(), [], 1.0, False
            ))

    overall_score, overall_rating = calculate_overall(results)
    return results, files, ctx, overall_score, overall_rating


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.no_color or not sys.stdout.isatty() or args.json:
        Style.disable()

    claude_home = Path(args.claude_home) if args.claude_home else None

    results, files, ctx, overall_score, overall_rating = run_diagnosis(
        claude_home, args.context_window
    )

    if args.json:
        print(format_json(results, files, ctx, overall_score,
                          overall_rating, args.context_window))
    else:
        print(format_report(results, files, ctx, overall_score,
                            overall_rating, args.context_window))


if __name__ == "__main__":
    main()
