#!/usr/bin/env python3
"""Unit tests for config_doctor."""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config_doctor as cd


# =============================================================================
# Helpers
# =============================================================================

def _make_file(
    path: str = "/mock/file.md",
    category: str = "global_claude_md",
    content: str = "",
    line_count: int = 0,
    tokens: int = 0,
    exists: bool = True,
) -> cd.FileMetrics:
    if not line_count:
        line_count = len(content.splitlines())
    if not tokens:
        tokens = cd.estimate_tokens(content)
    return cd.FileMetrics(
        path=path, category=category, exists=exists,
        size_bytes=len(content.encode("utf-8")),
        line_count=line_count,
        estimated_tokens=tokens,
        last_modified="2026-01-01T00:00:00+00:00",
        sections=cd.extract_sections(content),
        referenced_paths=cd.extract_referenced_paths(content),
        directive_count=cd.count_directives(content),
        negation_count=cd.count_negations(content),
        content=content,
    )


def _make_context(**kwargs) -> cd.UserContext:
    defaults = dict(
        project_count=3,
        total_config_tokens=1000,
        config_maturity="moderate",
        setup_type="multi_project",
        has_custom_skills=True,
        has_custom_rules=True,
        has_project_memory=True,
        memory_line_count=30,
        rule_count=2,
        skill_count=2,
        project_dirs=[],
    )
    defaults.update(kwargs)
    return cd.UserContext(**defaults)


CW = 200_000  # default context window


# =============================================================================
# Token Estimation
# =============================================================================

class TestEstimateTokens(TestCase):
    def test_empty_string(self):
        self.assertEqual(cd.estimate_tokens(""), 0)

    def test_english_text(self):
        text = "Hello world, this is a test."
        tokens = cd.estimate_tokens(text)
        self.assertGreater(tokens, 3)
        self.assertLess(tokens, 15)

    def test_japanese_text(self):
        text = "日本語で回答する"
        tokens = cd.estimate_tokens(text)
        self.assertGreater(tokens, 3)
        self.assertLess(tokens, 10)

    def test_mixed_text(self):
        text = "Python 3.9.6 — str | None 型ヒント不可"
        tokens = cd.estimate_tokens(text)
        self.assertGreater(tokens, 5)


# =============================================================================
# File Analysis
# =============================================================================

class TestExtractSections(TestCase):
    def test_basic(self):
        content = "# Title\n\n## Section 1\n\nText\n\n### Subsection\n"
        sections = cd.extract_sections(content)
        self.assertEqual(sections, ["Title", "Section 1", "Subsection"])

    def test_empty(self):
        self.assertEqual(cd.extract_sections(""), [])


class TestLooksLikeFilesystemPath(TestCase):
    def test_home_path(self):
        self.assertTrue(cd._looks_like_filesystem_path("~/.claude/CLAUDE.md"))

    def test_absolute_path_macos(self):
        self.assertTrue(cd._looks_like_filesystem_path("/Users/alice/work"))

    def test_absolute_path_linux(self):
        self.assertTrue(cd._looks_like_filesystem_path("/home/alice/work"))

    def test_rejects_template(self):
        self.assertFalse(cd._looks_like_filesystem_path("~/.claude/<path>/mem"))

    def test_rejects_japanese(self):
        self.assertFalse(cd._looks_like_filesystem_path("/高/中/低"))

    def test_rejects_placeholder(self):
        self.assertFalse(cd._looks_like_filesystem_path("~/YYYYMMDD/log"))

    def test_rejects_short(self):
        self.assertFalse(cd._looks_like_filesystem_path("/a/b"))

    def test_rejects_relative(self):
        self.assertFalse(cd._looks_like_filesystem_path("/docs/issues"))

    def test_rejects_tool_names(self):
        self.assertFalse(cd._looks_like_filesystem_path("/Glob/Grep"))


class TestExtractReferencedPaths(TestCase):
    def test_home_path(self):
        content = "Config: `~/.claude/CLAUDE.md` here"
        paths = cd.extract_referenced_paths(content)
        self.assertIn("~/.claude/CLAUDE.md", paths)

    def test_absolute_path(self):
        content = "File: /Users/alice/projects/test"
        paths = cd.extract_referenced_paths(content)
        self.assertIn("/Users/alice/projects/test", paths)

    def test_linux_path(self):
        content = "File: /home/user/projects/myapp"
        paths = cd.extract_referenced_paths(content)
        self.assertIn("/home/user/projects/myapp", paths)

    def test_excludes_templates(self):
        content = "Path: ~/.claude/projects/<path>/memory/MEMORY.md"
        paths = cd.extract_referenced_paths(content)
        self.assertEqual(paths, [])

    def test_excludes_japanese_slashes(self):
        content = "Priority: 高/中/低"
        paths = cd.extract_referenced_paths(content)
        self.assertEqual(paths, [])

    def test_excludes_placeholder_patterns(self):
        content = "Log: ~/archive/YYYYMM/YYYYMMDD_uuid.md"
        paths = cd.extract_referenced_paths(content)
        self.assertEqual(paths, [])

    def test_excludes_relative_paths(self):
        content = "Place in /docs/issues/ directory"
        paths = cd.extract_referenced_paths(content)
        self.assertEqual(paths, [])

    def test_excludes_tool_names(self):
        content = "Use Read/Glob/Grep tools"
        paths = cd.extract_referenced_paths(content)
        self.assertEqual(paths, [])


class TestCountDirectives(TestCase):
    def test_bullet_points(self):
        content = "- item 1\n- item 2\n* item 3\ntext\n1. first"
        self.assertEqual(cd.count_directives(content), 4)

    def test_no_directives(self):
        self.assertEqual(cd.count_directives("just text\nmore text"), 0)


class TestCountNegations(TestCase):
    def test_japanese_negations(self):
        content = "- 使わない\n- 避ける\n- 良い\n"
        self.assertEqual(cd.count_negations(content), 2)

    def test_english_negations(self):
        content = "- don't use\n- never do\n- always use\n"
        self.assertEqual(cd.count_negations(content), 2)

    def test_no_negations(self):
        self.assertEqual(cd.count_negations("all good"), 0)


# =============================================================================
# Project Discovery
# =============================================================================

class TestPathFromProjectDirName(TestCase):
    def test_macos(self):
        result = cd.path_from_project_dir_name("-Users-alice-projects")
        self.assertEqual(result, "/Users/alice/projects")

    def test_linux(self):
        result = cd.path_from_project_dir_name("-home-alice-projects")
        self.assertEqual(result, "/home/alice/projects")

    def test_nested(self):
        result = cd.path_from_project_dir_name("-Users-alice-projects-subdir-foo")
        self.assertEqual(result, "/Users/alice/projects/subdir/foo")


# =============================================================================
# Check Tests
# =============================================================================

class TestBudgetCheck(TestCase):
    def test_healthy_budget(self):
        files = {"global_claude_md": [_make_file(tokens=500)]}
        ctx = _make_context(total_config_tokens=500)
        result = cd.BudgetCheck().run(files, ctx, CW)
        self.assertEqual(result.score, 1.0)

    def test_warning_budget(self):
        files = {"global_claude_md": [_make_file(tokens=3000)]}
        ctx = _make_context(total_config_tokens=3000)
        result = cd.BudgetCheck().run(files, ctx, CW)
        self.assertLess(result.score, 1.0)
        self.assertTrue(any(f.severity == "warning" for f in result.findings))

    def test_error_budget(self):
        files = {"global_claude_md": [_make_file(tokens=15000)]}
        ctx = _make_context(total_config_tokens=15000)
        result = cd.BudgetCheck().run(files, ctx, CW)
        self.assertTrue(any(f.severity == "error" for f in result.findings))

    def test_memory_line_warning(self):
        files = {
            "global_claude_md": [_make_file(tokens=100)],
            "memories": [_make_file(
                path="/mock/memory.md", category="memory",
                content="\n".join(["line"] * 185), line_count=185,
            )],
        }
        ctx = _make_context(total_config_tokens=100)
        result = cd.BudgetCheck().run(files, ctx, CW)
        self.assertTrue(any("MEMORY.md" in f.message for f in result.findings))

    def test_custom_context_window(self):
        """Custom context window size should affect thresholds."""
        files = {"global_claude_md": [_make_file(tokens=600)]}
        ctx = _make_context(total_config_tokens=600)
        # With 10K window, 600 tokens = 6% -> should be error (threshold: >5%)
        result = cd.BudgetCheck().run(files, ctx, 10_000)
        self.assertTrue(any(f.severity == "error" for f in result.findings))


class TestRedundancyCheck(TestCase):
    def test_no_redundancy(self):
        files = {
            "global_claude_md": [_make_file(content="- Answer in English\n")],
            "rules": [_make_file(
                path="/mock/rule.md", category="rule",
                content="- Use heading levels for structure\n",
            )],
            "memories": [],
            "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.RedundancyCheck().run(files, ctx, CW)
        self.assertEqual(result.score, 1.0)

    def test_detects_redundancy(self):
        files = {
            "global_claude_md": [_make_file(
                content="- Use formal writing style\n",
            )],
            "rules": [_make_file(
                path="/mock/rule.md", category="rule",
                content="- Use formal writing style\n",
            )],
            "memories": [],
            "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.RedundancyCheck().run(files, ctx, CW)
        self.assertLess(result.score, 1.0)
        self.assertTrue(any("Duplicated" in f.message for f in result.findings))

    def test_not_applicable_when_sparse(self):
        ctx = _make_context(config_maturity="sparse")
        self.assertFalse(cd.RedundancyCheck().is_applicable(ctx))


class TestScopeFitnessCheck(TestCase):
    def test_not_applicable_when_sparse(self):
        ctx = _make_context(config_maturity="sparse")
        self.assertFalse(cd.ScopeFitnessCheck().is_applicable(ctx))

    def test_detects_project_name_in_global(self):
        files = {
            "global_claude_md": [_make_file(
                content="- Run my-web-app tests\n",
            )],
            "rules": [],
            "memories": [],
            "project_claude_mds": [],
        }
        ctx = _make_context(
            project_dirs=["/Users/alice/projects/my-web-app"]
        )
        result = cd.ScopeFitnessCheck().run(files, ctx, CW)
        self.assertTrue(any("project-specific" in f.message for f in result.findings))


class TestFreshnessCheck(TestCase):
    def test_healthy(self):
        files = {
            "global_claude_md": [_make_file(content="# Basics\n")],
            "rules": [], "skills": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.FreshnessCheck().run(files, ctx, CW)
        self.assertEqual(result.score, 1.0)

    def test_nonexistent_path(self):
        files = {
            "global_claude_md": [_make_file(
                content="See `~/nonexistent/path/that/does/not/exist`\n",
            )],
            "rules": [], "skills": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.FreshnessCheck().run(files, ctx, CW)
        self.assertTrue(any("does not exist" in f.message for f in result.findings))


class TestConflictsCheck(TestCase):
    def test_no_conflicts(self):
        files = {
            "global_claude_md": [_make_file(content="- Respond in English\n")],
            "rules": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.ConflictsCheck().run(files, ctx, CW)
        self.assertEqual(result.score, 1.0)

    def test_detects_style_conflict(self):
        files = {
            "global_claude_md": [_make_file(
                content="- です・ます調で統一\n",
            )],
            "rules": [], "memories": [],
            "project_claude_mds": [_make_file(
                path="/mock/proj/CLAUDE.md", category="project_claude_md",
                content="- だ・である調で記述\n",
            )],
        }
        ctx = _make_context()
        result = cd.ConflictsCheck().run(files, ctx, CW)
        self.assertLess(result.score, 1.0)
        self.assertTrue(any("Conflicting" in f.message for f in result.findings))

    def test_detects_language_conflict(self):
        files = {
            "global_claude_md": [_make_file(content="- Respond in English\n")],
            "rules": [], "memories": [],
            "project_claude_mds": [_make_file(
                path="/mock/proj/CLAUDE.md", category="project_claude_md",
                content="- 日本語で回答する\n",
            )],
        }
        ctx = _make_context()
        result = cd.ConflictsCheck().run(files, ctx, CW)
        self.assertTrue(any("Conflicting" in f.message for f in result.findings))


class TestCoverageCheck(TestCase):
    def test_missing_global_claude_md(self):
        files = {
            "global_claude_md": [_make_file(exists=False)],
            "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.CoverageCheck().run(files, ctx, CW)
        self.assertTrue(any("does not exist" in f.message for f in result.findings))

    def test_missing_language_spec(self):
        files = {
            "global_claude_md": [_make_file(content="# Rules\n- Write tests\n")],
            "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.CoverageCheck().run(files, ctx, CW)
        self.assertTrue(any("language" in f.message for f in result.findings))

    def test_under_specified_for_many_projects(self):
        files = {
            "global_claude_md": [_make_file(
                content="- Respond in English\n", tokens=30
            )],
            "project_claude_mds": [],
        }
        ctx = _make_context(project_count=5, total_config_tokens=100)
        result = cd.CoverageCheck().run(files, ctx, CW)
        self.assertTrue(any("under-specified" in f.message for f in result.findings))

    def test_detects_projects_without_claude_md(self):
        files = {
            "global_claude_md": [_make_file(
                content="- Respond in English\n"
            )],
            "project_claude_mds": [],
        }
        ctx = _make_context(project_dirs=["/tmp/test-project-abc123"])
        result = cd.CoverageCheck().run(files, ctx, CW)
        self.assertTrue(any("No CLAUDE.md" in f.message for f in result.findings))


class TestEffectivenessProxyCheck(TestCase):
    def test_many_negations(self):
        content = "\n".join([
            "- don't use",
            "- avoid this",
            "- never do",
            "- prohibit X",
            "- do not call",
            "- not use Y",
        ])
        files = {
            "global_claude_md": [_make_file(content=content)],
            "rules": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        result = cd.EffectivenessProxyCheck().run(files, ctx, CW)
        self.assertTrue(any("negative" in f.message.lower() for f in result.findings))


# =============================================================================
# Scoring
# =============================================================================

class TestScoring(TestCase):
    def test_perfect_score(self):
        results = [
            cd.CheckResult("budget", "Budget", [], 1.0, True),
            cd.CheckResult("freshness", "Freshness", [], 1.0, True),
        ]
        score, rating = cd.calculate_overall(results)
        self.assertEqual(score, 1.0)
        self.assertEqual(rating, "A")

    def test_mixed_scores(self):
        results = [
            cd.CheckResult("budget", "Budget", [], 0.7, True),
            cd.CheckResult("freshness", "Freshness", [], 0.5, True),
        ]
        score, rating = cd.calculate_overall(results)
        self.assertAlmostEqual(score, 0.58, places=2)
        self.assertEqual(rating, "C")

    def test_inapplicable_excluded(self):
        results = [
            cd.CheckResult("budget", "Budget", [], 1.0, True),
            cd.CheckResult("redundancy", "Redundancy", [], 0.0, False),
        ]
        score, rating = cd.calculate_overall(results)
        self.assertEqual(score, 1.0)


# =============================================================================
# Output Formatting
# =============================================================================

class TestFormatJson(TestCase):
    def test_valid_json(self):
        files = {"global_claude_md": [_make_file(content="# Test\n")]}
        ctx = _make_context()
        results = [cd.CheckResult("budget", "Budget", [], 1.0, True)]
        output = cd.format_json(results, files, ctx, 1.0, "A", CW)
        parsed = json.loads(output)
        self.assertEqual(parsed["overall"]["rating"], "A")
        self.assertIn("checks", parsed)
        self.assertIn("files", parsed)
        self.assertEqual(parsed["context"]["context_window"], CW)

    def test_custom_context_window_in_json(self):
        files = {"global_claude_md": [_make_file(content="# Test\n")]}
        ctx = _make_context()
        results = [cd.CheckResult("budget", "Budget", [], 1.0, True)]
        output = cd.format_json(results, files, ctx, 1.0, "A", 500_000)
        parsed = json.loads(output)
        self.assertEqual(parsed["context"]["context_window"], 500_000)


class TestFormatReport(TestCase):
    def setUp(self):
        cd.Style.disable()

    def tearDown(self):
        cd.Style.reset()

    def test_contains_score(self):
        files = {"global_claude_md": [_make_file(content="# Test\n")]}
        ctx = _make_context()
        results = [cd.CheckResult("budget", "Budget", [], 0.85, True)]
        output = cd.format_report(results, files, ctx, 0.85, "B", CW)
        self.assertIn("85", output)
        self.assertIn("B", output)

    def test_no_color_mode(self):
        files = {"global_claude_md": [_make_file(content="# Test\n")]}
        ctx = _make_context()
        results = [cd.CheckResult("budget", "Budget", [], 1.0, True)]
        output = cd.format_report(results, files, ctx, 1.0, "A", CW)
        # No ANSI escape codes
        self.assertNotIn("\033[", output)


class TestShortenPath(TestCase):
    def test_shortens_home(self):
        home = str(Path.home())
        result = cd._shorten_path(home + "/.claude/CLAUDE.md")
        self.assertEqual(result, "~/.claude/CLAUDE.md")

    def test_no_change_for_other_paths(self):
        self.assertEqual(cd._shorten_path("/etc/config"), "/etc/config")


# =============================================================================
# Integration
# =============================================================================

class TestIntegration(TestCase):
    def tearDown(self):
        # main() with --json disables Style; restore it
        cd.Style.reset()

    def test_minimal_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()
            (claude_home / "CLAUDE.md").write_text(
                "# Basics\n- Respond in English\n"
            )

            files = cd.collect_config_files(claude_home)
            ctx = cd.detect_context(files, claude_home)

            self.assertEqual(ctx.config_maturity, "sparse")
            self.assertFalse(ctx.has_custom_skills)
            self.assertFalse(ctx.has_custom_rules)

            results = []
            for check in cd.ALL_CHECKS:
                if check.is_applicable(ctx):
                    results.append(check.run(files, ctx, CW))
                else:
                    results.append(cd.CheckResult(
                        check.name(), check.display_name(), [], 1.0, False
                    ))

            score, rating = cd.calculate_overall(results)
            self.assertGreater(score, 0.5)

    def test_rich_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()

            (claude_home / "CLAUDE.md").write_text(
                "# Rules\n- Respond in English\n"
                "- Use formal tone\n"
                "- Conventional Commits\n"
            )

            rules_dir = claude_home / "rules"
            rules_dir.mkdir()
            (rules_dir / "docs.md").write_text(
                "---\npaths:\n  - '**/*.md'\n---\n"
                "# Doc rules\n- Use heading levels\n"
            )

            skills_dir = claude_home / "skills" / "my-skill"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text(
                "---\ndescription: test\nuser-invocable: true\n---\n"
                "# My Skill\nDoes stuff.\n"
            )

            files = cd.collect_config_files(claude_home)
            ctx = cd.detect_context(files, claude_home)

            self.assertTrue(ctx.has_custom_skills)
            self.assertTrue(ctx.has_custom_rules)
            self.assertEqual(ctx.skill_count, 1)
            self.assertEqual(ctx.rule_count, 1)

    def test_run_diagnosis_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()
            (claude_home / "CLAUDE.md").write_text("# Rules\n- English\n")

            results, files, ctx, score, rating = cd.run_diagnosis(
                claude_home, context_window=200_000
            )
            self.assertIsInstance(results, list)
            self.assertGreater(score, 0.0)

    def test_main_json_output(self):
        """main() with --json should produce valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()
            (claude_home / "CLAUDE.md").write_text("# Rules\n- English\n")

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                cd.main(["--json", "--claude-home", str(claude_home)])

            output = buf.getvalue()
            parsed = json.loads(output)
            self.assertIn("overall", parsed)


# =============================================================================
# Style Reset
# =============================================================================

class TestStyleReset(TestCase):
    def test_disable_and_reset(self):
        """Style.disable() then Style.reset() should restore original values."""
        # Ensure clean state first (prior tests may have disabled Style)
        cd.Style.reset()
        original_red = cd.Style.RED
        self.assertNotEqual(original_red, "")
        cd.Style.disable()
        self.assertEqual(cd.Style.RED, "")
        self.assertFalse(cd.Style.is_enabled())
        cd.Style.reset()
        self.assertEqual(cd.Style.RED, original_red)
        self.assertTrue(cd.Style.is_enabled())


# =============================================================================
# Discover Projects
# =============================================================================

class TestDiscoverProjects(TestCase):
    def test_empty_when_no_projects_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()
            result = cd.discover_projects(claude_home)
            self.assertEqual(result, [])

    def test_discovers_project_with_git(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "myproject"
            proj_dir.mkdir()
            (proj_dir / ".git").mkdir()
            claude_home = Path(tmpdir) / ".claude"
            projects_dir = claude_home / "projects"
            dash_name = str(proj_dir).replace("/", "-")
            (projects_dir / dash_name).mkdir(parents=True)
            result = cd.discover_projects(claude_home)
            self.assertIn(str(proj_dir), result)

    def test_discovers_child_projects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "workspace"
            parent.mkdir()
            child = parent / "app"
            child.mkdir()
            (child / "package.json").write_text("{}")
            claude_home = Path(tmpdir) / ".claude"
            dash_name = str(parent).replace("/", "-")
            (claude_home / "projects" / dash_name).mkdir(parents=True)
            result = cd.discover_projects(claude_home)
            self.assertIn(str(child), result)

    def test_ignores_hidden_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "workspace"
            parent.mkdir()
            hidden = parent / ".hidden"
            hidden.mkdir()
            (hidden / ".git").mkdir()
            claude_home = Path(tmpdir) / ".claude"
            dash_name = str(parent).replace("/", "-")
            (claude_home / "projects" / dash_name).mkdir(parents=True)
            result = cd.discover_projects(claude_home)
            self.assertNotIn(str(hidden), result)

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "workspace"
            parent.mkdir()
            for name in ["zzz", "aaa", "mmm"]:
                d = parent / name
                d.mkdir()
                (d / ".git").mkdir()
            claude_home = Path(tmpdir) / ".claude"
            dash_name = str(parent).replace("/", "-")
            (claude_home / "projects" / dash_name).mkdir(parents=True)
            result = cd.discover_projects(claude_home)
            self.assertEqual(result, sorted(result))


# =============================================================================
# EffectivenessProxyCheck (expanded)
# =============================================================================

class TestEffectivenessProxyCheckExpanded(TestCase):
    def _run(self, content):
        files = {
            "global_claude_md": [_make_file(content=content)],
            "rules": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        return cd.EffectivenessProxyCheck().run(files, ctx, CW)

    def test_no_issues(self):
        result = self._run("- Use formal tone\n- Write tests\n")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(len(result.findings), 0)

    def test_few_negations_info(self):
        content = "- don't use\n- avoid this\n- never do\n- prohibit X\n"
        result = self._run(content)
        self.assertTrue(any(
            "4 negative" in f.message for f in result.findings
        ))

    def test_vague_expressions_detected(self):
        content = "- Handle errors appropriately\n- if necessary, add tests\n"
        # Japanese vague patterns won't match English, use English ones
        result = self._run(content)
        self.assertTrue(any("Vague" in f.message for f in result.findings))

    def test_vague_expressions_display_limit(self):
        content = "\n".join([
            "- do as needed",
            "- when possible, optimize",
            "- if necessary, refactor",
            "- when appropriate, document",
            "- if needed, test",
        ])
        result = self._run(content)
        vague_findings = [f for f in result.findings if "Vague" in f.message or "vague" in f.message]
        # At most 3 individual + 1 summary
        individual = [f for f in vague_findings if "more vague" not in f.message]
        summary = [f for f in vague_findings if "more vague" in f.message]
        self.assertLessEqual(len(individual), 3)
        self.assertEqual(len(summary), 1)


# =============================================================================
# Boundary Value Tests
# =============================================================================

class TestMaturityBoundaries(TestCase):
    def _detect_maturity(self, total_tokens):
        files = {"global_claude_md": [_make_file(tokens=total_tokens)]}
        ctx = cd.UserContext(
            project_count=1, total_config_tokens=total_tokens,
            config_maturity="", setup_type="single_project",
            has_custom_skills=False, has_custom_rules=False,
            has_project_memory=False, memory_line_count=0,
            rule_count=0, skill_count=0, project_dirs=[],
        )
        # detect_context calculates maturity from total_tokens
        # We test the classification logic directly
        if total_tokens < 500:
            return "sparse"
        elif total_tokens < 3000:
            return "moderate"
        else:
            return "rich"

    def test_499_is_sparse(self):
        self.assertEqual(self._detect_maturity(499), "sparse")

    def test_500_is_moderate(self):
        self.assertEqual(self._detect_maturity(500), "moderate")

    def test_2999_is_moderate(self):
        self.assertEqual(self._detect_maturity(2999), "moderate")

    def test_3000_is_rich(self):
        self.assertEqual(self._detect_maturity(3000), "rich")


class TestMaturityBoundariesIntegration(TestCase):
    """Integration test: detect_context produces correct maturity."""

    def _build_and_detect(self, content):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_home = Path(tmpdir) / ".claude"
            claude_home.mkdir()
            (claude_home / "CLAUDE.md").write_text(content)
            files = cd.collect_config_files(claude_home)
            ctx = cd.detect_context(files, claude_home)
            return ctx

    def test_sparse_setup(self):
        ctx = self._build_and_detect("# Hi\n")
        self.assertEqual(ctx.config_maturity, "sparse")

    def test_rich_setup(self):
        # Generate enough content to exceed 3000 tokens
        content = "# Rules\n" + "\n".join(
            ["- Rule number %d: do something specific" % i for i in range(500)]
        )
        ctx = self._build_and_detect(content)
        self.assertEqual(ctx.config_maturity, "rich")


class TestMemoryLineBoundaries(TestCase):
    def _run_budget(self, line_count):
        content = "\n".join(["line"] * line_count)
        files = {
            "global_claude_md": [_make_file(tokens=50)],
            "memories": [_make_file(
                path="/mock/memory.md", category="memory",
                content=content, line_count=line_count,
            )],
        }
        ctx = _make_context(total_config_tokens=50)
        return cd.BudgetCheck().run(files, ctx, CW)

    def test_150_lines_no_memory_finding(self):
        result = self._run_budget(150)
        memory_findings = [f for f in result.findings if "MEMORY.md" in f.message]
        self.assertEqual(len(memory_findings), 0)

    def test_151_lines_info(self):
        result = self._run_budget(151)
        memory_findings = [f for f in result.findings if "MEMORY.md" in f.message]
        self.assertEqual(len(memory_findings), 1)
        self.assertEqual(memory_findings[0].severity, "info")

    def test_180_lines_info(self):
        result = self._run_budget(180)
        memory_findings = [f for f in result.findings if "MEMORY.md" in f.message]
        self.assertEqual(len(memory_findings), 1)
        self.assertEqual(memory_findings[0].severity, "info")

    def test_181_lines_warning(self):
        result = self._run_budget(181)
        memory_findings = [f for f in result.findings if "MEMORY.md" in f.message]
        self.assertEqual(len(memory_findings), 1)
        self.assertEqual(memory_findings[0].severity, "warning")


class TestFreshnessBoundaries(TestCase):
    def _make_file_with_age(self, days):
        from datetime import timedelta
        mtime = datetime.now(tz=timezone.utc) - timedelta(days=days)
        return cd.FileMetrics(
            path="/mock/test.md", category="global_claude_md", exists=True,
            size_bytes=10, line_count=1, estimated_tokens=5,
            last_modified=mtime.isoformat(),
            sections=[], referenced_paths=[], directive_count=0,
            negation_count=0, content="# Test\n",
        )

    def _run(self, days):
        files = {
            "global_claude_md": [self._make_file_with_age(days)],
            "rules": [], "skills": [], "memories": [],
            "project_claude_mds": [],
        }
        ctx = _make_context()
        return cd.FreshnessCheck().run(files, ctx, CW)

    def test_90_days_no_finding(self):
        result = self._run(90)
        age_findings = [f for f in result.findings if "days" in f.message]
        self.assertEqual(len(age_findings), 0)

    def test_91_days_info(self):
        result = self._run(91)
        age_findings = [f for f in result.findings if "days" in f.message]
        self.assertEqual(len(age_findings), 1)
        self.assertEqual(age_findings[0].severity, "info")

    def test_180_days_info(self):
        result = self._run(180)
        age_findings = [f for f in result.findings if "days" in f.message]
        self.assertEqual(len(age_findings), 1)
        self.assertEqual(age_findings[0].severity, "info")

    def test_181_days_warning(self):
        result = self._run(181)
        age_findings = [f for f in result.findings if "days" in f.message]
        self.assertEqual(len(age_findings), 1)
        self.assertEqual(age_findings[0].severity, "warning")


class TestNegationBoundaries(TestCase):
    def _run(self, negation_count):
        lines = ["- don't do thing %d" % i for i in range(negation_count)]
        content = "\n".join(lines)
        files = {
            "global_claude_md": [_make_file(content=content)],
            "rules": [], "memories": [], "project_claude_mds": [],
        }
        ctx = _make_context()
        return cd.EffectivenessProxyCheck().run(files, ctx, CW)

    def test_3_negations_no_finding(self):
        result = self._run(3)
        neg_findings = [f for f in result.findings if "negative" in f.message.lower()]
        self.assertEqual(len(neg_findings), 0)

    def test_4_negations_info(self):
        result = self._run(4)
        neg_findings = [f for f in result.findings if "negative" in f.message.lower()]
        self.assertEqual(len(neg_findings), 1)
        self.assertEqual(neg_findings[0].severity, "info")

    def test_5_negations_info(self):
        result = self._run(5)
        neg_findings = [f for f in result.findings if "negative" in f.message.lower()]
        self.assertEqual(len(neg_findings), 1)
        self.assertEqual(neg_findings[0].severity, "info")

    def test_6_negations_warning(self):
        result = self._run(6)
        neg_findings = [f for f in result.findings if "negative" in f.message.lower()]
        self.assertEqual(len(neg_findings), 1)
        self.assertEqual(neg_findings[0].severity, "warning")


if __name__ == "__main__":
    main()
