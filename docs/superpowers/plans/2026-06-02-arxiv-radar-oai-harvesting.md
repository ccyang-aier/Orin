# ArXiv Radar OAI Harvesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile API/search-first arXiv radar collection path with a timezone-aware OAI-PMH collector and execute the real daily report for `2026-06-01`.

**Architecture:** Keep the existing CLI surface, but switch `collect` to single-day OAI polling across the configured arXiv sets. Convert `v1` GMT timestamps to `Asia/Shanghai`, filter by the target date, enrich each kept paper from the abstract page, and render a fully populated Markdown report with heuristic Chinese analysis instead of placeholders.

**Tech Stack:** Python 3.12, `urllib`, `xml.etree.ElementTree`, pytest, Markdown output.

---

### Task 1: Add failing coverage for the new collection semantics

**Files:**
- Modify: `code/arxiv_paper_radar/tests/test_orin_arxiv_radar.py`
- Test: `code/arxiv_paper_radar/tests/test_orin_arxiv_radar.py`

- [x] **Step 1: Write failing tests for OAI timestamp parsing, query-day windowing, and report auto-fill**
- [x] **Step 2: Run test to verify it fails**

### Task 2: Implement OAI-based collection and analysis helpers

**Files:**
- Modify: `code/arxiv_paper_radar/orin_arxiv_radar.py`
- Test: `code/arxiv_paper_radar/tests/test_orin_arxiv_radar.py`

- [ ] **Step 1: Add OAI helpers for single-day polling, record parsing, and timezone conversion**
- [ ] **Step 2: Replace `collect` source discovery with OAI-PMH and controlled fulltext extraction**
- [ ] **Step 3: Add automatic Chinese analysis generation for render output**

### Task 3: Verify, execute, and publish the real automation run

**Files:**
- Modify: `notes/papers/arxiv-ai-daily/2026-06-01 arxiv ai papers.md`
- Modify: `rag/arxiv_papers/processed_papers.json`
- Output: `code/arxiv_paper_radar/.tmp/2026-06-01-papers.json`

- [ ] **Step 1: Run pytest for the radar test file**
- [ ] **Step 2: Run the real `collect`, `render-report`, and `mark-analyzed` commands for target `2026-06-01`**
- [ ] **Step 3: Run final verification, then `git status`, `git add -A`, commit, and push**
