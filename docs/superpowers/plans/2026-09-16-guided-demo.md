# Guided npuloop Demo Implementation Plan

> **For agentic workers:** Use executing-plans to implement task-by-task.

**Goal:** Turn the existing long experimental dashboard into a four-step example followed by optional evidence.

**Architecture:** Keep one self-contained HTML with three hash-addressable views. Preserve existing cost-model functions and section IDs. Build injects a pure, tested data adapter, scoped CSS and a new controller into the current template.

**Tech Stack:** Existing Python builder, vanilla JavaScript/CSS, pytest + Node, Chromium + Playwright.

**Spec:** docs/superpowers/specs/2026-09-16-guided-demo.md

## Global Constraints
No new measurements, no remote writes, no dependency changes, no missing values rendered as zero. Preserve original result/PDF/core hashes. Self-contained HTML, network-blocked browser verification. Direct file:// navigation was blocked by sandbox policy; exercised identical bytes via set_content and checked target files separately. No hosted/file-navigation success claim.

### Task 1: Recorded case and research adapter
Files: demo/guided-data.js, tests/test_guided_demo.py.
Consumes: embedded results.e9_customer_intake/e15_fidelity/e16_ln_emulation; produces NpuDemoData.getCase(data,model,preset), getResearch(data,model,scheme), getLayerNorm(data,scheme).
- [x] Write tests for exact strict/standard cycles, E9 sample counts, unrecorded interventions, unknown conditions, nonmutation and E15/E16 scope.
- [x] Run `python -m pytest tests/test_guided_demo.py -q` and observe missing feature failure.
- [x] Implement validated pure functions; rerun the tests.

### Task 2: Guided interface and navigation
Files: demo/index.template.html, demo/guided.css, demo/guided-ui.js, demo/build.py.
- [x] Test the three views and four-step control are present in the source, with a hidden-by-default research/validation view.
- [x] Add the compact hero, case panel, persistent case context, and four mutually exclusive step panels.
- [x] Group all old sections into native details under validation; put the full research tables behind summaries.
- [x] Build in architecture-only mode with explicit provenance; inject all assets inline.

### Task 3: UI and regression verification
Files: tools/check_guided_browser.py, tools/check_demo_browser.py, verification/redesign/.
- [x] Test default ViT case, all step clicks/back/reset, preset changes, missing after records, all research selections, deep links and browser back.
- [x] Test mobile 390px, desktop 1440px, dark mode, no horizontal overflow or network dependency, visible focus and links.
- [x] Run existing cost/UI controls after revealing their containing view/details.
- [x] Run full pytest and quickstart; compare protected hashes.

### Task 4: Deliverable
Files: START_HERE.md, docs/DEMO_GUIDE.md, docs/DEMO_REDESIGN_REPORT.md, MANIFEST_SHA256.json.
- [x] Update navigation instructions and report actual evidence/limitations.
- [x] Package complete source with fresh HTML and verification logs, excluding caches, binaries, fonts and secrets.
- [x] Verify ZIP contents and copy final preview images for the user.

## Execution notes

Completed inline in an isolated extracted archive; no user checkout or remote repository was changed. Final packaging is checked by the archive manifest verifier. Evidence and limits: docs/DEMO_REDESIGN_REPORT.md.
