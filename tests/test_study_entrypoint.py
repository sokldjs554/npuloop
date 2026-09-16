from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def test_model_study_is_the_initial_workspace():
    source = (ROOT / 'demo/index.template.html').read_text()
    assert 'data-view="studies"' in source, 'Missing model experiment workspace'
    assert '<h1 id="page-title">모델 실험·경량화</h1>' in source
    assert 'id="study-import"' in source


def test_build_inlines_study_modules_and_research_references():
    source = (ROOT / 'demo/build.py').read_text()
    for name in ['study-data.js', 'study-ui.js', 'STUDY_RUNNER.md', 'RESEARCH_SUMMARY.md', 'REPRODUCTION_REQUIREMENTS.md']:
        assert name in source, 'Self-contained build must include ' + name


def test_published_study_assets_match_their_sources():
    html = (ROOT / 'docs/index.html').read_text()
    assert html == (ROOT / 'demo/index.html').read_text()
    for name in ['study-data.js', 'study-ui.js', 'study.css', 'workbench-ui.js']:
        assert (ROOT / 'demo' / name).read_text() in html, 'Stale generated asset: ' + name
