"""End-to-end checks of the model-research workspace with real stored evidence."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def sr_studies(page):
    """Every super-resolution study the page offers.

    E17 carries one per depth, so hard-coding a single id here is what broke this check the first time a second
    depth landed. Read the ids off the page instead: a new depth is then covered automatically.
    """
    ids = page.eval_on_selector_all('#study-select option', 'os => os.map(o => o.value)')
    found = [i for i in ids if i.startswith('e17-')]
    if not found:
        raise SystemExit('no super-resolution study in #study-select: ' + ', '.join(ids))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'verification/model-study/browser')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []

    def check(ok, name):
        checks.append({'name': name, 'passed': bool(ok)})
        assert ok, name

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.set_content((ROOT / 'docs/index.html').read_text(), wait_until='load')
        page.wait_for_function('document.documentElement.dataset.appReady === "true"')
        check(page.locator('h1').inner_text() == '모델 실험·경량화', 'model research is the default workspace')
        check(page.locator('#view-studies').is_visible(), 'study workspace visible on root URL')
        check(page.locator('#study-select').input_value() == 'e4-activation', 'default surgery experiment')
        sr = sr_studies(page)
        check('89.74%' in page.locator('#study-metrics').inner_text(), 'measured recovered FP32 accuracy')
        check(page.locator('#study-training svg').count() == 1, 'actual recovery learning curve')
        check('40.36%' in page.locator('#study-context').inner_text(), 'immediate-change control shown')
        page.screenshot(path=str(args.output / 'model-study.png'), full_page=True)
        page.select_option('#study-select', 'e6-resnet20_relu')
        check('판정에 필요한 기록 부족' in page.locator('#study-decision').inner_text(), 'unrecorded strict costs do not pass')
        page.select_option('#study-preset', 'edge-10tops')
        check('모의 양자화' in page.locator('#study-variants').inner_text(), 'pruning metric is fake quantization')
        check('87.22%' in page.locator('#study-variants').inner_text(), 'real pruning accuracy listed')
        page.fill('#study-max-drop', '0')
        page.fill('#study-min-saving', '90')
        page.click('#study-apply-budget')
        check('조건을 만족하는 변경 없음' in page.locator('#study-decision').inner_text(), 'no fabricated recommendation')
        page.screenshot(path=str(args.output / 'pruning.png'), full_page=True)
        page.select_option('#study-select', 'e9-vit')
        page.select_option('#study-preset', 'tiny-1tops')
        check('미기록' in page.locator('#study-variants').inner_text(), 'missing after costs remain missing')
        page.select_option('#study-preset', 'edge-10tops-strict')
        check('2,000' in page.locator('#study-context').inner_text(), 'integer subset scope explicit')
        page.click('#study-source')
        raw = json.loads(page.locator('#record-json').inner_text())
        check(bool(raw), 'real source JSON available')
        page.keyboard.press('Escape')
        with page.expect_download() as download:
            page.click('#study-export')
        target = args.output / 'study-export.json'
        download.value.save_as(str(target))
        exported = json.loads(target.read_text())
        check(exported['hardware_measured'] is False, 'export preserves simulated hardware provenance')
        page.select_option('#study-select', sr[0])
        check('PSNR' in page.locator('#view-studies').inner_text(), 'task-specific super-resolution metric')
        page.select_option('#study-task', 'classification')
        check(page.locator('#study-select').input_value() not in sr, 'task filter changes available studies')
        # Outside the --report branch: inner_text() returns '' for a hidden element, so the invalid-report check
        # below silently passed on anything whenever the panel stayed collapsed.
        page.locator('.study-run summary').click()
        if args.report:
            page.set_input_files('#study-import', str(args.report))
            page.wait_for_function("document.querySelector('#study-import-result').textContent.includes('실행 결과')")
            check('FP32' in page.locator('#study-import-result').inner_text(), 'real local run import')
        page.set_input_files('#study-import', {'name': 'invalid.json', 'mimeType': 'application/json', 'buffer': b'{"schema_version":"wrong"}'})
        page.wait_for_function("document.querySelector('#study-import-status').textContent.length > 0")
        check('지원하지 않는' in page.locator('#study-import-status').inner_text(), 'invalid report rejected')
        for width in [1440, 1024, 768, 390, 320]:
            page.set_viewport_size({'width': width, 'height': 1000})
            page.select_option('#study-task', 'all')
            for study in ['e4-activation', 'e6-resnet20_relu', 'e9-vit', 'e12-imagenette', *sr]:
                page.select_option('#study-select', study)
                check(page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), f'no viewport overflow {width}/{study}')
                body = page.locator('body').inner_text()
                check('NaN' not in body and 'Infinity' not in body, f'finite display {width}/{study}')
            page.select_option('#study-select', 'e4-activation')
            if width in [1440, 390]:
                page.screenshot(path=str(args.output / f'study-{width}.png'), full_page=True)
        page.set_viewport_size({'width': 1440, 'height': 1100})
        page.click('#theme-toggle')
        page.screenshot(path=str(args.output / 'study-dark.png'), full_page=True)
        check(not errors, 'no browser errors')
        (args.output / 'report.json').write_text(json.dumps({'passed': True, 'checks': checks, 'errors': errors}, ensure_ascii=False, indent=2))
        print(json.dumps({'passed': True, 'checks': len(checks), 'errors': errors}))
        browser.close()


if __name__ == '__main__':
    main()
