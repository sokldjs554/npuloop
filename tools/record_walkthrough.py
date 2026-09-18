"""docs/index.html 을 헤드리스로 몰아 워크벤치 둘러보기 GIF 를 만든다.

    python tools/record_walkthrough.py [--out docs/model_study_walkthrough.gif] [--frames <dir>]

저장된 실험 기록만 읽는 정적 페이지이므로 서버도 네트워크도 필요 없다. 프레임마다
`file://docs/index.html` 을 실제로 조작한 뒤 뷰포트를 그대로 캡처한다. 합성 화면이 아니다.

GIF 는 연속 녹화가 아니라 **정지 컷 + 긴 체류**다. 같은 용량이면 그쪽이 숫자를 읽히게 한다
(연속 스크롤은 프레임레이트에서 전부 뭉갠다). 컷 12장 · 약 50초 · 1 MB 남짓.

ffmpeg 은 쓰지 않는다 — 이 환경에 있는 것은 Playwright 전용 webm 빌드라 gif muxer 가 없다.
Pillow 로 전역 팔레트를 만들어 직접 합성한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/index.html'
CHROMIUM = '/opt/pw-browsers/chromium'
VIEW = {'width': 1440, 'height': 900}
GIF_WIDTH = 1040          # README 본문 폭(약 830px)보다 조금 크게 — 축소되며 글자가 선명해진다
COLORS = 128              # UI 는 평면 색이라 128색이면 밴딩이 보이지 않는다
TOPBAR = 80               # 스크롤 위치를 잡을 때 상단 고정 바만큼 띄운다


def settle(page, view, tab=None):
    """route() 는 hashchange 를 타므로 비동기다. 컨트롤러가 반영될 때까지 기다린 뒤 한 번 더 그린다."""
    page.wait_for_function(
        '([v, t]) => { const s = window.NpuWorkbenchApp.getState();'
        '  return s.view === v && (t === null || s.tab === t)'
        '      && !document.querySelector(`[data-view=${v}]`).hidden; }',
        arg=[view, tab])
    page.evaluate('document.activeElement && document.activeElement.blur()')
    page.wait_for_timeout(120)


def scroll_to(page, selector: str):
    page.evaluate(
        '([sel, pad]) => { const e = document.querySelector(sel);'
        '  window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - pad); }',
        arg=[selector, TOPBAR])
    page.wait_for_timeout(100)


def top(page):
    page.evaluate('window.scrollTo(0, 0)')
    page.wait_for_timeout(100)


# (설명, 동작, 체류 ms). 설명은 프레임 파일 이름과 실행 로그에만 쓴다 — 화면에는 자막을 넣지 않는다.
def scenes():
    def s01(page):
        settle(page, 'studies')
        page.wait_for_selector('#study-training svg')
        top(page)

    def s02(page):
        # 후보 여섯 개와 '조건 적용' 패널을 먼저 보여 준 뒤에 행을 눌러야 인과가 읽힌다.
        scroll_to(page, '#study-variants')

    def s03(page):
        # 기록이 없는 변경안을 고르면 곡선을 지어내지 않고 '미기록'을 띄운다. 그 동작을 찍는다.
        page.click('#study-variants [data-study-variant="swap-relu-heal0"]')
        page.wait_for_timeout(200)
        scroll_to(page, '#study-metrics')

    def s04(page):
        page.click('#study-variants [data-study-variant="swap-relu-heal3"]')
        page.wait_for_timeout(200)
        page.wait_for_selector('#study-training svg')
        scroll_to(page, '#study-metrics')

    def s05(page):
        page.click('[data-nav=analysis]')
        settle(page, 'analysis', 'summary')
        page.wait_for_selector('#analysis-summary .bar-row')
        top(page)

    def s06(page):
        page.click('#tab-comparison')
        settle(page, 'analysis', 'comparison')
        page.wait_for_selector('#analysis-comparison .comparison-grid')
        top(page)

    def s07(page):
        page.click('[data-nav=research]')
        settle(page, 'research')
        page.wait_for_selector('#fidelity-chart svg')
        top(page)

    def s08(page):
        # '연산자별 출력 차이' 제목은 #research-fidelity 의 자식이 아니라 형제다.
        scroll_to(page, '#view-research > .section-heading')

    def s09(page):
        page.eval_on_selector('#view-research details.detail-section', 'd => { d.open = true }')
        page.fill('#research-search', 'norm')
        page.wait_for_selector('#fidelity-table table tbody tr')
        page.evaluate('document.activeElement && document.activeElement.blur()')
        scroll_to(page, '#view-research details.detail-section')

    def s10(page):
        page.fill('#research-search', '')
        page.eval_on_selector('#view-research details.detail-section', 'd => { d.open = false }')
        page.evaluate('document.activeElement && document.activeElement.blur()')
        scroll_to(page, '#research-layernorm')

    def s11(page):
        scroll_to(page, '#research-imagenet')

    def s12(page):
        top(page)
        page.click('#research-source')
        page.wait_for_selector('#record-dialog[open] #record-json')
        page.wait_for_timeout(150)

    return [
        ('01-studies-e4', s01, 3800),
        ('02-studies-metrics', s02, 3200),
        ('03-studies-before-heal', s03, 3400),
        ('04-studies-after-heal', s04, 3800),
        ('05-analysis-summary', s05, 5000),
        ('06-analysis-comparison', s06, 4600),
        ('07-research-headline', s07, 6000),
        ('08-research-chart', s08, 4400),
        ('09-research-layernorm-node', s09, 4400),
        ('10-research-e16', s10, 4800),
        ('11-research-e20-imagenet', s11, 5000),
        ('12-research-source-json', s12, 4200),
    ]


def record(frames_dir: Path) -> list[tuple[Path, int]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    shots: list[tuple[Path, int]] = []
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True,
                                    args=['--no-sandbox', '--force-color-profile=srgb'])
        ctx = browser.new_context(viewport=VIEW, device_scale_factor=1,
                                  reduced_motion='reduce', color_scheme='light')
        page = ctx.new_page()
        page.on('pageerror', lambda e: problems.append(f'pageerror: {e}'))
        page.on('console', lambda m: problems.append(f'console: {m.text}') if m.type == 'error' else None)

        page.goto(DOC.resolve().as_uri(), wait_until='load')   # page.route 로 막으면 file:// 문서 자체가 죽는다
        page.wait_for_function('document.documentElement.dataset.appReady === "true"')

        for name, action, dwell in scenes():
            action(page)
            page.evaluate('document.activeElement && document.activeElement.blur()')
            if page.locator('#error-banner').is_visible():
                problems.append(f'{name}: error banner is visible')
            path = frames_dir / f'{name}.png'
            page.screenshot(path=str(path))          # full_page 는 화면마다 높이가 달라져 GIF 가 안 된다
            shots.append((path, dwell))
            print(f'  {name:28s} {dwell/1000:4.1f}s')
        browser.close()
    if problems:
        raise SystemExit('페이지가 오류를 냈다:\n  ' + '\n  '.join(problems))
    return shots


def assemble(shots: list[tuple[Path, int]], out: Path) -> None:
    height = round(VIEW['height'] * GIF_WIDTH / VIEW['width'])
    images = [Image.open(p).convert('RGB').resize((GIF_WIDTH, height), Image.LANCZOS) for p, _ in shots]
    # 프레임마다 팔레트를 따로 뽑으면 정지 컷 사이에서 색이 떨린다. 전 프레임에서 하나를 뽑아 공유한다.
    strip = Image.new('RGB', (GIF_WIDTH, height * len(images)))
    for i, im in enumerate(images):
        strip.paste(im, (0, i * height))
    palette = strip.quantize(colors=COLORS, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.NONE) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out, save_all=True, append_images=quantized[1:],
                      duration=[d for _, d in shots], loop=0, optimize=True, disposal=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/model_study_walkthrough.gif')
    ap.add_argument('--frames', type=Path, default=Path('/tmp/npuloop-walkthrough-frames'))
    args = ap.parse_args(argv)

    if not DOC.is_file():
        raise SystemExit(f'{DOC} 가 없다. 먼저 python demo/build.py 를 실행한다.')
    print(f'녹화: {DOC}')
    shots = record(args.frames)
    assemble(shots, args.out)
    seconds = sum(d for _, d in shots) / 1000
    size = args.out.stat().st_size
    print(f'{args.out}  {size:,} bytes  ·  컷 {len(shots)}장  ·  {seconds:.1f}초')
    return 0


if __name__ == '__main__':
    sys.exit(main())
