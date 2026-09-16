"""Capture the generated workbench; leave numerical research records untouched."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('--browser',required=True);args=p.parse_args()
root=Path.cwd()
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=args.browser,headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1,color_scheme='light')
    page.route('**/*',lambda route:route.abort())
    page.set_content((root/'docs/index.html').read_text(encoding='utf-8'),wait_until='load')
    page.wait_for_function('document.documentElement.dataset.appReady === "true"')
    page.screenshot(path=str(root/'docs/demo_overview.png'))
    (root/'verification/publish/preview.png').write_bytes((root/'docs/demo_overview.png').read_bytes())
    browser.close()
print('Captured docs/demo_overview.png from the generated HTML.')
