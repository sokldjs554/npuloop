"""Exercise the complete static workbench in Chromium without a network or model server.

The HTML bytes are loaded with set_content; this is not a Windows/file-URL or
hosted-site deployment test. Downloads and source-record content are verified.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',default='/usr/bin/chromium')
    parser.add_argument('--output',type=Path,default=ROOT/'verification/workbench/browser')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[];requests=[]
    def check(condition,name):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.browser,headless=True,args=['--no-sandbox'])
        ctx=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
        def blocked(route):requests.append(route.request.url);route.abort()
        page.route('**/*',blocked)
        page.set_content((ROOT/'docs/index.html').read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function('document.documentElement.dataset.appReady === "true"')
        def state():return page.evaluate('NpuWorkbenchApp.getState()')
        def result():return page.evaluate('NpuWorkbenchApp.getAnalysis()')
        def nav(hash):
            page.evaluate('(h)=>location.hash=h',hash)
            page.wait_for_timeout(25)
        def bodyclean(name):
            check(not page.locator('#error-banner').is_visible(),name+' error banner')
            text=page.locator('body').inner_text()
            check(all(word not in text for word in ['할까요','무엇을 고쳐야','따라가','눌러보','살펴보','처방']),name+' UI copy')
            check('NaN' not in text and 'Infinity' not in text,name+' finite display')
            check(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),name+' page width')
        check(page.locator('h1').inner_text()=='NPU 모델 분석·검증','application title')
        check(page.locator('.nav-item').count()==4,'four workspace routes')
        check(result()['estimate']['total']==2503878,'default exact cost')
        bodyclean('initial')
        page.screenshot(path=str(out/'analysis.png'),full_page=True)
        # Actual native select change events: every registered model and preset.
        models=page.locator('#model-select option').evaluate_all('(xs)=>xs.map(x=>x.value)')
        presets=page.locator('#preset-select option').evaluate_all('(xs)=>xs.map(x=>x.value)')
        for model in models:
            page.select_option('#model-select',model)
            for preset in presets:
                page.select_option('#preset-select',preset);r=result()
                check(r['model']==model and r['preset']==preset,model+'|'+preset+' state')
                check(r['estimate']['total']>0,model+'|'+preset+' positive cost')
                shown=page.locator('#analysis-summary .metric-value').first.inner_text()
                check(shown==f"{r['estimate']['total']:,.0f}",model+'|'+preset+' displayed cost')
                check(not r['custom'],model+'|'+preset+' original preset')
                bodyclean(model+'|'+preset)
        page.select_option('#model-select','cust_vit');page.select_option('#preset-select','edge-10tops-strict')
        page.click('#tab-operators');page.wait_for_timeout(25)
        for filt in ['all','host','array','vector']:
            page.select_option('#operator-filter',filt)
            for order in ['cycles','graph','name']:
                page.select_option('#operator-sort',order)
                check(page.locator('#operator-table tbody tr').count()>0,'operator filter '+filt+'/'+order)
        page.select_option('#operator-filter','host')
        check(page.locator('#operator-table tbody tr').count()==25,'strict host nodes')
        page.select_option('#operator-filter','all');page.select_option('#operator-sort','cycles')
        page.fill('#operator-search','layernorm')
        check(page.locator('#operator-table tbody tr').count()==13,'operator search by kind')
        node=page.locator('#operator-table tr[data-node]').first.get_attribute('data-node')
        page.locator('#operator-table tr[data-node]').first.locator('td').nth(1).click()
        check(page.locator('#operator-inspector h3').inner_text()==node,'whole row selection inspector')
        page.fill('#operator-search','no-such-node-123')
        check('일치하는 노드 없음' in page.locator('#operator-table').inner_text(),'empty node search')
        page.fill('#operator-search','')
        if page.locator('#operator-inspector [data-select-node]').count():
            target=page.locator('#operator-inspector [data-select-node]').first.get_attribute('data-select-node')
            page.locator('#operator-inspector [data-select-node]').first.click()
            check(state()['node']==target,'node connection navigation')
        with page.expect_download() as dl:page.click('#export-operators')
        dl.value.save_as(str(out/'operators.csv'))
        rows=list(csv.reader(io.StringIO((out/'operators.csv').read_text(encoding='utf-8-sig'))))
        check(rows[0][0]=='node' and len(rows)==len(result()['estimate']['rows'])+1,'CSV export rows and headers')
        page.screenshot(path=str(out/'operators.png'),full_page=True)
        # Recorded comparisons must not change or borrow another experiment's accuracy.
        page.click('#tab-comparison');page.wait_for_timeout(25)
        for preset in ['edge-10tops-strict','edge-10tops']:
            page.select_option('#preset-select',preset)
            check('2,000장' in page.locator('#analysis-comparison').inner_text(),'E9 sample size '+preset)
            check('80.80%' in page.locator('#analysis-comparison').inner_text(),'E9 baseline '+preset)
        page.select_option('#preset-select','edge-10tops-strict')
        page.screenshot(path=str(out/'comparison.png'),full_page=True)
        page.click('#analysis-comparison [data-source]')
        saved=json.loads(page.locator('#record-json').inner_text())
        check(saved['model']=='cust_vit' and saved['after']['cycles_strict']==1708230,'E9 raw record contents')
        page.keyboard.press('Escape');check(not page.locator('#record-dialog').is_visible(),'dialog escape')
        page.select_option('#model-select','cust_inception')
        check('변경 후 실험 기록 없음' in page.locator('#analysis-comparison').inner_text(),'absent after result')
        page.click('[data-action="default-comparison"]')
        check(state()['model']=='cust_vit','default comparison action')
        # All settings and boundary form validation.
        page.click('#tab-settings');page.wait_for_timeout(25)
        initial=result()['estimate']['total'];page.click('#cost-settings button[type=submit]')
        check(not result()['custom'] and result()['estimate']['total']==initial,'unchanged settings are not custom')
        for arr in ['8','16','32','64','128','256']:
            page.select_option('#setting-array',arr);page.click('#cost-settings button[type=submit]')
            check(result()['spec']['pe_rows']==int(arr),'PE array '+arr)
        for cores in ['1','2','4','8']:
            page.fill('#setting-cores',cores);page.click('#cost-settings button[type=submit]')
            check(result()['spec']['cores']==int(cores),'cores '+cores)
        for bw in ['0.5','12.8','80']:
            page.fill('#setting-dram',bw);page.click('#cost-settings button[type=submit]')
            check(result()['spec']['dram_gbps']==float(bw),'DRAM '+bw)
        for dw in ['no','yes']:
            page.select_option('#setting-depthwise',dw);page.click('#cost-settings button[type=submit]')
            check((result()['spec']['dw_lanes']>0)==(dw=='yes'),'depthwise '+dw)
        page.select_option('#setting-activation','lut');page.click('#cost-settings button[type=submit]')
        check(result()['hostCount']==19,'LUT retains normalization restrictions')
        check(result()['recordedComparison'] is None,'no recorded comparison for custom configuration')
        page.click('#tab-comparison');page.wait_for_timeout(25)
        check('변경 후 실험 기록 없음' in page.locator('#analysis-comparison').inner_text(),'custom comparison explicit empty')
        page.click('#tab-settings');page.wait_for_timeout(25)
        before=result()['estimate']['total'];page.fill('#setting-cores','0');page.click('#cost-settings button[type=submit]')
        check(not page.locator('#setting-cores').evaluate('(x)=>x.checkValidity()'),'invalid core form boundary')
        check(result()['estimate']['total']==before,'invalid form preserves analysis')
        page.click('#reset-settings');check(not result()['custom'],'preset reset')
        page.screenshot(path=str(out/'settings.png'),full_page=True)
        with page.expect_download() as dl:page.click('#export-report')
        dl.value.save_as(str(out/'analysis.json'));export=json.loads((out/'analysis.json').read_text())
        check(export['provenance']=='simulated' and export['hardware_measured'] is False,'analysis export provenance')
        # Research uses the full split; E16 and E9 remain separate.
        page.click('[data-nav=research]');page.wait_for_timeout(25)
        rm=page.locator('#research-model option').evaluate_all('(xs)=>xs.map(x=>x.value)')
        for m in rm:
            page.select_option('#research-model',m)
            for sc in ['npu-default','per-tensor']:
                page.select_option('#research-scheme',sc)
                r=page.evaluate('NpuWorkbenchApp.getResearch()')
                check(r['summary']['model']==m and r['summary']['scheme']==sc,'research scope '+m+'/'+sc)
                check(r['summary']['images']==(3925 if m.startswith('imagenette') else 10000),'research sample '+m+'/'+sc)
                check(page.locator('#fidelity-chart svg').count()==1,'research chart '+m+'/'+sc)
                bodyclean('research '+m+'/'+sc)
        page.select_option('#research-model','cust_vit');page.select_option('#research-scheme','npu-default')
        check('250장' in page.locator('#research-layernorm').inner_text() and '10,000장' in page.locator('#research-layernorm').inner_text(),'E16 scopes')
        check('80.97%' in page.locator('#research-fidelity').inner_text(),'E15 accuracy distinct from E9')
        page.screenshot(path=str(out/'research.png'),full_page=True)
        page.click('.detail-section summary');page.fill('#research-search','layernorm')
        check(page.locator('#fidelity-table tbody tr').count()==13,'research layer search')
        page.click('#research-source');raw=json.loads(page.locator('#record-json').inner_text())
        check(raw['n_test']==10000 and raw['int_acc']==.8097,'research source exact')
        page.click('#close-dialog');page.click('.detail-section summary')
        # Catalog: every experiment's original JSON and every record remain accessible.
        page.click('[data-nav=experiments]');page.wait_for_timeout(25)
        exps=page.locator('[data-experiment]').evaluate_all('(xs)=>xs.map(x=>x.dataset.experiment)')
        check(len(exps)==17,'catalog contains 17 experiments')
        for key in exps:
            page.click('[data-experiment="'+key+'"]')
            check(page.locator('#experiment-detail tbody tr').count()>0,'experiment table '+key)
            page.locator('#experiment-detail [data-record-index]').first.click()
            record=json.loads(page.locator('#record-json').inner_text())
            reference=json.loads((ROOT/'results'/(key+'.json')).read_text())['records'][0]
            reference.pop('sensitivity_full',None)
            check(record==reference,'raw record fidelity '+key)
            page.click('#close-dialog')
            bodyclean('experiment '+key)
        page.click('[data-experiment=e3_lint_vs_drop]')
        page.click('[data-record-page=next]');check(state()['recordPage']==1,'record pagination next')
        page.click('[data-record-page=prev]');check(state()['recordPage']==0,'record pagination previous')
        page.select_option('#record-model','cust_vit')
        check('cust_vit' in page.locator('#record-model').input_value(),'record model filter')
        page.fill('#experiment-search','not-found-987');check(page.locator('[data-experiment]').count()==0,'catalog empty search')
        page.fill('#experiment-search','LayerNorm');check(page.locator('[data-experiment]').count()>0,'catalog search')
        page.fill('#experiment-search','');page.select_option('#experiment-category','정수 출력')
        check(page.locator('[data-experiment]').count()==4,'catalog category filter')
        page.select_option('#experiment-category','all');page.click('[data-experiment=e6_pruning]')
        page.screenshot(path=str(out/'experiments.png'),full_page=True)
        with page.expect_download() as dl:page.click('#experiment-detail [data-download-source]')
        dl.value.save_as(str(out/'e6.json'));check(json.loads((out/'e6.json').read_text())==json.loads((ROOT/'results/e6_pruning.json').read_text()),'full experiment JSON download')
        page.click('[data-nav=resources]');page.wait_for_timeout(25);bodyclean('resources');page.screenshot(path=str(out/'resources.png'),full_page=True)
        for anchor in page.locator('a[href]').all():
            href=anchor.get_attribute('href')
            if href and not href.startswith(('#','http','data:','blob:')):check((ROOT/'docs'/href).is_file(),'relative link exists '+href)
        # Browser back/forward keeps view state; skip navigation keeps the active page.
        nav('#research'); nav('#resources')
        page.evaluate('history.back()');page.wait_for_function("NpuWorkbenchApp.getState().view==='research'")
        check(state()['view']=='research','browser history back')
        page.evaluate('history.forward()');page.wait_for_function("NpuWorkbenchApp.getState().view==='resources'")
        check(state()['view']=='resources','browser history forward')
        nav('#research');page.locator('.skip-link').focus();page.keyboard.press('Enter');page.wait_for_timeout(30)
        check(state()['view']=='research','skip navigation preserves page')
        check(page.locator('#main').evaluate('(x)=>document.activeElement===x'),'skip navigation focuses main')
        nav('#experiments/e4_surgery')
        labels=page.locator('#experiment-detail tbody tr td:first-child').all_text_contents()
        check(len(labels)==13 and all(x!='—' for x in labels),'all surgery conditions named')
        check(page.locator('#experiment-detail tbody tr').nth(3).locator('td').nth(2).inner_text()=='89.74%','changed FP32 not original baseline')
        # Direct links and keyboard tab selection.
        for alias,view,tab in [('cost','analysis','settings'),('intake','analysis','summary'),('prune','experiments',None),('tflite','experiments',None),('validation','resources',None)]:
            nav('#'+alias);check(state()['view']==view,'legacy route '+alias)
            if tab:check(state()['tab']==tab,'legacy tab '+alias)
        nav('#analysis');page.focus('#tab-summary');page.keyboard.press('ArrowRight');page.wait_for_timeout(25)
        check(state()['tab']=='operators','keyboard tab next');page.keyboard.press('End');page.wait_for_timeout(25)
        check(state()['tab']=='settings','keyboard tab end');page.keyboard.press('Home');page.wait_for_timeout(25)
        check(state()['tab']=='summary','keyboard tab home')
        # Responsive geometry, no question-style or tutorial text, both themes.
        for width in [1440,1024,768,390,320]:
            page.set_viewport_size({'width':width,'height':950})
            for route in ['analysis','analysis/operators','analysis/comparison','analysis/settings','research','experiments','resources']:
                nav('#'+route);bodyclean(f'{width} {route}')
            nav('#analysis')
            if width in [1440,390,320]:page.screenshot(path=str(out/f'analysis_{width}.png'),full_page=True)
        page.set_viewport_size({'width':1440,'height':1000});page.click('#theme-toggle');bodyclean('dark analysis');page.screenshot(path=str(out/'analysis_dark.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':950});page.click('#theme-mobile');check(page.locator('html').get_attribute('data-theme')=='light','mobile theme control')
        check(not errors,'no console or page errors');check(not requests,'no network requests')
        report={'passed':True,'checks':len(checks),'details':checks,'errors':errors,'network_requests':requests,'browser':browser.version,'mode':'offline HTML bytes via Playwright set_content','viewports':[1440,1024,768,390,320],'not_tested':['Windows native file double-click','hosted GitHub Pages','NPU hardware']}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        browser.close();print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
