"""Narrow documentation/reproduction checks, not a replacement for experiment reruns.

Examples (run from repository root):
  python tools/submission_quality.py --paper
  python tools/submission_quality.py --regenerate --report paper-check.json
  python tools/submission_quality.py --inventory runs/cust_vit/best.pt --report artifacts.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Row counts the committed tables must have. Fixed by experiment design, except where DERIVED_ROWS says
# otherwise: a table that tracks a growing experiment cannot have its size written here, which is what failed
# this check the moment E17 gained a second depth.
TABLE_ROWS = {'th_operators.tex':9, 'th_baselines.tex':5, 'th_requant.tex':9, 'th_requant_seeds.tex':7,
              'th_tflite.tex':12, 'th_vela.tex':5, 'tab1_rows_ko.tex':12,
              'tab2_rows_ko.tex':4, 'tab3_rows_ko.tex':4}
DERIVED_ROWS = {'tab4_rows_ko.tex':'e17_dense_output',   # one table row per record of that experiment
                'th_adaround.tex':'e19_adaround',
                'th_imagenet.tex':'e20_imagenet_scale'}


def expected_rows(root: Path) -> dict[str,int]:
    rows=dict(TABLE_ROWS)
    for name,experiment in DERIVED_ROWS.items():
        path=root/'results'/(experiment+'.json')
        if path.is_file():
            rows[name]=len(json.loads(path.read_text(encoding='utf-8'))['records'])
    return rows
FIGURES = ('fig1_decomposition_ko.pdf','fig2_rounding_ko.pdf','fig3_operators_ko.pdf')
BAD_CLAIMS = (
    (r'어텐션 블록\s*12개', 'ViT block count is not the attention matmul count'),
    (r'무너지는 것은 Q7 이하', 'Q7/Q3 description contradicts the table'),
    (r'단조롭게 올라가', 'Observed propagation curves are not monotonic'),
    (r'a simulator cannot be repaired into a source of bit-accurate vectors one operator at a time', 'LN-only intervention is not a general impossibility result'),
    (r'\[저자 성명\]', 'Unfilled author placeholder'),
    (r'정확도 이외의 축에서 측정된 바가 없다', 'Unqualified novelty claim'),
)

MANUSCRIPTS = ('paper/npuloop_thesis.tex','paper/npuloop_esl.tex','paper/npuloop_ko.tex')
PLACEHOLDERS = ('[Author~Name]','[AUTHOR NAME]','[City]','[저자 성명]','[도시]')


def placeholder_errors(root: Path) -> list[str]:
    """No manuscript may ship an unfilled bracketed field.

    claim_errors() reads the long-form manuscript only, so the two short-form drafts were never
    covered. The short drafts intentionally carry no author line; what is forbidden is a leftover
    placeholder, not an absent name.
    """
    errors=[]
    for relative in MANUSCRIPTS:
        path=root/relative
        if not path.is_file():continue  # a missing thesis is already reported by paper_errors()
        text=path.read_text(encoding='utf-8')
        errors.extend(f'{relative}: unfilled placeholder {token}' for token in PLACEHOLDERS if token in text)
    return errors


def _safe_path(root: Path, relative: str) -> Path:
    """Keep inventories within the specified repository; never inspect .git or outside symlinks."""
    root = root.resolve()
    p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts or '.git' in p.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    target = root / p
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Path leaves repository: {relative}')
    if any(part.is_symlink() for part in [target, *target.parents] if part != root and part.is_relative_to(root)):
        raise ValueError(f'Symlink is not an artifact input: {relative}')
    return target

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def claim_errors(text: str) -> list[str]:
    """Catch only known wording regressions. This is not a semantic review of all claims."""
    normalized = re.sub(r'\s+',' ',text)
    return [message for pattern,message in BAD_CLAIMS if re.search(pattern,normalized)]

def snapshot_errors(root: Path, references: list[dict[str,Any]]) -> list[str]:
    errors=[]
    for item in references:
        p = _safe_path(root,item['path'])
        if not p.is_file():
            errors.append(f"Missing snapshot file: {item['path']}")
        elif file_sha256(p) != item['sha256']:
            errors.append(f"Snapshot bytes differ: {item['path']}")
    return errors

def artifact_inventory(root: Path, paths: list[str]) -> list[dict[str,Any]]:
    records=[]
    for rel in paths:
        p = _safe_path(root,rel)
        if p.exists() and not p.is_file():
            raise ValueError(f'Not a regular artifact file: {rel}')
        present = p.is_file()
        records.append({'path':rel,'status':'present' if present else 'missing',
                        'size_bytes':p.stat().st_size if present else None,
                        'sha256':file_sha256(p) if present else None,
                        'historical_identity_verified':False})
    return records

def fidelity_summary(records: list[dict[str,Any]]) -> str:
    """Generate a data-dependent E15 headline; do not hard-code that every CI contains zero."""
    if not records:
        raise ValueError('No E15 records; cannot report fidelity')
    pairs=[]
    for record in records:
        p=record['int_vs_fake']; d=float(p['delta']); se=float(p['se'])
        if not math.isfinite(d) or not math.isfinite(se) or se<0:
            raise ValueError('E15 delta and SE must be finite; SE must be nonnegative')
        pairs.append((d,se))
    zero=sum(d-1.96*se<=0<=d+1.96*se for d,se in pairs)
    maximum=max(abs(d) for d,se in pairs)*100
    return (f'95% Wald 구간의 0 포함 {zero}/{len(pairs)}; 최대 관측 정확도 차이 '
            f'{maximum:.2f}%p (정수 − 모의, 전체 테스트셋). 허용 오차 내 동등성 입증은 아님')

def column_errors(name: str, table: str) -> list[str]:
    """A generated table whose column spec disagrees with its header is a LaTeX error at build time.

    Cheap to check here and awkward to find in a 3,000-line lualatex log, so it is checked here.
    """
    spec=re.search(r'\\begin\{tabular\}\{([^}]*)\}',table)
    if not spec:return [f'{name}: no tabular environment']
    columns=sum(1 for c in spec.group(1) if c in 'lcr')
    rows=[l for l in table.splitlines() if '&' in l]
    if not rows:return [f'{name}: no rows']
    widths={l.count('&')+1 for l in rows}
    return [f'{name}: {columns} columns declared, rows have {sorted(widths)}'] if widths!={columns} else []


def paper_errors(root: Path) -> list[str]:
    errors=[]
    source=root/'paper/npuloop_thesis.tex'
    if not source.is_file():return ['Missing paper/npuloop_thesis.tex']
    text=source.read_text(encoding='utf-8'); errors.extend(claim_errors(text))
    errors.extend(placeholder_errors(root))
    for token in ('\\begin{document}','\\end{document}','\\appendix','\\begin{thebibliography}',
                  '\\label{tab:operators}','\\label{tab:fidelity}','\\label{tab:layernorm}','\\label{tab:dense}',
                  '\\label{tab:requantseeds}','\\label{tab:adaround}','\\label{tab:imagenet}'):
        if token not in text:errors.append(f'Missing structural token: {token}')
    if len(re.findall(r'\\chapter\{',text))<9:errors.append('Expected seven chapters and two appendices')
    for name,expected in expected_rows(root).items():
        path=root/'paper'/name
        if not path.is_file():errors.append(f'Missing table: {name}');continue
        table=path.read_text(encoding='utf-8')
        if '\\midrule' not in table or '\\bottomrule' not in table:
            errors.append(f'Malformed table: {name}');continue
        body=table.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
        count=sum('&' in line and line.rstrip().endswith('\\\\') for line in body.splitlines())
        if count!=expected:errors.append(f'{name}: {count} data rows, expected {expected}')
        errors.extend(column_errors(name,table))
    for name in FIGURES:
        path=root/'paper'/name
        if not path.is_file():errors.append(f'Missing figure: {name}')
        elif not path.read_bytes().startswith(b'%PDF-'):errors.append(f'Not a PDF figure: {name}')
    return errors

def regeneration_errors(root: Path) -> tuple[list[str],list[dict[str,Any]]]:
    """Recreate tables in an isolated copy; do not silently edit committed results or tables.

    This checks table freshness, not historical checkpoint identity or independent numerical results.
    Figure generators are smoke-tested, not compared by PDF bytes (metadata/font versions vary).
    """
    needed=['paper/make_tables.py','paper/make_thesis.py','paper/make_figs.py','results']
    missing=[p for p in needed if not (root/p).exists()]
    if missing:return [f'Missing regeneration input: {p}' for p in missing],[]
    runs=[];errors=[]
    with tempfile.TemporaryDirectory(prefix='npuloop-paper-') as temp:
        work=Path(temp)/'repo'
        work.mkdir()
        # These generators read only paper/ and results/; do not copy credentials or weights.
        for name in ('paper','results'):
            shutil.copytree(root/name,work/name,ignore=shutil.ignore_patterns('__pycache__','*.aux','*.log','*.out','*.toc'))
        commands=[['paper/make_tables.py','--lang','ko'],['paper/make_thesis.py'],['paper/make_figs.py','--lang','ko']]
        for args in commands:
            try:
                done=subprocess.run([sys.executable,*args],cwd=work,capture_output=True,text=True,timeout=180,check=False)
            except subprocess.TimeoutExpired:
                errors.append(f'Generator timed out: {args[0]}');break
            runs.append({'command':[sys.executable,*args],'exit_code':done.returncode,'stdout':done.stdout,'stderr':done.stderr})
            if done.returncode:
                errors.append(f'Generator failed: {args[0]}');break
        if not errors:
            for name in TABLE_ROWS:
                before=root/'paper'/name;after=work/'paper'/name
                if not before.is_file() or not after.is_file():errors.append(f'Missing generated table: {name}')
                elif before.read_bytes().replace(b'\r\n',b'\n')!=after.read_bytes().replace(b'\r\n',b'\n'):
                    errors.append(f'Stale table: paper/{name}')
    return errors,runs

def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--paper',action='store_true')
    parser.add_argument('--regenerate',action='store_true')
    parser.add_argument('--snapshot',type=Path,help='JSON list of path/sha256 references for a pinned delivery')
    parser.add_argument('--inventory',nargs='+',metavar='RELATIVE_FILE')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if not any((args.paper,args.regenerate,args.snapshot,args.inventory)):
        parser.error('Choose --paper, --regenerate, --snapshot, or --inventory')
    result:dict[str,Any]={'checks':[],'errors':[],'independent_experiment_rerun':False,'historical_checkpoint_identity_verified':False}
    try:
        root=args.root.resolve(strict=True)
        if args.paper or args.regenerate:
            result['errors']+=paper_errors(root);result['checks'].append('paper_structure_and_known_claim_regressions')
        if args.snapshot:
            result['errors']+=snapshot_errors(root,json.loads(args.snapshot.read_text(encoding='utf-8')))
            result['checks'].append('pinned_snapshot_bytes')
        if args.regenerate:
            e,runs=regeneration_errors(root);result['errors']+=e;result['generator_runs']=runs
            result['checks'].append('table_regeneration_attempt')
        if args.inventory:
            result['artifacts']=artifact_inventory(root,args.inventory)
            result['checks'].append('artifact_availability_and_hashes')
            result['errors'] += [f"Missing artifact: {x['path']}" for x in result['artifacts'] if x['status']=='missing']
    except (ValueError,OSError,KeyError,TypeError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['status']='failed' if result['errors'] else 'passed'
    output=json.dumps(result,ensure_ascii=False,indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(output+'\n',encoding='utf-8')
    return int(bool(result['errors']))

if __name__=='__main__':raise SystemExit(main())
