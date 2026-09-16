"""Validate publication assets without claiming a historical experiment rerun."""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

root=Path.cwd()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'verification/publish/source-import.json').read_text())
for name,digest in manifest['source_files'].items():
    assert sha(root/name)==digest, 'Approved source changed: '+name
for prefix,key in [('results','protected_results'),('paper','protected_tables')]:
    for name,digest in manifest[key].items():
        assert sha(root/prefix/name)==digest, 'Protected result changed: '+name
page=(root/'docs/index.html').read_text(encoding='utf-8')
assert (root/'docs/index.html').read_bytes()==(root/'demo/index.html').read_bytes()
assert 'NPU 모델 분석·검증' in page and 'NpuWorkbenchApp' in page
assert '모델을 INT8로 바꾼 뒤, 무엇을 고쳐야 할까요?' not in page
pdf=root/'paper/npuloop_thesis.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==27, 'Unexpected manuscript page count'
cover=reader.pages[0].extract_text()
assert '윤기혁' in cover and '[저자 성명]' not in cover
for name in ['paper/npuloop_thesis_reviewed.pdf','docs/research/npuloop_thesis.pdf','demo/research/npuloop_thesis.pdf']:
    assert sha(root/name)==sha(pdf), 'PDF copy mismatch: '+name
for folder in ['docs','demo']:
    for name in ['USAGE.md','VALIDATION_SCOPE.md','EXPERIMENTS.md']:
        assert (root/folder/'reference'/name).is_file(), folder+'/'+name
result={'status':'passed','source_files':len(manifest['source_files']),
        'research_results_unchanged':len(manifest['protected_results']),
        'table_sources_unchanged':len(manifest['protected_tables']),
        'manuscript_pages':len(reader.pages),'pdf_sha256':sha(pdf),
        'html_sha256':sha(root/'docs/index.html'),
        'historical_experiments_rerun':False,'hardware_tested':False}
(root/'verification/publish/outputs.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
