"""Apply the approved, hash-checked source delta only after a complete preflight."""
from pathlib import Path
import hashlib
import json
import lzma
import subprocess

ROOT = Path.cwd().resolve()
def sha(data): return hashlib.sha256(data).hexdigest()
def safe(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or '.git' in p.parts or not p.parts:
        raise ValueError('Unsafe path: ' + name)
    out = ROOT / p
    if not out.resolve().is_relative_to(ROOT) or any(x.is_symlink() for x in [out, *out.parents] if x.is_relative_to(ROOT)):
        raise ValueError('Symlink or escaping path: ' + name)
    return out

def load_payload():
    ready = json.loads(safe('.publish/ready.json').read_text())
    chunks = []
    for item in ready['parts']:
        b = safe(item['path']).read_bytes()
        assert len(b) == item['size'] and sha(b) == item['sha256'], item['path']
        chunks.append(b)
    archive = b''.join(chunks)
    assert sha(archive) == ready['archive_sha256'], 'Archive digest'
    raw = lzma.decompress(archive)
    assert sha(raw) == ready['payload_sha256'], 'Payload digest'
    payload = json.loads(raw)
    assert payload['base_commit'] == ready['base_commit']
    assert len(payload['files']) == ready['source_files']
    return payload

def main():
    payload = load_payload()
    base = subprocess.check_output(['git', 'rev-parse', 'origin/master'], text=True).strip()
    assert base == payload['base_commit'], 'Remote base changed; reconcile before importing'
    planned = {}
    for name, item in payload['files'].items():
        path = safe(name)
        before = path.read_bytes() if path.exists() else None
        assert (sha(before) if before is not None else None) == item['before'], 'Source conflict: ' + name
        if 'text' in item:
            result = item['text'].encode('utf-8')
        else:
            lines = before.decode('utf-8').splitlines(keepends=True)
            pieces = []
            cursor = 0
            for start, end, replacement in item['edits']:
                assert cursor <= start <= end <= len(lines), name
                pieces += lines[cursor:start] + replacement
                cursor = end
            result = ''.join(pieces + lines[cursor:]).encode('utf-8')
        assert sha(result) == item['sha256'], 'Result digest: ' + name
        planned[name] = result
    for prefix, key in [('results', 'protected_results'), ('paper', 'protected_tables')]:
        for name, digest in payload[key].items():
            assert sha(safe(prefix + '/' + name).read_bytes()) == digest, name
    assert not payload['remove'], 'This publication must not delete existing research'
    for name, result in planned.items():
        path = safe(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result)
    manifest = {'base_commit': base, 'source_files': {k:v['sha256'] for k,v in payload['files'].items()},
                'protected_results':payload['protected_results'], 'protected_tables':payload['protected_tables'],
                'status':'source-import-verified', 'all_preconditions_checked':True}
    out = safe('verification/publish/source-import.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('Imported', len(planned), 'approved source files; protected results and tables unchanged.')

if __name__ == '__main__': main()
