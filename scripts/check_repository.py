"""Check candidate files, source hashes and documentation without network access."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def candidate_files() -> list[Path]:
    if (ROOT / '.git').exists():
        listing = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
        return sorted({ROOT / name.decode() for name in listing.split(b'\0') if name})
    excluded = {'.venv', '.git', '__pycache__', 'runs', 'data', 'build', 'dist'}
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts) & excluded) and not any(part.endswith('.egg-info') for part in p.parts))


def main() -> None:
    failures: list[str] = []
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(f"Source hash mismatch: {entry['path']}")
    candidates = candidate_files()
    text_files = 0
    link_count = 0
    patterns = {
        'cleartext SSH password command': re.compile(r'sshpass\s+-p\s+'),
        'connection account': re.compile(r'root@connect\.[A-Za-z0-9.-]+'),
        'API token': re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
        'GitHub token': re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}\b'),
        'AWS access ID': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
        'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'personal home path': re.compile(re.escape('/' + 'Users' + '/') + r'[^\s"<>]+|' + re.escape('/' + 'home' + '/') + r'[^\s"<>]+'),
    }
    oversized = []
    for path in candidates:
        if path.is_symlink():
            failures.append(f"Unexpected symlink: {path.relative_to(ROOT)}")
            continue
        size = path.stat().st_size
        if size >= 50 * 1024 * 1024:
            oversized.append(str(path.relative_to(ROOT)))
        if path.suffix.lower() in {'.pdf', '.xlsx', '.xls', '.docx', '.sqlite', '.db', '.pem', '.key'} or path.name in {'.env', 'cookies.txt'}:
            failures.append(f"Unexpected private/binary input: {path.relative_to(ROOT)}")
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            failures.append(f"Unexpected non-text candidate: {path.relative_to(ROOT)}")
            continue
        text_files += 1
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"Potential {label}: {path.relative_to(ROOT)} (matched value suppressed)")
        if path.suffix == '.py':
            try:
                compile(text, str(path.relative_to(ROOT)), 'exec')
            except SyntaxError as exc:
                failures.append(f"Syntax error: {path.relative_to(ROOT)}:{exc.lineno}")
        if path.suffix == '.md':
            fence_count = len(re.findall(r'^```', text, flags=re.M))
            if fence_count % 2:
                failures.append(f"Unbalanced code fence: {path.relative_to(ROOT)}")
            links = re.findall(r'\]\(([^)]+)\)', text) + re.findall(r'<img[^>]+src="([^"]+)"', text)
            for target in links:
                target = target.split(' "', 1)[0]
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith('#'):
                    continue
                resolved = (path.parent / unquote(parsed.path)).resolve()
                link_count += 1
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    failures.append(f"Broken/local-escape link: {path.relative_to(ROOT)} -> {target}")
    if oversized:
        failures.append(f"Files above 50 MiB: {oversized}")
    for language in ('README.md', 'README.zh-CN.md'):
        headings = re.findall(r'^## (.+)$', (ROOT / language).read_text(), re.M)
        if len(headings) != 8:
            failures.append(f"README section parity mismatch: {language}")
    for path in [ROOT / 'pyproject.toml', ROOT / 'uv.lock']:
        tomllib.loads(path.read_text())
    import xml.etree.ElementTree as ET
    icon = ET.parse(ROOT / 'assets/mark.svg').getroot()
    if icon.tag != '{http://www.w3.org/2000/svg}svg':
        failures.append('Invalid SVG root')
    for path in candidates:
        if path.suffix == '.json':
            json.loads(path.read_text())
    if any(e['copy_kind'] not in {'byte-identical', 'portable-default-paths-only'} for e in manifest['files']):
        failures.append('Unknown source transformation')
    for entry in manifest['files']:
        if entry['copy_kind']=='byte-identical' and entry['sha256']!=entry['source_sha256']:
            failures.append('Identity copy has differing source hash')
    import yaml
    workflow = yaml.safe_load((ROOT / '.github/workflows/check.yml').read_text())
    if not workflow.get('jobs',{}).get('offline',{}).get('steps'):
        failures.append('Missing offline workflow steps')
    summary = {
        'source_manifest_files_checked': len(manifest['files']),
        'candidate_files': len(candidates),
        'candidate_total_bytes': sum(p.stat().st_size for p in candidates),
        'largest_candidate_bytes': max(p.stat().st_size for p in candidates),
        'text_files_scanned': text_files,
        'local_document_links_checked': link_count,
        'files_at_or_above_50_mib': oversized,
        'failures': failures,
        'scan_scope': 'heuristic token patterns and local paths; not an exhaustive security audit',
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
