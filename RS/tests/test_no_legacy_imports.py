from __future__ import annotations

from pathlib import Path


def test_rs_mainline_dirs_have_no_legacy_imports():
    roots = [
        Path('/root/autodl-tmp/RouterSense/RS/src'),
        Path('/root/autodl-tmp/RouterSense/RS/deploy'),
        Path('/root/autodl-tmp/RouterSense/RS/experiments'),
    ]
    forbidden = ('routesense_poc1', 'routesense_poc2', 'legacy.poc1', 'legacy.poc2', 'experiment.poc1', 'experiment.poc2')
    skip_names = {'test_no_legacy_imports.py', 'test_package_source_only.py'}
    for root in roots:
        for path in root.rglob('*'):
            if path.name in skip_names:
                continue
            if path.suffix not in {'.py', '.sh', '.md', '.ps1', '.yaml', '.yml', '.toml'}:
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            for token in forbidden:
                assert token not in text, f'{path} references {token}'
