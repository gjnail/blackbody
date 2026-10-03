"""Every source file compiles. A syntax error in a module that only a window imports would otherwise go unnoticed by the
other tests (the export dialog once had one, and the app could not start). GPU-free."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_source_file_compiles():
    bad = []
    for top in ('blackbody', 'tools', 'tests'):
        for p in sorted((ROOT / top).rglob('*.py')):
            try:
                compile(p.read_text(encoding='utf-8'), str(p), 'exec')
            except SyntaxError as e:
                bad.append(f'{p.relative_to(ROOT)}:{e.lineno}: {e.msg}')
    assert not bad, bad
