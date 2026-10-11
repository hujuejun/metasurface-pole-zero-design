"""Cross-platform manifest ordering and genuine mismatch regression checks."""
import pytest
import check_release


def test_inventory_uses_platform_independent_string_order(tmp_path, monkeypatch):
    for name in ['cases.json', 'CITATION.cff', 'README.md', 'check_release.py']:
        (tmp_path/name).write_text('test', encoding='utf-8')
    monkeypatch.setattr(check_release, 'ROOT', tmp_path)
    names=[p.relative_to(tmp_path).as_posix() for p in check_release.inventory()]
    assert names==['CITATION.cff', 'README.md', 'cases.json', 'check_release.py']


def test_manifest_accepts_windows_and_posix_ordering():
    rows=[{'file':name,'sha256':name} for name in ['cases.json','CITATION.cff','README.md']]
    windows=sorted(rows,key=lambda r:r['file'].lower())
    posix=sorted(rows,key=lambda r:r['file'])
    assert windows!=posix
    check_release.check_manifest(windows,posix)


@pytest.mark.parametrize('actual, message',[
    ([], 'Missing'),
    ([{'file':'a','sha256':'different'}], 'changed'),
    ([{'file':'a','sha256':'hash'},{'file':'b','sha256':'hash'}], 'added'),
    ([{'file':'a','sha256':'hash'},{'file':'a','sha256':'hash'}], 'Duplicate'),
])
def test_manifest_rejects_real_mismatches(actual,message):
    with pytest.raises(ValueError,match=message):
        check_release.check_manifest(actual,[{'file':'a','sha256':'hash'}])
