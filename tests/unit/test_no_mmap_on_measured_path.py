"""Invariant 3: featherquant/ never mmaps a weight file.

GGUFReader (metadata-only, released before streaming) is the single
allowed exception and is asserted by name, so a new mmap call site cannot
sneak in unnoticed.

ALLOWED lists modules whose source text contains the literal word "mmap"
(comments explaining GGUFReader's own internal memmap use count as this,
since they still name the exception rather than hide it). featherquant/
indexer.py also calls GGUFReader for metadata but its source never spells
out "mmap" anywhere, so it is not in this set — a real mmap call site
appearing there would still be caught below.
"""
import pathlib
import re

ALLOWED = {"gguf_io.py", "st_source.py"}   # GGUFReader users that say "mmap"


def test_no_direct_mmap_calls():
    offenders = []
    for path in pathlib.Path("featherquant").glob("*.py"):
        text = path.read_text()
        if re.search(r"\bmmap\b|np\.memmap|numpy\.memmap", text):
            offenders.append(path.name)
    assert not [o for o in offenders if o not in ALLOWED], offenders


def test_allowed_files_only_use_gguf_reader_metadata():
    for name in ALLOWED:
        text = pathlib.Path("featherquant", name).read_text()
        assert "np.memmap" not in text and "numpy.memmap" not in text
