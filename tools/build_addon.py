"""Build a dependency-free Blender add-on ZIP from an explicit source allowlist."""
from pathlib import Path
import os
import tempfile
import zipfile
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'neural_operator_blender'
output = ROOT / 'dist' / 'caffe-gui-tool-2.0.0.zip'
output.parent.mkdir(exist_ok=True)

def add_file(archive, file, name):
    # Fixed ZIP metadata makes the same source produce the same deliverable.
    entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    archive.writestr(entry, file.read_bytes())

with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip', delete=False) as stream:
    temporary = Path(stream.name)
try:
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(PACKAGE.rglob('*')):
            if file.is_file() and file.suffix in {'.py', '.json', '.prototxt'}:
                add_file(archive, file, str(file.relative_to(ROOT)))
        for relative in ['LICENSE', 'LICENSES/Unlicense.txt', 'LICENSES/MIT.txt', 'NOTICE', 'README.md']:
            add_file(archive, ROOT / relative, 'neural_operator_blender/' + relative)
    temporary.chmod(0o644)
    os.replace(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
print(output)
