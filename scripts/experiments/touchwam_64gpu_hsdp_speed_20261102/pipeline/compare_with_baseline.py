"""HSDP speed run vs r2 on the same steps, with the release's own compare tool (speed per modality,
data stalls, memory, losses). The wait for a step's data and the fastest rank's step are both reported.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
from runtime import BASELINE_RUN, OUTPUT, ROOT  # noqa: E402

run = sorted(p for p in OUTPUT.iterdir() if p.is_dir())[-1]
with tempfile.TemporaryDirectory() as root:
    for tag, source in (('r2', BASELINE_RUN), ('hsdp', run)):
        (Path(root) / tag).mkdir()
        (Path(root) / tag / source.name).symlink_to(source)
    subprocess.run([sys.executable, str(ROOT / 'release/scripts/perf/touchwam_a800/compare.py'), 'r2', 'hsdp',
                    '--root', root, '--json', str(ROOT / 'results/baseline_comparison.json'), *sys.argv[1:]], check=True)
