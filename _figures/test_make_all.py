"""Verify that the public rebuild command propagates failed numeric validation.

Run: python3 -m unittest discover -s _figures -p 'test_*.py'
External rendering commands are stubbed in an isolated project under _tmp/.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RebuildValidationTest(unittest.TestCase):
    def run_rebuild(self, status):
        with tempfile.TemporaryDirectory(dir=ROOT / "_tmp", prefix="rebuild-test-") as tmp:
            project = Path(tmp)
            figures, commands = project / "_figures", project / "bin"
            figures.mkdir()
            commands.mkdir()
            shutil.copyfile(ROOT / "_figures/make_all.sh", figures / "make_all.sh")
            (figures / "fig_dummy.py").write_text("# test fixture\n")
            scripts = {
                "python": '#!/bin/sh\nif [ "$1" = check_paper_numbers.py ]; then exit "$MOCK_CHECK_STATUS"; fi\nexit 0\n',
                "pdflatex": '#!/bin/sh\ntouch ../_tmp/fig01/fig_01_study_design.pdf\n',
                "pdftoppm": "#!/bin/sh\nexit 0\n",
            }
            for name, source in scripts.items():
                path = commands / name
                path.write_text(source)
                path.chmod(0o755)
            env = dict(os.environ, PY=str(commands / "python"),
                       PATH=f"{commands}:{os.environ['PATH']}", MOCK_CHECK_STATUS=str(status))
            return subprocess.run(["bash", str(figures / "make_all.sh")], env=env,
                                  capture_output=True, text=True, timeout=30)

    def test_failed_number_check_stops_rebuild(self):
        result = self.run_rebuild(17)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertNotIn("done", result.stdout.splitlines())

    def test_successful_number_check_allows_completion(self):
        result = self.run_rebuild(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-1], "done")


if __name__ == "__main__":
    unittest.main()
