import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_release_source_is_consistent_utf8_and_current(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'release_verify.py'), '--check'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
