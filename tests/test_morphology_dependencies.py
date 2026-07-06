import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MorphologyDependencyTests(unittest.TestCase):
    def test_requirements_include_kiwi_model_package(self):
        text = (ROOT / 'requirements.txt').read_text(encoding='utf-8')
        self.assertIn('kiwipiepy-model', text)

    def test_spec_collects_kiwi_model_package(self):
        text = (ROOT / 'StudentRecordExplorer.spec').read_text(encoding='utf-8')
        self.assertIn("'kiwipiepy_model'", text)


if __name__ == '__main__':
    unittest.main()
