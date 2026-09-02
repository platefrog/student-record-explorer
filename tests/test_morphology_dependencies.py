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

    def test_requirements_and_spec_include_truststore(self):
        requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8')
        spec = (ROOT / 'StudentRecordExplorer.spec').read_text(encoding='utf-8')
        self.assertIn('truststore', requirements)
        self.assertIn("'truststore'", spec)

    def test_release_bundle_includes_major_corpus(self):
        spec = (ROOT / 'StudentRecordExplorer.spec').read_text(encoding='utf-8')
        release_check = (ROOT / 'scripts' / 'release_verify.py').read_text(encoding='utf-8')
        dependency_check = (ROOT / 'scripts' / 'dependency_audit.py').read_text(encoding='utf-8')
        self.assertIn('data/major_corpus.db', spec)
        self.assertIn('major_corpus.db', release_check)
        self.assertIn('major_corpus.db', dependency_check)


if __name__ == '__main__':
    unittest.main()
