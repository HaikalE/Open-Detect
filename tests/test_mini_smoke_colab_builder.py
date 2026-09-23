import ast
import unittest

from pilot.build_mini_smoke_colab import build


class MiniSmokeColabBuilderTest(unittest.TestCase):
    def test_generated_cells_compile_and_tests_use_discovery(self):
        notebook = build()
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        ast.parse(code)
        self.assertIn("'unittest', 'discover'", code)
        self.assertNotIn("'tests.test_temporal_fusion'", code)
        self.assertIn('print(result.stderr)', code)


if __name__ == '__main__':
    unittest.main()
