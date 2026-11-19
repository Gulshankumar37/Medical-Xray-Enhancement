"""Exercise the same menus and parameters that the Streamlit dashboard displays."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTests(unittest.TestCase):
    def test_all_nine_menu_choices_produce_a_download(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.selectbox[0].options), 4)
        seen = set()
        for category in app.selectbox[0].options:
            app.selectbox[0].select(category).run()
            for technique in app.selectbox[1].options:
                with self.subTest(category=category, technique=technique):
                    app.selectbox[1].select(technique).run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertEqual(len(app.get("download_button")), 1)
                    seen.add(technique)
        self.assertEqual(len(seen), 9)

    def test_invalid_settings_remove_the_previous_download_then_recover(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.selectbox[0].select("Contrast enhancement").run()
        app.selectbox[1].select("Contrast stretching").run()
        app.slider[0].set_value((50, 50)).run()
        self.assertFalse(app.exception)
        self.assertIn("two different percentiles", app.error[0].value)
        self.assertEqual(len(app.get("download_button")), 0)
        app.slider[0].set_value((0, 100)).run()
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
