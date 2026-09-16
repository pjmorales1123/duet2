"""Ensure cabinet browser requests match the local server's strict schemas."""
import unittest
from pathlib import Path
from pydantic import ValidationError
from simulation_lab.server import LanguageCommand, Reset, TaskCommand


class ServerContractTests(unittest.TestCase):
    def test_browser_instruction_accepts_recording_flag(self):
        request = LanguageCommand(text='set the table', mode='programmed', record=False)
        self.assertFalse(request.record)

    def test_drawer_task_and_reset_fields_are_not_available(self):
        with self.assertRaises(ValidationError):
            TaskCommand(kind='drawer_open')
        with self.assertRaises(ValidationError):
            Reset(drawer_open=True)

    def test_web_has_five_skill_goal_and_no_drawer_controls(self):
        web = Path(__file__).parents[1]/'simulation_lab'/'web'
        html = (web/'index.html').read_text(encoding='utf-8')
        script = (web/'app.js').read_text(encoding='utf-8')
        self.assertIn('Run all five skills', html)
        for control in ('drawer-open', 'drawer-status', 'drawer-value'):
            self.assertNotIn(control, html)
            self.assertNotIn(control, script)


if __name__ == '__main__':
    unittest.main()
