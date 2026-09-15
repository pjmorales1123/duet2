import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.verify_live_policy import run


class VerificationEvidenceTests(unittest.TestCase):
    def test_existing_evidence_is_refused_before_starting_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json';output.write_text('preserved failure evidence')
            with patch('scripts.verify_live_policy.LearnedBottleTask') as task:
                with self.assertRaises(FileExistsError):run(SimpleNamespace(output=str(output)))
                task.assert_not_called()
            self.assertEqual(output.read_text(),'preserved failure evidence')


if __name__=='__main__':unittest.main()
