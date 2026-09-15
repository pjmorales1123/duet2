import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from simulation_lab.storage import require_space, GIB


class StorageTests(unittest.TestCase):
    def test_refuses_before_writing_with_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'not-created'/'dataset'
            with patch('simulation_lab.storage.shutil.disk_usage', return_value=SimpleNamespace(free=10*GIB+99)):
                with self.assertRaises(OSError):require_space(target, 100)
            self.assertFalse(target.exists())

    def test_checks_current_free_space_each_time(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('simulation_lab.storage.shutil.disk_usage', side_effect=[SimpleNamespace(free=12*GIB),SimpleNamespace(free=9*GIB)]):
                self.assertEqual(require_space(folder,GIB)['reserve_bytes'],10*GIB)
                with self.assertRaises(OSError):require_space(folder,0)
