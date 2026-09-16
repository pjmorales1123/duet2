"""Public boundary and compatibility checks without starting a renderer."""
import time
import unittest
from simulation_lab.language import CommandError
from simulation_lab.public_trial import trial_report, validate_request


class PublicTrialContract(unittest.TestCase):
    def test_invalid_or_fractional_seed_is_refused(self):
        for seed in [-1, .5, float('nan'), True, 2147483648]:
            with self.subTest(seed=seed), self.assertRaises(CommandError):
                validate_request('place the bottle', seed, 'learned', 'upright', 'opposite')

    def test_stop_is_not_a_new_scene_instruction(self):
        with self.assertRaises(CommandError):
            validate_request('stop', 42, 'learned', 'upright', 'opposite')

    def test_programmed_intent_list_is_supported_without_private_details(self):
        class Task:
            def snapshot(self):
                return {'status':'running','message':'Moving','intent_plan':[{'object_id':'bottle'}],
                        'steps':['bottle'],'checkpoint':'private/path','results':[]}
        report=trial_report(Task(),42,'programmed','upright',time.perf_counter())
        self.assertEqual(report['plan'],['bottle'])
        self.assertNotIn('checkpoint',report)


if __name__ == '__main__':
    unittest.main()
