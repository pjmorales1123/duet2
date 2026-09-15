import unittest
from simulation_lab.learned_plan import plan_learned_steps
from simulation_lab.language import parse_command,CommandError


class LearnedPlanTests(unittest.TestCase):
    def test_camera_clearance_changes_the_plan(self):
        blocked={'bottle_path_clear':False,'drawer_path_clear':False,'drawer_open':False}
        clear={'bottle_path_clear':True,'drawer_path_clear':True,'drawer_open':True}
        self.assertEqual(plan_learned_steps(parse_command('place the plate'),blocked)['steps'],['bottle','plate'])
        self.assertEqual(plan_learned_steps(parse_command('place the plate'),clear)['steps'],['plate'])
        self.assertEqual(plan_learned_steps(parse_command('place the fork'),blocked)['steps'],['bottle','plate','drawer','fork'])
        self.assertEqual(plan_learned_steps(parse_command('place the fork'),clear)['steps'],['fork'])

    def test_unknown_observation_and_untrained_destination_are_refused(self):
        for text,scene in [('place the plate',{}),('place the fork',{}),
                           ('place the bottle in the right spot',{}),('place the mug with the left arm',{})]:
            with self.subTest(text=text),self.assertRaises(CommandError):
                plan_learned_steps(parse_command(text),scene)

    def test_explicit_steps_do_not_duplicate_dependencies(self):
        result=plan_learned_steps(parse_command('place the bottle then place the plate then open the drawer'),
                                 {'bottle_path_clear':False,'drawer_path_clear':False,'drawer_open':False})
        self.assertEqual(result['steps'],['bottle','plate','drawer'])

    def test_wider_bottle_uses_both_arms_only_when_models_are_loaded(self):
        scene={'bottle_region':'left_reach','bottle_path_clear':False,
               'drawer_path_clear':False,'drawer_open':False}
        plan=parse_command('place the plate')
        with self.assertRaises(CommandError):plan_learned_steps(plan,scene)
        skills=['reverse_bottle_right','reverse_bottle_left','plate']
        self.assertEqual(plan_learned_steps(plan,scene,skills)['steps'],skills)
        with self.assertRaises(CommandError):
            plan_learned_steps(parse_command('place the bottle with the left arm'),scene,skills)

    def test_relay_requires_complete_pair_and_correct_starting_arm(self):
        plan=parse_command('pass the bottle to the right arm')
        skills=['relay_bottle_left','relay_bottle_right']
        with self.assertRaises(CommandError):plan_learned_steps(plan,{},skills[:1])
        result=plan_learned_steps(plan,{},skills)
        self.assertEqual(result['steps'],skills)
        self.assertIn('table-supported',result['reasons'][0])


if __name__=='__main__':unittest.main()
