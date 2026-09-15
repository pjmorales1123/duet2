import unittest
from simulation_lab.language import parse_command,CommandError

class LanguageTests(unittest.TestCase):
    def test_examples(self):
        p=parse_command('Put the plate in the right spot')['steps'][0]
        self.assertEqual(p['object_id'],'plate');self.assertEqual(p['destination']['side'],'right')
        p=parse_command('take the spoon out of the drawer and put it in the left spot')['steps'][0]
        self.assertTrue(p['destination']['from_drawer']);self.assertEqual(p['destination']['side'],'left')
        p=parse_command('put the pottle infront of the right plate')['steps'][0]
        self.assertEqual(p['object_id'],'bottle');self.assertEqual(p['destination']['reference_side'],'right')
    def test_order_context_and_unsupported_commands(self):
        p=parse_command('open the drawer then place the spoon then move it to the right spot')
        self.assertEqual([s['object_id'] for s in p['steps']],[None,'spoon','spoon'])
        for bad in ('delete all files','put the plate under the table',"do not move the bottle",'move it','put the bottle next to the bottle'):
            with self.subTest(bad=bad),self.assertRaises(CommandError):parse_command(bad)
    def test_cancel(self):self.assertEqual(parse_command('stop')['control'],'cancel')

    def test_speechmatics_spacing(self):
        plan=parse_command('Open the  drawer  , then  place  the  spoon  .')
        self.assertEqual([s['kind'] for s in plan['steps']],['drawer_open','dinner_place'])
        self.assertEqual(plan['steps'][1]['object_id'],'spoon')
        self.assertEqual(parse_command('Place.  The  bottle.')['steps'][0]['object_id'],'bottle')
        with self.assertRaises(CommandError):parse_command('Open the treasure.')

if __name__=='__main__':unittest.main()
