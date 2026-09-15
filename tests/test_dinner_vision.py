import unittest
import numpy as np
from simulation_lab.dinner_vision import dinner_features,scene_observation
from simulation_lab.policy_observation import ObservationRejected


class DinnerVisionTests(unittest.TestCase):
    def test_pixels_move_the_detected_centroid_and_ambiguity_is_refused(self):
        images=[np.zeros((240,320,3),dtype=np.uint8) for _ in range(3)]
        images[0][120:130,120:130]=[150,220,230]
        first=dinner_features(images,'plate')
        images[0]=np.roll(images[0],8,axis=1)
        second=dinner_features(images,'plate')
        self.assertAlmostEqual(float(second[0]-first[0]),8/320,places=6)
        images[0][145:155,160:170]=[150,220,230]
        with self.assertRaises(ObservationRejected):dinner_features(images,'plate')

    def test_missing_and_nonfinite_images_cannot_supply_a_scene_plan(self):
        images=[np.zeros((240,320,3),dtype=np.uint8) for _ in range(3)]
        scene=scene_observation(images)
        self.assertTrue(all(scene[k] is None for k in ('bottle_path_clear','drawer_path_clear','drawer_open')))
        for skill in ('plate','mug','fork','spoon','drawer'):
            with self.subTest(skill=skill),self.assertRaises(ObservationRejected):dinner_features(images,skill)
        images[0]=np.full((240,320,3),np.nan)
        with self.assertRaises(ObservationRejected):dinner_features(images,'plate')


if __name__=='__main__':unittest.main()
