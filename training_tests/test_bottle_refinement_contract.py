"""Torch training-coordinate contracts, independent of any trained weights."""
import unittest
import numpy as np
import torch
from simulation_lab.bottle_refinement import image_crops, reconstruct
from simulation_lab.rgb_servo_cameras import project


class RefinementContracts(unittest.TestCase):
    def test_fractional_crop_preserves_linear_pixel_coordinates(self):
        yy, xx = torch.meshgrid(torch.arange(90.), torch.arange(120.), indexing='ij')
        image = torch.stack([xx/120, yy/90, torch.ones_like(xx)*.3])[None]
        centers = torch.tensor([[[40.25, 40.75], [75.75, 46.25]]])
        crops = image_crops(image, centers)
        for key in range(2):
            for row, col in ((0, 0), (31, 31), (63, 63)):
                expected = centers[0, key]+torch.tensor([col-31.5, row-31.5])
                self.assertAlmostEqual(float((crops[key, 0, row, col]+1)*60), float(expected[0]), places=4)
                self.assertAlmostEqual(float((crops[key, 1, row, col]+1)*45), float(expected[1]), places=4)
            self.assertTrue(torch.all(crops[key, 3] == key*2-1))

    def test_calibrated_geometry_needs_two_confident_views(self):
        world = np.array([[.04, -.1, .785], [.04, -.1, .888]])
        matrices = [np.array([[300., 0, 160, shift], [0, 300, 120, 0], [0, 0, 1, 0]]) for shift in (-35., 0., 35.)]
        pixels = np.array([project(world, matrix) for matrix in matrices])
        settings = {'minimum_visibility_probability': .9, 'minimum_heatmap_peak': .03,
                    'maximum_heatmap_variance_px2': 12., 'minimum_views': 2,
                    'grasp_workspace_m': [[-.2, -.25, .86], [.28, .13, 1.0]]}
        calibrations = [{'projection': matrix.tolist()} for matrix in matrices]
        yes = reconstruct(pixels, np.ones((3, 2))*.1, np.ones((3, 2)), np.ones((3, 2)), calibrations, settings)
        self.assertEqual(yes['status'], 'observed')
        np.testing.assert_allclose(yes['grasp_point_m'], world[1], atol=1e-9)
        visibility = np.zeros((3, 2)); visibility[0] = 1
        no = reconstruct(pixels, np.ones((3, 2))*.1, np.ones((3, 2)), visibility, calibrations, settings)
        self.assertEqual(no['status'], 'refused')


if __name__ == '__main__':
    unittest.main()
