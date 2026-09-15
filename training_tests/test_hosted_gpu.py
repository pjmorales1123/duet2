"""Exercise the hosted neural cache boundary without a cloud allocation."""
import math
from queue import Queue
from threading import Event
import unittest
import numpy as np
import torch
from simulation_lab.hosted_gpu import GeneratedTrajectoryNet
from simulation_lab.policy_observation import ObservationRejected


class HostedTrajectoryContract(unittest.TestCase):
    def test_one_fresh_inference_and_exact_twenty_hz_queries(self):
        requests, responses = Queue(), Queue()
        actions = np.repeat(np.arange(21, dtype='float32')[:, None], 12, axis=1)
        responses.put({'actions': actions, 'device': 'test-device'})
        net = GeneratedTrajectoryNet('bottle', 1., requests, responses, Event())
        visual = torch.zeros(5, 32)
        result = net(visual, torch.tensor([0., .05, .10, 1., 1.05]))
        self.assertEqual(result[:, 0].tolist(), [0., 1., 2., 20., 20.])
        self.assertEqual(requests.get_nowait()[0], 'inference')
        net(visual, torch.tensor([.20]*5))
        self.assertTrue(requests.empty())
        with self.assertRaises(ObservationRejected):
            net(visual+1, torch.tensor([.20]*5))

    def test_invalid_or_mixed_context_cannot_silently_use_cache(self):
        net = GeneratedTrajectoryNet('bottle', 1., Queue(), Queue(), Event())
        for visual, seconds in [(torch.zeros(2, 32), torch.tensor([math.nan])),
                                (torch.tensor([[0.]*32, [1.]*32]), torch.tensor([0.])),
                                (torch.zeros(1, 31), torch.tensor([0.]))]:
            with self.subTest(shape=visual.shape), self.assertRaises(ObservationRejected):
                net(visual, seconds)

    def test_nonfinite_cloud_output_is_refused(self):
        replies = Queue(); actions = np.zeros((21, 12), dtype='float32'); actions[4, 0] = np.nan
        replies.put({'actions': actions, 'device': 'test-device'})
        net = GeneratedTrajectoryNet('bottle', 1., Queue(), replies, Event())
        with self.assertRaises(ObservationRejected):
            net(torch.zeros(1, 32), torch.tensor([0.]))


if __name__ == '__main__':
    unittest.main()
