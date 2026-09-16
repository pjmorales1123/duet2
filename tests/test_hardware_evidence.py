import unittest
from simulation_lab.hardware_evidence import all_inference_devices_are_intel


class InferenceIdentityTests(unittest.TestCase):
    def test_generic_gpu_identifier_does_not_establish_intel_inference(self):
        measured={'bottle':{'device':'GPU','name':'NVIDIA GeForce RTX 4070 (dGPU)','parity_passed':True}}
        self.assertFalse(all_inference_devices_are_intel(measured))

    def test_every_skill_needs_identified_intel_device_and_passing_parity(self):
        measured={'bottle':{'device':'CPU','name':'Intel Core Ultra 7 258V','parity_passed':True},
                  'mug':{'device':'GPU.0','name':'Intel Arc Graphics','parity_passed':True}}
        self.assertTrue(all_inference_devices_are_intel(measured))
        measured['mug']['parity_passed']=False
        self.assertFalse(all_inference_devices_are_intel(measured))
        measured['mug']={'device':'GPU','parity_passed':True}
        self.assertFalse(all_inference_devices_are_intel(measured))

    def test_no_benchmark_is_not_a_success(self):
        self.assertFalse(all_inference_devices_are_intel({}))


if __name__=='__main__':unittest.main()
