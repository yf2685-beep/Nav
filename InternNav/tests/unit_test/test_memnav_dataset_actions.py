import unittest

import numpy as np

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset


class MemNavBuildActionsTest(unittest.TestCase):
    def test_aux_goal_translation_uses_the_actual_endpoint(self):
        dataset = object.__new__(MemNav_Dataset)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
        ])
        dataset.process_actions = lambda *args, **kwargs: (
            points, None, None, None, np.array([0, 1])
        )
        # Like NavDP.xyz_to_xyt, the last row stores the penultimate point.
        dataset.xyz_to_xyt = lambda *args, **kwargs: np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
        ])
        actions, goal = dataset._build_actions(
            np.repeat(np.eye(4)[None], 3, axis=0), np.eye(4), pred_digit=1
        )
        np.testing.assert_allclose(goal, [2.0, 0.5, 0.25])
        self.assertEqual(actions.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
