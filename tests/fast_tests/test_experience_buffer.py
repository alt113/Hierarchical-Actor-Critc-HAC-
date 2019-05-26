import unittest
import numpy as np
from hac.experience_buffer import ExperienceBuffer


class TestExperienceBuffer(unittest.TestCase):
    """Tests the methods of the ExperienceBuffer object."""

    def test_experience_buffer(self):
        # some fixed variables for this test
        batch_size = 2
        max_buffer_size = 12
        state_dim = 5
        goal_dim = 4
        action_dim = 3

        # Instantiate the buffer object.
        buff = ExperienceBuffer(max_buffer_size, batch_size)

        # Test the `add` method.
        for i in range(max_buffer_size-1):
            # add a new (random) element
            state = np.random.rand(state_dim)
            next_state = np.random.rand(state_dim)
            action = np.random.rand(action_dim)
            goal = np.random.rand(goal_dim)
            terminal = False
            reward = 0
            info = {}
            buff.add([state, action, reward, next_state, goal, terminal, info])

            # check the size of the buffer after adding each element
            self.assertEqual(buff.size, i+1)

        # Test the `add` method one the max buffer size is hit.
        #
        # This is done by adding a new element once the buffer is full, and
        # checking the new size of the buffer once this is done.
        state = np.random.rand(state_dim)
        next_state = np.random.rand(state_dim)
        action = np.random.rand(action_dim)
        goal = np.random.rand(goal_dim)
        terminal = False
        reward = 0
        info = {}
        buff.add([state, action, reward, next_state, goal, terminal, info])
        self.assertEqual(buff.size, int(5/6*max_buffer_size))

        # Test the `get_batch` method.
        s, a, r, s_p, g, t = buff.get_batch()
        self.assertTupleEqual(np.array(r).shape, (batch_size,))
        self.assertTupleEqual(np.array(t).shape, (batch_size,))
        self.assertTupleEqual(np.array(s).shape, (batch_size, state_dim))
        self.assertTupleEqual(np.array(a).shape, (batch_size, action_dim))
        self.assertTupleEqual(np.array(g).shape, (batch_size, goal_dim))
        self.assertTupleEqual(np.array(s_p).shape, (batch_size, state_dim))


if __name__ == '__main__':
    unittest.main()
