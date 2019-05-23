"""Contains the experience replay buffer object."""
import numpy as np


class ExperienceBuffer:
    """Experience replay buffer object.

    TODO

    Attributes
    ----------
    size : int
        the current number of samples in the replay buffer
    max_buffer_size : int
        the maximum number of elements allowed to be in the buffer
    experiences : list of list
        list of element samples, stored (by index) as follows:

        * TODO
    batch_size : int
        the number of elements to be returned whenever `get_batch` is called
    """

    def __init__(self, max_buffer_size, batch_size):
        """Instantiate the experience replay object.

        Parameters
        ----------
        max_buffer_size : int
            TODO
        batch_size : int
            TODO
        """
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size

    def add(self, experience):
        """Add a sample to the replay buffer.

        When the buffer is filled, this method also removes the first sixth of
        the element from the buffer. This is done because only removing a
        single transition slows down performance.

        Parameters
        ----------
        experience : TODO
            TODO

        Raises
        ------
        AssertionError
            TODO
        AssertionError
            TODO
        """
        assert len(experience) == 7, \
            'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool

        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer.
        # Only removing a single transition slows down performance
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self):
        """TODO

        Returns
        -------
        TODO
            TODO
        """
        states, actions, rewards, new_states, goals, is_terminals = \
            [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=self.batch_size)

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals
