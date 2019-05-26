"""Contains the experience replay buffer object."""
import numpy as np


class ExperienceBuffer:
    """Experience replay buffer object.

    Attributes
    ----------
    size : int
        the current number of samples in the replay buffer
    max_buffer_size : int
        the maximum number of elements allowed to be in the buffer
    experiences : dict  TODO: convert to this
        list of element samples, stored in array_like format, with the
        following keys: "state", "action", "next_state", "goal", "reward", and
        "is_terminal"
    batch_size : int
        the number of elements to be returned whenever `get_batch` is called
    """

    def __init__(self, max_buffer_size, batch_size):
        """Instantiate the experience replay object.

        Parameters
        ----------
        max_buffer_size : int
            the maximum number of elements allowed to be in the buffer
        batch_size : int
            the number of elements to be returned whenever `get_batch` is
            called
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
        experience : list
            A new sample element, consisting of the following terms, in order:
            [state, action, reward, next_state, goal, terminal, grip_info]

        Raises
        ------
        AssertionError
            If the sample does not contain 7 elements.
        AssertionError
            If the sixth element is not a boolean term
        """
        assert len(experience) == 7, \
            'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert isinstance(experience[5], bool)

        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer.
        # Only removing a single transition slows down performance
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self):
        """Return a batch of samples.

        Returns
        -------
        array_like
            states, of shape (batch_size, state_sim)
        array_like
            actions, of shape (batch_size, action_sim)
        array_like
            rewards, of shape (batch_size,)
        array_like
            next states, of shape (batch_size, state_sim)
        array_like
            goals, of shape (batch_size, goal_sim)
        array_like
            terminal flags, of shape (batch_size,)
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
