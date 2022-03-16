from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
from collections import deque
from gym import spaces
from enum import Enum
import numpy as np
import operator
import warnings
import random
import pickle
import torch
import os


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self.neutral_element = neutral_element
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

    def remove_items(self, num_items):
        del self._value[self._capacity:(self._capacity + num_items)]
        neutral_elements = [self.neutral_element for _ in range(num_items)]
        self._value += neutral_elements
        for idx in range(self._capacity - 1, 0, -1):
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


class OUNoise(object):
    def __init__(self, dim, low, high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10_000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = dim
        self.low = low
        self.high = high

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        action = action.cpu().detach().numpy()
        return np.clip(action + ou_state, self.low, self.high)


def empty_torch_queue(q):
    while True:
        try:
            o = q.get_nowait()
            del o
        except:
            break
    q.close()


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create replay buffer.
        Args:
            size (int): max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)
        self._storage.append(data)
        self._next_idx += 1

    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return [np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            gammas)]

    def sample(self, batch_size, **kwags):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        weights = np.zeros(len(idxes))
        inds = np.zeros(len(idxes))
        return self._encode_sample(idxes) + [weights, inds]

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
}


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, save_dir):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size * 2:  # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def remove(self, num_samples):
        super().remove(num_samples)
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.6):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx  # < len(self._storage)
            idx = int(idx)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


def get_time_limit(env, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.
    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return 1,
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return int(len(observation_space.nvec)),
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return int(observation_space.n),
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


from gym.spaces import box


class HerReplayBuffer(ReplayBuffer):
    def __init__(self, config, save_dir, buffer_size: int, device: Union[torch.device, str] = "cpu", replay_buffer=None,
        max_episode_length: Optional[int] = None, n_sampled_goal: int = 4, goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = False, handle_timeout_termination: bool = True, clip_obs: float = 10.0, epsilon: float = 1e-8):
        super(HerReplayBuffer, self).__init__(size=buffer_size)

        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"

        self.n_sampled_goal = n_sampled_goal
        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = config['max_ep_length']
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = buffer_size if online_sampling else self.max_episode_length

        self.buffer_size = her_buffer_size
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.config = config

        if online_sampling:
            replay_buffer = None
        self.replay_buffer = replay_buffer
        self.online_sampling = online_sampling
        self.distance_threshold = 0.05

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = max(self.buffer_size // self.max_episode_length, 1)
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0
        self.full = False
        self.pos = 0
        self.current_idx = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(box.Box(shape=[self.config['obs_shape']], high=np.inf, low=-np.inf))
        self.goal_shape = get_obs_shape(box.Box(shape=[self.config['goal_shape']], high=np.inf, low=-np.inf))

        # input dimensions for buffer initialization
        input_shape = {
            "observation": (1,) + self.obs_shape,
            "achieved_goal": (1,) + self.goal_shape,
            "desired_goal": (1,) + self.goal_shape,
            "action": (self.config['action_dim'],),
            "reward": (1,),
            "next_obs": (1,) + self.obs_shape,
            "next_achieved_goal": (1,) + self.goal_shape,
            "next_desired_goal": (1,) + self.goal_shape,
            "done": (1,),
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def sample_her(self, batch_size, env):
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.
        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        if self.replay_buffer is not None:
            return self.replay_buffer.sample(batch_size, env)
        return self._sample_transitions(batch_size, maybe_vec_env=env, online_sampling=True)  # pytype: disable=bad-return-type

    def _sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.
        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # `maybe_vec_env=None` as we should store unnormalized transitions,
        # they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.
        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self._buffer["achieved_goal"][her_episode_indices, transitions_indices]

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def _sample_transitions(self, batch_size: Optional[int], maybe_vec_env, online_sampling: bool, n_sampled_goal: Optional[int] = None):
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.compute_reward(
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )[0]
        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        return observations, next_observations, transitions["action"], transitions["reward"]

    def _normalize_obs(self, obs: np.ndarray, obs_rms) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        gamma):

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done
            )

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            if not self.online_sampling:
                # sample virtual transitions and store them in replay buffer
                self._sample_her_transitions()
                # clear storage for current episode
                self.reset()

            self.episode_steps = 0

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals to create virtual transitions for the last episode.
        observations, next_observations, actions, rewards = self._sample_offline(n_sampled_goal=self.n_sampled_goal)

        # Store virtual transitions in the replay buffer, if available
        if len(observations) > 0:
            for i in range(len(observations["observation"])):
                self.replay_buffer.add(
                    {key: obs[i] for key, obs in observations.items()},
                    {key: next_obs[i] for key, next_obs in next_observations.items()},
                    actions[i],
                    rewards[i],
                    # We consider the transition as non-terminal
                    done=[False],
                    infos=[{}],
                )

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        current_idx = self.current_idx

        # truncate interrupted episode
        if current_idx > 0:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # get current episode and transition index
            pos = self.pos
            # set episode length for current episode
            self.episode_lengths[pos] = current_idx
            # set done = True for current episode
            # current_idx was already incremented
            self._buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
            # reset current transition index
            self.current_idx = 0
            # increment episode counter
            self.pos = (self.pos + 1) % self.max_episode_stored
            # update "full" indicator
            self.full = self.full or self.pos == 0

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


def create_replay_buffer(config, save_dir):
    size = config['replay_mem_size']
    if config['replay_memory_prioritized']:
        alpha = config['priority_alpha']
        return PrioritizedReplayBuffer(size=size, alpha=alpha, save_dir=save_dir)
    elif config['her_memory']:
        return HerReplayBuffer(config=config, buffer_size=size, save_dir=save_dir)
    return ReplayBuffer(size)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fast_clip_grad_norm(parameters, max_norm):
    r"""Clips gradient norm of an iterable of parameters.
    Only support norm_type = 2
    max_norm = 0, skip the total norm calculation and return 0
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    if abs(max_norm) < 1e-6:  # max_norm = 0
        return 0
    else:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = torch.stack([(p.grad.detach().pow(2)).sum() for p in parameters]).sum().sqrt().item()
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        return total_norm


class TanhNormal(object):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, config, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        super().__init__()
        self.config = config
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = torch.distributions.Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1+value) / (1-value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (self.normal_mean + self.normal_std * torch.distributions.Normal(torch.zeros(self.normal_mean.size()), torch.ones(self.normal_std.size())).sample().to(self.config['device']))
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return torch.from_numpy(np_array_or_other).float()
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return tensor_or_other.detach().numpy()
    else:
        return tensor_or_other


def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    if isinstance(outputs, tuple):
        return tuple(np_ify(x) for x in outputs)
    else:
        return np_ify(outputs)

"""
def test_goals(t):
    if t < 25:
        return [2.5, -2.0]
    elif 25 <= t < 50:
        return [3.5, -0.5]
    elif 50 <= t < 75:
        return [0.5, -3.5]
    elif t >= 75:
        return [3.5, -3.5]


def test_goals(t):
    if t < 25:
        return [-1.5, -1.5]
    elif 25 <= t < 50:
        return [1.5, -1.5]
    elif 50 <= t < 75:
        return [-1.5, 1.5]
    elif t >= 75:
        return [1.5, 1.5]
"""

def test_goals(t):
    if t < 2:
        return [-1.5, -1.5]
    elif 2 <= t < 3:
        return [1.5, -1.5]
    elif 3 <= t < 4:
        return [-1.5, 1.5]
    elif t >= 4:
        return [1.5, 1.5]

"""
def test_goals(t):
    if t < 2:
        return [1.5, -1.5]  # return [1.5, -1.5] [2.5, -2.0]
    elif 2 <= t < 3:
        return [3.5, -0.5]
    elif 3 <= t < 4:
        return [0.5, -3.5]
    elif t >= 4:
        return [3.5, -3.5]
"""
