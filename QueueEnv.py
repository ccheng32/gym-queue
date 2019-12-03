import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces.space import Space
from gym.spaces.discrete import Discrete
import numpy as np

class QueueEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, max_slack=float(1e9), queue_size=int(1e6), max_wrongs=3, past_steps=10, seed=None):
      # Fixed variables.
      self.max_slack = max_slack
      self.past_steps = past_steps
      self.max_wrongs = max_wrongs
      self.seed = seed
      self.queue_size = queue_size
      self.init_queue = list()
      init_slack_rng, _ = seeding.np_random(self.seed)
      self.observation_space = Space([queue_size], np.dtype(float))
      self.action_space = Discrete(queue_size)
      for i in range(self.queue_size):
          self.init_queue.append(init_slack_rng.random_sample() * self.max_slack)

      # Variables that can be reset.
      self.slack_rng, _ = seeding.np_random(self.seed)
      self.wrong_deques = [0] * self.past_steps
      self.wrong_deque_idx = 0
      self.queue = list(self.init_queue)

  # action is a set of indices. The indices point to the packets that should be removed from the queue
  # in this step.
  def step(self, action):
      rewards = 0.0
      done = False

      # First deque packet from queue.
      deleted_packet = self.queue[action]
      del self.queue[action]

      # Check if the deleted packet matches LSTF policy.
      self.wrong_deques[self.wrong_deque_idx] = 0
      if len(self.queue) > 0:
          min_slack = min(self.queue)
          if deleted_packet <= min_slack:
              rewards += 1.0
          else:
              self.wrong_deques[self.wrong_deque_idx] += 1
              rewards -= 1.0
      
      # If there are more than max_wrongs wrong deques in the previous past_steps steps,
      # then we end the episode.
      self.wrong_deque_idx = (self.wrong_deque_idx + 1) % self.past_steps
      if sum(self.wrong_deques) > self.max_wrongs:
          done = True

      # Enque one incoming packet.
      self.queue.append(self.max_slack * self.slack_rng.random_sample())

      # Decrement the slacks in the queue.
      for i in range(len(self.queue)):
          self.queue[i] -= 1.0

      # Observation is (# of packets to deque, the queue itself)
      obs = list(self.queue)

      return obs, rewards, done, None

  def reset(self):
      self.slack_rng, _ = seeding.np_random(self.seed)
      self.queue = list()
      self.wrong_deques = [0] * self.past_steps
      self.wrong_deque_idx = 0
      self.queue = list(self.init_queue)
      obs = list(self.queue)
      return obs

  def render(self, mode='human', close=False):
      print(self.queue)
