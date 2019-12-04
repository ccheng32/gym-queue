import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces.space import Space
from gym.spaces.discrete import Discrete
import numpy as np

class QueueEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, max_slack=float(1e9), queue_size=int(1e6), max_wrongs=3, past_steps=10, seed=None, target_queue_type="LSTF"):
      # Fixed variables.
      self.max_slack = max_slack
      self.past_steps = past_steps
      self.max_wrongs = max_wrongs
      self.seed = seed
      self.queue_size = queue_size
      self.target_queue_type = target_queue_type
      self.init_queue = list()
      init_slack_rng, _ = seeding.np_random(self.seed)
      self.observation_space = Space([queue_size], np.dtype(int))
      self.action_space = Discrete(queue_size)
      for i in range(self.queue_size):
          self.init_queue.append(init_slack_rng.randint(self.max_slack))

      # Variables that can be reset.
      self.slack_rng, _ = seeding.np_random(self.seed)
      self.wrong_deques = [0] * self.past_steps
      self.wrong_deque_idx = 0
      self.queue = list(self.init_queue)
  
  def evaluate_action(self, action, deleted_packet):
      self.wrong_deques[self.wrong_deque_idx] = 0
      rewards = 0.0
      done = False
      correct = None

      # evaluate rewards depending on different queue types.
      if len(self.queue) > 0:
          if self.target_queue_type == "LSTF":
              min_slack = min(self.queue)
              correct = deleted_packet <= min_slack
          elif self.target_queue_type == "FIFO":
              correct = action == 0
          elif self.target_queue_type == "LIFO":
              correct = action == self.queue_size - 1
          else:
              print("Queue type " + self.target_queue_type + " not defined.")
      
      if correct is None:
          rewards = 0.0
      elif correct:
          rewards += 1.0
      else:
          self.wrong_deques[self.wrong_deque_idx] += 1
          rewards -= 1.0
      
      # If there are more than max_wrongs wrong deques in the previous past_steps steps,
      # then we end the episode.
      self.wrong_deque_idx = (self.wrong_deque_idx + 1) % self.past_steps
      if sum(self.wrong_deques) > self.max_wrongs:
          done = True

      return rewards, done


  # action is a set of indices. The indices point to the packets that should be removed from the queue
  # in this step.
  def step(self, action):
      # First deque packet from queue.
      deleted_packet = self.queue[action]
      del self.queue[action]

      # Evaluate the rewards of this action and whether this action
      # causes the episode to finish.
      rewards, done = self.evaluate_action(action, deleted_packet)

      # Enque one incoming packet.
      self.queue.append(self.slack_rng.randint(self.max_slack))

      # Decrement the slacks in the queue.
      for i in range(len(self.queue)):
          self.queue[i] -= 1

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
