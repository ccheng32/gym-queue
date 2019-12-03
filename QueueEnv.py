import gym
from gym import error, spaces, utils
from gym.utils import seeding


class QueueEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, max_slack=float(1e9), in_rate=4, out_rate=3, max_wrongs=3, past_steps=10, seed=None):
      # Fixed variables.
      self.max_slack = max_slack
      self.in_rate = in_rate
      self.out_rate = out_rate
      self.past_steps = past_steps
      self.max_wrongs = max_wrongs
      self.seed = seed

      # Variables that can be reset.
      self.slack_rng, _ = seeding.np_random(self.seed)
      self.queue = list()
      self.wrong_deques = [0] * self.past_steps
      self.wrong_deque_idx = 0

  # action is a set of indices. The indices point to the packets that should be removed from the queue
  # in this step.
  def step(self, action):
      rewards = 0
      done = False

      # First deque packets from queue.
      deleted_packets = list()
      sorted_action = list(action)
      sorted_action.sort(reverse=True)
      for act in sorted_action:
          deleted_packets.append(self.queue[act])
          del self.queue[act]

      # Check if the deleted packets matches LSTF policy.
      self.wrong_deques[self.wrong_deque_idx] = 0
      if len(self.queue) > 0:
          min_slack = min(self.queue)
          for s in deleted_packets:
              if s <= min_slack:
                  rewards += 1
              else:
                  self.wrong_deques[self.wrong_deque_idx] += 1
                  rewards -= 1
      
      # If there are more than max_wrongs wrong deques in the previous past_steps steps,
      # then we end the episode.
      self.wrong_deque_idx = (self.wrong_deque_idx + 1) % self.past_steps
      if sum(self.wrong_deques) > self.max_wrongs:
          done = True

      # Enque the incoming packets.
      for i in range(self.in_rate):
          self.queue.append(self.max_slack * self.slack_rng.random_sample())

      # Decrement the slacks in the queue.
      for i in range(len(self.queue)):
          self.queue[i] -= 1.0

      # Observation is (# of packets to deque, the queue itself)
      obs = (min(len(self.queue), self.out_rate), list(self.queue))

      return obs, rewards, done, None

  def reset(self):
      self.slack_rng, _ = seeding.np_random(self.seed)
      self.queue = list()
      self.wrong_deques = [0] * self.past_steps
      self.wrong_deque_idx = 0
      obs = (min(len(self.queue), self.out_rate), list(self.queue))
      return obs

  def render(self, mode='human', close=False):
      print(self.queue)
