import gym
import random
import QueueEnv
from gym import wrappers, logger

class RandomAgent(object):
    """The world's simplest agent!"""
    def act(self, observation, reward):
        if len(observation) < 1:
            return -1
        a = random.randint(0, len(observation)-1)
        return a


env = QueueEnv.QueueEnv()
episode_count = 100
reward = 0
done = False
agent = RandomAgent()

for i in range(episode_count):
    ob = env.reset()
    survived_steps = 0
    while True:
        action = agent.act(ob, reward)
        ob, reward, done, _ = env.step(action)
        #env.render()
        if done:
            break
        survived_steps += 1
    print("Episode " + str(i) + " done. Survived " + str(survived_steps) + " steps.")

# Close the env and write monitor result info to disk
env.close()
