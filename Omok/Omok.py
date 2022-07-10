import gym
import gym_omok.envs.__init__

env = gym.make("Omok-v0")

obs = env.reset()

print(obs)