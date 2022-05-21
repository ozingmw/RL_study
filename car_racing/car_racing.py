from tf_agents.environments import suite_gym

from agent import Agent

env = suite_gym.load("CarRacing-v1")
obs = env.reset()

for _ in range(1000):
    env.render(mode="human")
    action = env.action_space.sample()
    
    # agent = Agent()
    # action = agent.policy()
    
    
    obs, reward, done, info = env.step(action)

env.close()