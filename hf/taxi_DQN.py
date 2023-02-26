import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('Taxi-v3', n_envs=100)
env.reset()

model = DQN(
    'MlpPolicy',
    env,
    exploration_fraction=0.3,
    verbose=1
)

model.learn(total_timesteps=1000000, progress_bar=True)
model_name = "./hf/model/dqn-taxi-v3"
model.save(model_name)

eval_env = gym.make('Taxi-v3')
eval_env.reset()
mean_reward, std_reward = evaluate_policy(model, eval_env)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model = DQN.load(model_name)
env = gym.make('Taxi-v3')

obs = env.reset()

episode_reward = 0

while True:
    action = model.predict(obs)
    obs, reward, done, info = env.step(int(action[0]))
    episode_reward += reward
    if done:
        break

print(f'episode_reward: {episode_reward:.3f}')

env.close()