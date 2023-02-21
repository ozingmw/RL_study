import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# env = gym.make('LunarLander-v2', render_mode='rgb_array')
env = make_vec_env('LunarLander-v2', n_envs=16)

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=1000000)
model_name = ".hf/model/ppo-LunarLander-v2"
model.save(model_name)