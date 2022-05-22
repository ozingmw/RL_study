import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import suite_gym
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function

import PIL, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

from agent import Agent

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


env = suite_gym.load("CarRacing-v1")
env = TFPyEnvironment(env)
obs = env.reset()

print(env.action_spec())

conv_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
fc_layer_params=[512]

q_net = QNetwork(
    env.observation_spec(),
    env.action_spec(),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
)

train_step = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=250000 // update_period,
    end_learning_rate=0.01,
)

agent = DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000,
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.95,
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step)
)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=750000,
)
replay_buffer_observer = replay_buffer.add_batch

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
logging.get_logger().set_level(logging.INFO)

collect_driver = DynamicStepDriver(
    env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period,
)

initial_collect_policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())
init_driver = DynamicStepDriver(
    env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=20000,
)
final_time_step, final_policy_state = init_driver.run()

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print(f"\r{iteration} loss:{train_loss.loss.numpy():.5f}", end="")
        if iteration % 10000 == 0:
            log_metrics(train_metrics)

train_agent(50000)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(env.pyenv.envs[0].render(mode="rgb_array"))

watch_driver = DynamicStepDriver(
    env,
    agent.policy,
    observers=[save_frames],
    num_steps=10000
)
final_time_step, final_policy_state = watch_driver.run()

image_path = os.path.join("images", "car_racing.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]
frame_images[0].save(image_path, format='GIF', append_images=frame_images[1:], save_all=True, duration=30, loop=0)

plot_animation(frames)