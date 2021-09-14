from utils import test
import gym
from agent import FeedForwardDQNAgent


learning_rate=0.0001
batch_size=32
max_buffer_size=2000
gamma=1
target_update_f = 100
use_replay=True

env = gym.make("CartPole-v1")

agent = FeedForwardDQNAgent(
    input_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    gamma=gamma,
    lr=learning_rate,
    max_buffer_size=max_buffer_size,
    batch_size=batch_size,
    use_replay=use_replay,
    target_update_freq=target_update_f
)

model_dir = "models/cart_pole_v1_and_replay.h5"

agent.load(model_dir)

test(agent, env, record_dir="DQN_fixed_and_replay", record=True)
