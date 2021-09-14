import gym
from agent import FeedForwardDQNAgent
from utils import train_cart_pole


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
    target_update_freq=target_update_f,
    use_replay=use_replay
)

print("TRAIN DQN W FIXED TARGET AND REPLAY...")
model_dir = "models/cart_pole_v1_fix.h5"
result_dir = "results/result_fix.csv"

train_cart_pole(agent, env, result_dir, model_dir, max_episode=2000)
