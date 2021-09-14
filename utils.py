from agent import BaseQLearningAgent
import numpy as np
import gym

def train_cart_pole(agent: BaseQLearningAgent, env, result_dir: str, model_dir: str, max_episode=10000):
    """train cart pole with an agent

    Args:
        agent (BaseQLearningAgent): a Q-Learning agent
        env (any): environment from gym
        result_dir (str): where to save the result
        model_dir (str): where to save the agent
        max_episode (int, optional): maximum episode to train. Defaults to 10000.
    """
    rewards = []
    max_reward = None
    for i in range(max_episode):
        observation = env.reset()
        observation = observation.reshape(1, -1)
        total_reward = 0
        
        done = False
        j = 0
        while not done:
            env.render()
            action = agent.select_action(observation)
            next_observation, reward, done, _ = env.step(action)
            next_observation = next_observation.reshape(1, -1)
            total_reward += reward
            if done and j < env._max_episode_steps:
                reward = -100
            agent.remember(observation, action, reward, next_observation, done)
            agent.update()
            observation = next_observation
            j += 1
        rewards.append(total_reward)
        if max_reward is None or np.mean(rewards) > max_reward:
            agent.save(model_dir)
            max_reward = np.mean(rewards)
            
        print("Episode {} completed, total reward: {}, eps: {}".format(i+1, total_reward, agent.epsilon))
    
        with open(result_dir, "w") as f:
            for reward in rewards:
                f.write(f"{reward}\n")
                
def test(agent: BaseQLearningAgent, env, record_dir="DQN", record=False):
    """test an agent on an environment

    Args:
        agent (BaseQLearningAgent): an Q-Learning agent to be tested
        env (any): environment from gym
        record_dir (str, optional): where to save the record of agent's performance. Defaults to "DQN".
        record (bool, optional): whether to record the performance. Defaults to False.
    """
    done = False
    total_reward = 0
    if record:
        env = gym.wrappers.Monitor(env, record_dir, force=True)
    observation = env.reset()
    observation = observation.reshape(1, -1)
    while not done:
        env.render()
        action = agent.get_best_action(observation)
        observation, reward, done, _  = env.step(action)
        observation = observation.reshape(1, -1)
        total_reward += reward
    env.close()
    print(total_reward)