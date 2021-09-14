from collections import deque
from typing import Tuple
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import random
import numpy as np

from abc import ABC, abstractmethod


class BaseQLearningAgent(ABC):
    @abstractmethod
    def remember(self, state: any, action: int, reward: float, next_state: any, done: bool, *args, **kwargs) -> None:
        """remember method. 
        Example: Basic Q-Learning only remembers (s, a, r, s) 
        whereas DQN remembers full history of (s, a, r, s)

        Args:
            state (any): state return from the environment
            action (int): action take based on state
            reward (float): reward for action taken
            next_state (any): state after performing action
            done (bool): denote whether next state is terminal state
        """
        pass
    
    @abstractmethod
    def update(self) -> None:
        """how to update the agent.
        Example: using gradient descent or just TD methods
        """
        pass
    
    @abstractmethod
    def state_action_value(self, state: any) -> np.ndarray:
        """Calculate all Q-value for a state

        Args:
            state (any): state from environment

        Returns:
            np.ndarray: a vector of shape (action, 1) containing all estimated Q(s, a)
        """
        pass
    
    @abstractmethod
    def get_best_action(self, state: any) -> int:
        """get best action based on the estimated Q vector

        Args:
            state (any): state from environment

        Returns:
            int: best action to take
        """
        pass
    
    def select_action(self, state: any) -> int:
        """select action in e-greedy fashion

        Args:
            state (any): state from environment

        Returns:
            int: action to take based on e-greedy fashion
        """
        random_number = np.random.uniform()
        if random_number >= self.epsilon:
            return self.get_best_action(state)
        return np.random.randint(0, self.num_actions)
    
    def save(self, model_name: str) -> None:
        """save agent

        Args:
            model_name (str): where to save the agent
        """
        self.model.save(model_name)
        
    def load(self, model_dir: str) -> None:
        """load agent

        Args:
            model_dir (str): where to load the agent
        """
        self.model = load_model(model_dir)


class FeedForwardDQNAgent(BaseQLearningAgent):
    def __init__(self, input_shape: Tuple, num_actions: int, 
                 gamma=1, lr=0.001, 
                 max_buffer_size=2000, batch_size=32,
                 target_update_freq=10,
                 use_replay=True) -> None:
        """constructor

        Args:
            input_shape (Tuple): the shape of input (should be a 1D vector)
            num_actions (int): size of action space
            gamma (float, optional): discounted factor. Defaults to 1.
            lr (float, optional): learning rate. Defaults to 0.001.
            max_buffer_size (int, optional): maximum capacity of replay buffer. Defaults to 2000.
            batch_size (int, optional): number of samples from replay buffer. Defaults to 32.
            target_update_freq (int, optional): frequency of updating target network. Defaults to 10.
            use_replay (bool, optional): whether agent should use replay buffer or not. Defaults to True.
        """
        
        super().__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.use_replay = use_replay
        
        if self.use_replay:
            self.max_buffer_size = max_buffer_size
            self.batch_size = batch_size
            self.replay_buffer = deque(maxlen=self.max_buffer_size)
        
        # exploration/exploitation control using decay factor
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        
        self.target_update_f = target_update_freq
        
        # two models with same weight
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.time = 0
        
    def remember(self, state, action, reward, next_state, done):
        if self.use_replay:
            # if use replay buffer, store it to replay buffer and update e
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) > self.batch_size:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                
        else:
            # otherwise, only remember the latest sars
            self.state = state
            self.action = action
            self.next_state = next_state
            self.done = done
            self.reward = reward
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.num_actions))
        model.summary()
        
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.lr))
        return model
    
    def state_action_value(self, state):
        return self.model.predict(state)
    
    def update(self):
        self.time += 1 # keep tracks of how many episodes have passed
        
        if self.use_replay and len(self.replay_buffer) >= self.batch_size:
            # sample whenever replay buffer size is higher than or equal to batch size
            minibatch_data = random.sample(self.replay_buffer, self.batch_size)
            states = np.zeros((self.batch_size, self.input_shape[0]))
            next_states = np.zeros((self.batch_size, self.input_shape[0]))
            actions = []
            rewards = []
            dones = []
                
            for i in range(self.batch_size):
                state, action, reward, next_state, done = minibatch_data[i]
                states[i] = state.flatten()
                next_states[i] = next_state.flatten()
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                    
            target = self.model.predict(states)
            target_next = self.target_model.predict(next_states)
            
            # fixed target
            for i in range(self.batch_size):
                if dones[i]:
                    target[i, actions[i]] = rewards[i]
                else:
                    target[i, actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
                    
            # update model
            self.model.fit(states, target, batch_size=self.batch_size, verbose=0)
        
        elif self.use_replay and len(self.replay_buffer) < self.batch_size:
            pass
        
        else:
            target = self.model.predict(self.state)
            target_next = self.target_model.predict(self.next_state)
            
            if self.done:
                target[0, self.action] = self.reward
            
            else:
                target[0, self.action] = self.reward + self.gamma * np.amax(target_next[0])   
            self.model.fit(self.state, target, verbose=0)
        
        # update target network after a set period of time
        if self.time % self.target_update_f == 0:
            self.target_model.set_weights(self.model.get_weights())
                
    def get_best_action(self, state):
        return np.argmax(self.model.predict(state).flatten())
