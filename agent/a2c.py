from agent import BaseAgent
from agent.ddpg import Buffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, activationFunction):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.activationFunction = activationFunction
        if activationFunction == 'tanh':
            self.final_activation = torch.tanh
        elif activationFunction == 'softmax':
            self.final_activation = nn.Softmax(dim=-1)
        else:
            self.final_activation = lambda x: x

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.final_activation(x)
        return x

class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class a2cModel(BaseAgent):
    def __init__(self, num_states, num_actions, std_dev, critic_lr, actor_lr, gamma, tau, activationFunction):
        self.num_states = num_states
        self.num_actions = num_actions
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_model = Actor(num_states, num_actions, activationFunction).to(self.device)
        self.critic_model = Critic(num_states).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.loss_fn = nn.MSELoss()

    def policy(self, state):
        self.actor_model.eval()
        with torch.no_grad():
            action_probs = self.actor_model(state.to(self.device))
        self.actor_model.train()
        return action_probs.squeeze()

    def addNoise(self, sampled_actions, thisEpNo, totalEpNo):
        # For A2C, usually no noise is added, but for compatibility:
        return sampled_actions

    def update_target(self, target_net, source_net):
        # No target network in vanilla A2C
        pass

    def model_summary(self):
        return str(self.actor_model) + '\n' + str(self.critic_model) 