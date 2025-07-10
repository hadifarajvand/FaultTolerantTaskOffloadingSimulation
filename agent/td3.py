from agent import BaseAgent
from agent.ddpg import Buffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, activationFunction):
        super(Actor, self).__init__()
        self.ln = nn.LayerNorm(num_states)
        self.fc1 = nn.Linear(num_states, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, num_actions)
        self.activationFunction = activationFunction
        if activationFunction == 'tanh':
            self.final_activation = torch.tanh
        elif activationFunction == 'softmax':
            self.final_activation = nn.Softmax(dim=-1)
        else:
            self.final_activation = lambda x: x
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, x):
        x = self.ln(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.final_activation(x)
        return x

class Critic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Critic, self).__init__()
        self.state_fc1 = nn.Linear(num_states, 300)
        self.state_fc2 = nn.Linear(300, 200)
        self.action_fc = nn.Linear(num_actions, 200)
        self.ln = nn.LayerNorm(200 + 200)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, state, action):
        s = torch.relu(self.state_fc1(state))
        s = torch.relu(self.state_fc2(s))
        a = torch.relu(self.action_fc(action))
        x = torch.cat([s, a], dim=-1)
        x = self.ln(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class td3Model(BaseAgent):
    def __init__(self, num_states, num_actions, std_dev, critic_lr, actor_lr, gamma, tau, activationFunction):
        self.activationFunction = activationFunction
        self.num_states = num_states
        self.num_actions = num_actions
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_model = Actor(num_states, num_actions, activationFunction).to(self.device)
        self.actor_target = Actor(num_states, num_actions, activationFunction).to(self.device)
        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)

        self.critic_1 = Critic(num_states, num_actions).to(self.device)
        self.critic_2 = Critic(num_states, num_actions).to(self.device)
        self.critic_target_1 = Critic(num_states, num_actions).to(self.device)
        self.critic_target_2 = Critic(num_states, num_actions).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        self.loss_fn = nn.MSELoss()
        self.policy_delay = 2  # Delayed policy updates
        self.learn_step = 0

    def policy(self, state):
        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(state.to(self.device))
        self.actor_model.train()
        return action.squeeze()

    def addNoise(self, sampled_actions, thisEpNo, totalEpNo):
        noise = np.random.normal(0, self.std_dev, size=np.shape(sampled_actions))
        noisy_actions = sampled_actions + noise
        noisy_actions = np.clip(noisy_actions, -1, 1)
        return noisy_actions.tolist()

    def update_target(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def model_summary(self):
        return str(self.actor_model) + '\n' + str(self.critic_1) + '\n' + str(self.critic_2)

    # Optionally, you can add a learn() method for TD3 here, but Buffer will call update_target as needed. 