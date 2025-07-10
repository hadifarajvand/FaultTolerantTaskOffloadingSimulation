
#ddpg.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent import BaseAgent

# OUActionNoise remains unchanged
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

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

class ddpgModel(BaseAgent):
    """
    Deep Deterministic Policy Gradient (DDPG) model for continuous control.
    Contains actor and critic networks, target networks, and noise process.
    """
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

        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1), theta=0.2)

        self.actor_model = Actor(num_states, num_actions, activationFunction).to(self.device)
        self.critic_model = Critic(num_states, num_actions).to(self.device)
        self.target_actor = Actor(num_states, num_actions, activationFunction).to(self.device)
        self.target_critic = Critic(num_states, num_actions).to(self.device)

        self.target_actor.load_state_dict(self.actor_model.state_dict())
        self.target_critic.load_state_dict(self.critic_model.state_dict())

        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)

        self.loss_fn = nn.MSELoss()

    def update_target(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def addNoise(self, sampled_actions, thisEpNo, totalEpNo):
        noise = self.ou_noise()
        noisy_actions = []
        for sa in sampled_actions:
            sa = sa + noise[0]
            sa = np.clip(sa, -1, 1)
            noisy_actions.append(sa)
        return noisy_actions

    def policy(self, state):
        self.actor_model.eval()
        with torch.no_grad():
            sampled_actions = self.actor_model(state.to(self.device))
        self.actor_model.train()
        return sampled_actions.squeeze()

    def model_summary(self):
        return str(self.actor_model) + '\n' + str(self.critic_model)

class Buffer:
    def __init__(self, ddpgObj, buffer_capacity=100000, batch_size=256):
        self.ddpgObj = ddpgObj
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        device = self.ddpgObj.device
        # Compute target actions
        with torch.no_grad():
            target_actions = self.ddpgObj.target_actor(next_state_batch)
            y = reward_batch + self.ddpgObj.gamma * self.ddpgObj.target_critic(next_state_batch, target_actions)
        # Critic update
        critic_value = self.ddpgObj.critic_model(state_batch, action_batch)
        critic_loss = self.ddpgObj.loss_fn(critic_value, y)
        self.ddpgObj.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.ddpgObj.critic_optimizer.step()
        # Actor update
        actions = self.ddpgObj.actor_model(state_batch)
        actor_loss = -self.ddpgObj.critic_model(state_batch, actions).mean()
        self.ddpgObj.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.ddpgObj.actor_optimizer.step()

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if record_range < self.batch_size:
            return
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = torch.FloatTensor(self.state_buffer[batch_indices]).to(self.ddpgObj.device)
        action_batch = torch.FloatTensor(self.action_buffer[batch_indices]).to(self.ddpgObj.device)
        reward_batch = torch.FloatTensor(self.reward_buffer[batch_indices]).to(self.ddpgObj.device)
        next_state_batch = torch.FloatTensor(self.next_state_buffer[batch_indices]).to(self.ddpgObj.device)
        self.update(state_batch, action_batch, reward_batch, next_state_batch)
