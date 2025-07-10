import pytest
import numpy as np
import torch
from agent.ddpg import ddpgModel, Buffer
from agent.td3 import td3Model
from agent.a2c import a2cModel

def test_ddpg_model_creation():
    model = ddpgModel(num_states=8, num_actions=4, std_dev=0.2, critic_lr=0.001, actor_lr=0.001, gamma=0.99, tau=0.005, activationFunction='tanh')
    assert model.actor_model is not None
    assert model.critic_model is not None
    dummy_state = torch.FloatTensor(np.random.rand(1, 8))
    action = model.policy(dummy_state)
    assert action.shape[-1] == 4

def test_td3_model_creation():
    model = td3Model(num_states=8, num_actions=4, std_dev=0.2, critic_lr=0.001, actor_lr=0.001, gamma=0.99, tau=0.005, activationFunction='tanh')
    assert model.actor_model is not None
    dummy_state = torch.FloatTensor(np.random.rand(1, 8))
    action = model.policy(dummy_state)
    assert action.shape[-1] == 4

def test_a2c_model_creation():
    model = a2cModel(num_states=8, num_actions=4, std_dev=0.2, critic_lr=0.001, actor_lr=0.001, gamma=0.99, tau=0.005, activationFunction='softmax')
    assert model.actor_model is not None
    assert model.critic_model is not None
    dummy_state = torch.FloatTensor(np.random.rand(1, 8))
    action_probs = model.policy(dummy_state)
    assert action_probs.shape[-1] == 4

def test_buffer_record_and_learn():
    model = ddpgModel(num_states=8, num_actions=2, std_dev=0.2, critic_lr=0.001, actor_lr=0.001, gamma=0.99, tau=0.005, activationFunction='tanh')
    buffer = Buffer(model, buffer_capacity=10, batch_size=2)
    for _ in range(5):
        s = np.random.rand(8)
        a = np.random.rand(2)
        r = np.random.rand(1)
        s2 = np.random.rand(8)
        buffer.record((s, a, r, s2))
    assert buffer.buffer_counter == 5
    # Should not raise error
    try:
        buffer.learn()
    except Exception as e:
        pytest.fail(f"Buffer learn failed: {e}") 