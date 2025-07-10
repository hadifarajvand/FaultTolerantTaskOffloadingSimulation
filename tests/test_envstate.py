import pytest
from env.EnvState import EnvironmentState

class DummyServer:
    def __init__(self, server_id, processing_frequency=10, failure_rate=0.01):
        self.server_id = server_id
        self.processing_frequency = processing_frequency
        self.failure_rate = failure_rate
        self.queue = []

class DummyTask:
    def __init__(self, id, computation_demand=5, task_size=10):
        self.id = id
        self.computation_demand = computation_demand
        self.task_size = task_size


def test_add_server_and_reset():
    env = EnvironmentState()
    server = DummyServer(server_id=1)
    env.add_server_and_init_environment(server)
    assert 1 in env.servers
    env.reset()
    assert env.servers == {}
    assert env.tasks == {}
    assert env.num_completed_tasks == 0

def test_add_and_remove_task():
    env = EnvironmentState()
    task = DummyTask(id=42)
    env.add_task(task)
    assert 42 in env.tasks
    env.remove_task(42)
    assert 42 not in env.tasks

def test_get_state_shape():
    env = EnvironmentState()
    for i in range(3):
        env.add_server_and_init_environment(DummyServer(server_id=i, processing_frequency=10+i, failure_rate=0.01*(i+1)))
    for i in range(5):
        env.add_task(DummyTask(id=i, computation_demand=5+i, task_size=10+i))
    state = env.get_state()
    assert isinstance(state, (list, tuple)) or hasattr(state, 'shape') 