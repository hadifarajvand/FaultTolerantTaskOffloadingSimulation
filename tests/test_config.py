from utils.configuration import ConfigLoader

def test_config_loader_basic():
    config = ConfigLoader()
    assert config.get('NUM_EDGE_SERVERS') == 6
    assert config.get('SCENARIO_TYPE') in ['homogeneous', 'heterogeneous']
    assert isinstance(config.get('TASK_SIZE_RANGE'), list) 