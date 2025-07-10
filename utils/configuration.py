#configuration.py

import yaml
import os

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            # Look for config.yaml in the project root
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

# Example usage:
# config = ConfigLoader()
# NUM_EDGE_SERVERS = config.get('NUM_EDGE_SERVERS')
    
