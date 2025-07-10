from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def policy(self, state):
        pass

    @abstractmethod
    def addNoise(self, sampled_actions, thisEpNo, totalEpNo):
        pass

    @abstractmethod
    def update_target(self, target_net, source_net):
        pass

    @abstractmethod
    def model_summary(self):
        pass
