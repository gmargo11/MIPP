from abc import ABC, abstractmethod

class Environment(ABC):
    @abstractmethod
    def load_prior_data(self):
        pass
    @abstractmethod
    def observe(self, x, feature):
        pass