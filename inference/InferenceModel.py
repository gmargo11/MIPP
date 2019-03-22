from abc import ABC, abstractmethod

class InferenceModel(ABC):
    @abstractmethod
    def load_environment(self, env):
        pass

    @abstractmethod
    def infer_joint_distribution(self, res):
        pass

    @abstractmethod
    def infer_independent_distribution(self, res):
        pass

    @abstractmethod
    def update(self, x, y):
        pass

    @abstractmethod
    def display(self):
        pass