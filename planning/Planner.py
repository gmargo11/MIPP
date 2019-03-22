from abc import ABC, abstractmethod

class Planner(ABC):
    @abstractmethod
    def policy(self, alpha, inference_model):
        pass