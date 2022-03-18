"""Module implements the abstract base deeplearning model"""
from abc import ABC, abstractmethod

import numpy as np


class Baseactivate(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def activate(self):
        pass


class Basecost(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute(self):
        pass


class Baseinitialize(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def initiate(self):
        pass


class Baseparams(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self):
        pass


class Basepropagate(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def propagate(self):
        pass


class Basepredict(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self):
        pass


class Baseoptimize(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def optimize(self):
        pass


class Baselayer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def determine(self):
        pass


class Basemodel(ABC):
    def __init__(self):
        self.parameters = None
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
