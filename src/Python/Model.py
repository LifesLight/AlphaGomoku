import torch
from Utilities import Utilities as Utils
from NeuralNet import ResidualNetwork as Resent
from NeuralNet import PolicyHead as PolicyHead
from NeuralNet import ValueHead as ValueHead

class Model:
    def __init__(self):
        self.resent = None
        self.polhead = None
        self.valhead = None

    def create(self):
        self.resent = Resent()
        self.polhead = PolicyHead()
        self.valhead = ValueHead()