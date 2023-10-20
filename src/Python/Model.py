import torch
from Utilities import Utilities as Utils
from Config import Config as Conf


class Model:
    def __init__(self, ModelName):
        self.name = ModelName
        self.resnet, 
        self.policy, 
        self.value = Utils.modelLoader(ModelName, Conf.NN_FILTERS, Conf.NN_RESNETLAYERS, Conf.HISTORYDEPTH, Conf.DEVICE)