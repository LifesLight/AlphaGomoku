import torch
from Utilities import Utilities as Utils
from NeuralNet import ResidualNetwork as Resnet
from NeuralNet import PolicyHead as PolicyHead
from NeuralNet import ValueHead as ValueHead
from Config import Config as Conf
import numpy as np

def modelLoader(model_name, model_path = Conf.MODEL_PATH):
    resnetPath = f'{model_path}/ResNet/{model_name}'
    policyPath = f'{model_path}/PolHead/{model_name}'
    valuePath = f'{model_path}/ValHead/{model_name}'

    resnetModel = Resnet(Conf.NN_FILTERS, Conf.NN_RESNETLAYERS, Conf.HISTORYDEPTH + 1)
    resnetModel.load_state_dict(torch.load(resnetPath, map_location=torch.device('cpu')))
    resnetModel.eval()
    resnetModel.to(Conf.DEVICE)

    policyModel = PolicyHead(Conf.NN_FILTERS)
    policyModel.load_state_dict(torch.load(policyPath, map_location=torch.device('cpu')))
    policyModel.eval()
    policyModel.to(Conf.DEVICE)

    valueModel = ValueHead(Conf.NN_FILTERS, Conf.NN_LINFILTERS)
    valueModel.load_state_dict(torch.load(valuePath, map_location=torch.device('cpu')))
    valueModel.eval()
    valueModel.to(Conf.DEVICE)

    return resnetModel, policyModel, valueModel

def scriptModel(resnet, policyhead, valuehead, name, TargetPath = '../../Models/scripted/'):
    resnetExample = np.zeros((1, Conf.HISTORYDEPTH + 1, 15, 15), dtype=np.float32) 
    resnetExample = torch.tensor(resnetExample)
    resnetExample = resnetExample.to(Conf.DEVICE)

    headsExample = np.zeros((1, Conf.NN_FILTERS, 15, 15), dtype=np.float32)
    headsExample = torch.tensor(headsExample)
    headsExample = headsExample.to(Conf.DEVICE)

    traced_resnet = torch.jit.trace(resnet, resnetExample)
    traced_polhead = torch.jit.trace(policyhead, headsExample)
    traced_valhead = torch.jit.trace(valuehead, headsExample)

    traced_resnet.save(f'{TargetPath}/ResNet/{name}.pt')
    traced_polhead.save(f'{TargetPath}/PolHead/{name}.pt')
    traced_valhead.save(f'{TargetPath}/ValHead/{name}.pt')

class Model:
    def __init__(self):
        self.resent = None
        self.polhead = None
        self.valhead = None
        self.name = "unnamed"

    def create(self):
        self.resent = Resnet()
        self.polhead = PolicyHead()
        self.valhead = ValueHead()

    def setName(self, name):
        self.name = name

    def load(self, path):
        self.resent, self.polhead, self.valhead = modelLoader(path)

    def saveScripted(self, path):
        scriptModel(self.resent, self.polhead, self.valhead, self.name, path)

    def trainResnetAndPolicyhead(self, dataset, epochs, batchsize):
        pass