import torch
import torch.nn as nn

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

    traced_resnet.save(f'{TargetPath}/ResNet/{name}')
    traced_polhead.save(f'{TargetPath}/PolHead/{name}')
    traced_valhead.save(f'{TargetPath}/ValHead/{name}')

def trainPolicy(dataloader, resNet, polNet, pol_loss, optimizer, device, logcount=5):
    size = len(dataloader.dataset)
    loginterval = len(dataloader) // logcount
    averagePolLoss = 0.0

    resNet.train()
    polNet.train()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        resNetOut = resNet(X)
        polPred = polNet(resNetOut)
        polLoss = pol_loss(polPred, Y)

        averagePolLoss += polLoss.detach().item()
        # Backpropagation
        polLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % loginterval == 0) and (batch > 0):
            logPolLoss = averagePolLoss / loginterval
            averagePolLoss = 0
            current = batch * len(X)
            print(f"Pol Loss: {logPolLoss:>8f} [{current:>5d}/{size:>5d}]")

def trainValue(dataloader, resNet, valNet, val_loss, optimizer, device, logcount=5):
    size = len(dataloader.dataset)
    loginterval = len(dataloader) // logcount
    averageValLoss = 0.0

    resNet.train()
    valNet.train()

    for param in resNet.parameters():
        param.requires_grad = False

    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        resNetOut = resNet(X)
        valPred = valNet(resNetOut)
        valLoss = val_loss(valPred, Y)

        averageValLoss += valLoss.detach().item()
        # Backpropagation
        valLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % loginterval == 0) and (batch > 0):
            logValLoss = averageValLoss / loginterval
            averageValLoss = 0
            current = batch * len(X)
            print(f"Val Loss: {logValLoss:>8f} [{current:>5d}/{size:>5d}]")

class Model:
    def __init__(self):
        self.resent = None
        self.polhead = None
        self.valhead = None
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.name = "unnamed"

    def create(self):

        self.resent = Resnet(Conf.NN_FILTERS, Conf.NN_RESNETLAYERS, Conf.HISTORYDEPTH + 1).to(self.device)
        self.polhead = PolicyHead(Conf.NN_FILTERS).to(self.device)
        self.valhead = ValueHead(Conf.NN_FILTERS, Conf.NN_LINFILTERS).to(self.device)

    def setName(self, name):
        self.name = name

    def load(self, name):
        self.resent, self.polhead, self.valhead = modelLoader(name)

    def loadScripted(self, name):
        self.resent = torch.jit.load(f'../../Models/scripted/ResNet/{name}')
        self.polhead = torch.jit.load(f'../../Models/scripted/PolHead/{name}')
        self.valhead = torch.jit.load(f'../../Models/scripted/ValHead/{name}')

    def saveScripted(self, name):
        scriptModel(self.resent, self.polhead, self.valhead, name)

    def getResnetParams(self):
        return self.resent.parameters()

    def getPolicyParams(self):
        return self.polhead.parameters()

    def getValueParams(self):
        return self.valhead.parameters()

    def train(self, dataset, learningRateResPol, learningRateVal, epochs = 1):
        print(f'Training Resnet and policy with: Torch: {torch.__version__} using {self.device} device')

        lossFunc = nn.CrossEntropyLoss()

        parameters = list(self.resent.parameters()) + list(self.polhead.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learningRateResPol)

        X = torch.from_numpy(dataset[0])
        YPol = torch.from_numpy(dataset[1])
        dataloader = Utils.toDataloader(X, YPol, Conf.BATCH_SIZE, True)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            trainPolicy(dataloader, self.resent, self.polhead, lossFunc, optimizer, self.device)

        print("Finished")

        print(f'Training Valuehead with: Torch: {torch.__version__} using {self.device} device')

        lossFunc = nn.MSELoss()

        parameters = list(self.valhead.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learningRateVal)

        X = torch.from_numpy(dataset[0])
        YVal = torch.from_numpy(dataset[2])
        dataloader = Utils.toDataloader(X, YVal, Conf.BATCH_SIZE, True)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            trainValue(dataloader, self.resent, self.valhead, lossFunc, optimizer, self.device)

        print("Finished")