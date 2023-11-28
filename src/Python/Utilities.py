import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from NeuralNet import ResidualNetwork as Resnet
from NeuralNet import PolicyNetwork as PolHead
from NeuralNet import ValueNetwork as ValHead

from Config import Config as Conf

class Utilities:
    def sliceGamestate(gamestate, depth = 0):
        HD = gamestate.shape[0] - 1
        if depth > HD - 2:
            print("[Utilities] WARNING: Gamestate sliced to deep")

        output = ""
        halfDepth = HD // 2
        if HD == 2:
            blackStones = gamestate[1][:][:]
            whiteStones = gamestate[2][:][:]
        else:
            tempDepth = depth // 2

            whiteIndex = halfDepth * 2 - tempDepth  
            blackIndex = halfDepth - tempDepth

            if depth % 2 == 1:
                if gamestate[0][0][0]:
                    blackIndex -= 1
                else:
                    whiteIndex -= 1

            blackStones = gamestate[blackIndex][:][:]
            whiteStones = gamestate[whiteIndex][:][:]

        output += "   --------------------------------------------------------------\n"
        for y in range(14, -1 ,-1):
            output += f'{y:2} |'
            for x in range(15):
                if blackStones[x][y] == 0 and whiteStones[x][y] == 0:
                    output += "   "
                elif blackStones[x][y] == 1:
                    output += " B "
                elif whiteStones[x][y] == 1:
                    output += " W "
                output += "|"
            output += "\n   --------------------------------------------------------------\n"
        output += "     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14\n"

        return output

    def makeMove(gamestate, x, y):
        HD = gamestate.shape[0] - 1 
        turn = gamestate[0, 0, 0]
        gamestate[0, :, :] = not turn

        if turn:
            for i in range(HD//2 + 1, HD):
                gamestate[i] = gamestate[i + 1].copy()
            gamestate[HD][x][y] = True
        else:
            for i in range(1, HD//2):
                gamestate[i] = gamestate[i + 1].copy()
            gamestate[HD//2][x][y] = True

        return gamestate

    def indexToCords(index):
        x = index // 15
        y = index % 15
        return x, y

    def cordsToIndex(x, y):
        return x * 15 + y

    def modelLoader(model_path, model_name):

        resnetPath = f'{model_path}/ResNet/{model_name}'
        policyPath = f'{model_path}/PolHead/{model_name}'
        valuePath = f'{model_path}/ValHead/{model_name}'

        resnetModel = Resnet(Conf.NN_FILTERS, Conf.NN_RESNETLAYERS, Conf.HISTORYDEPTH + 1)
        resnetModel.load_state_dict(torch.load(resnetPath, map_location=torch.device('cpu')))
        resnetModel.eval()
        resnetModel.to(Conf.DEVICE)

        policyModel = PolHead(Conf.NN_FILTERS)
        policyModel.load_state_dict(torch.load(policyPath, map_location=torch.device('cpu')))
        policyModel.eval()
        policyModel.to(Conf.DEVICE)

        valueModel = ValHead(Conf.NN_FILTERS, Conf.NN_LINFILTERS)
        valueModel.load_state_dict(torch.load(valuePath, map_location=torch.device('cpu')))
        valueModel.eval()
        valueModel.to(Conf.DEVICE)

        return resnetModel, policyModel, valueModel

    def loadDataset(Path, Shape, SourceType):
        return torch.from_numpy(np.fromfile(Path, dtype=SourceType).astype(np.float32).reshape((Shape)))

    def toDataloader(X, Y, BatchSize=128, Shuffle=False):
        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=BatchSize, shuffle=Shuffle)

    # Change this to take a model class as input
    def trainableParameterCount(model):
        pp = 0
        for p in model.parameters():
            if p.requires_grad:
                nn = 1
                for s in p.size():
                    nn *= s
                pp += nn
        return pp

    def scriptModel(model, TargetPath = '../../Models/scripted/'):
        resnetExample = np.zeros((1, Conf.HISTORYDEPTH + 1, 15, 15), dtype=np.float32) 
        resnetExample = torch.tensor(resnetExample)
        resnetExample = resnetExample.to(Conf.DEVICE)

        headsExample = np.zeros((1, Conf.NN_FILTERS, 15, 15), dtype=np.float32)
        headsExample = torch.tensor(headsExample)
        headsExample = headsExample.to(Conf.DEVICE)

        traced_resnet = torch.jit.trace(model.resnet, resnetExample)
        traced_polhead = torch.jit.trace(model.policy, headsExample)
        traced_valhead = torch.jit.trace(model.value, headsExample)

        traced_resnet.save(f'{TargetPath}/ResNet/{model.name}.pt')
        traced_polhead.save(f'{TargetPath}/PolHead/{model.name}.pt')
        traced_valhead.save(f'{TargetPath}/ValHead/{model.name}.pt')

    def parseDatasetFromDatapoints(self, path, amount):
        lines = []
        with open(path, 'r') as f:
            lines = f.readlines()

        print(lines)