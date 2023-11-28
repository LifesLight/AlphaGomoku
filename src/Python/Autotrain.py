import torch
import subprocess
import numpy as np
from Utilities import Utilities as Utils
from Config import Config as Conf
from Model import Model
import sys
import random as rand

DATAPOINT_PATH = "../../Datasets/Selfplay/data.txt"

class Datapoint:
    def __init__(self, datapointString):
        rawMoves = []
        # Parse the string
        i = 0
        # Parse moves
        splittedComma = datapointString.split(",")
        rawMoves = splittedComma[:-1]
        splittedSemicolon = splittedComma[-1].split(";")
        rawMoves.append(splittedSemicolon[0])
        self.bestMove = splittedSemicolon[1]
        self.result = splittedSemicolon[2]

        self.moves = []
        for i, move in enumerate(rawMoves):
            # Convert string to int
            if (move == ""):
                continue
            intMove = int(move)
            x, y = Utils.indexToCords(intMove)
            self.moves.append(((x, y), i % 2))


    def getGamestate(self):
        moves = self.moves

        if (len(moves) == 0):
            return np.zeros((Conf.HISTORYDEPTH + 1, 15, 15), dtype=bool)

        HD = Conf.HISTORYDEPTH
        halfHistory = HD // 2
        numpyGamestate = np.zeros((HD + 1, 15, 15), dtype=bool)
        toMoveColor = not moves[-1][1]
        numpyGamestate[0, :, :] = toMoveColor
        generalMoves = moves[:max(len(moves) - HD, 0)]
        slicedMoves = moves[max(len(moves) - HD, 0):]
        for move, color in generalMoves:
            if color:
                for i in range(halfHistory + 1, halfHistory * 2 + 1):
                    numpyGamestate[i][move] = True
            else:
                for i in range(1, halfHistory + 1):
                    numpyGamestate[i][move] = True

        for iterator, (move, color) in enumerate(slicedMoves):
            localIterator = iterator // 2
            if color:
                for i in range(localIterator + 1):
                    index = HD - i
                    numpyGamestate[index][move] = True
            else:
                for i in range(localIterator + 1):
                    index = halfHistory - i
                    numpyGamestate[index][move] = True

        return numpyGamestate


    def getResult(self):
        return self.result

    def getBestMove(self):
        return self.bestMove

class DatapointSampler:
    def __init__(self, path):
        file = open(path, 'r')
        self.datapoints = []
        while True:
            line = file.readline()
            if not line:
                break
            self.datapoints.append(line[:-1])
        rand.shuffle(self.datapoints)
        file.close()
        self.itterator = 0

    def sample(self):
        x = Datapoint(self.datapoints[self.itterator])

        self.itterator += 1
        if (self.itterator >= len(self.datapoints)):
            self.itterator = 0
            rand.shuffle(self.datapoints)
            print("Reshuffled data")
        return x

def makeGamestatesToDataset(gamestates, results, bestMoves):
    X = np.zeros((len(gamestates), Conf.HISTORYDEPTH + 1, 15, 15), dtype=np.float32)
    YPol = np.zeros((len(gamestates), 225), dtype=np.float32)
    YVal = np.zeros((len(gamestates), 1), dtype=np.float32)
    for i, gamestate in enumerate(gamestates):
        X[i] = gamestate
        bestMove = bestMoves[i]
        x, y = Utils.indexToCords(int(bestMove))
        YPol[i][Utils.cordsToIndex(x, y)] = 1.0
        YVal[i] = float(results[i])
    return X, YPol, YVal

def makeDataset(sampleCount):
    sampler = DatapointSampler(DATAPOINT_PATH)
    gamestates = []
    results = []
    bestMoves = []
    for i in range(sampleCount):
        sample = sampler.sample()
        gamestates.append(sample.getGamestate())
        results.append(sample.getResult())
        bestMoves.append(sample.getBestMove())
    return makeGamestatesToDataset(gamestates, results, bestMoves)

# Get commandline params
argv = sys.argv

SourceModelName = 'test2.pt'

TargetDatapoints = 107690
dataset = makeDataset(TargetDatapoints)
learningRate = 1e-6

model = Model()
model.loadScripted(SourceModelName)
model.train(dataset, learningRate, learningRate)

model.saveScripted('test4.pt')