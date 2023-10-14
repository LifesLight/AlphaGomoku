import xml.etree.ElementTree as ET
import numpy as np
import os

# -----Generation Parameters------ #

DATASET_SOURCE = '../../Datasets/HumanExamples/RenjunetDatasets/renjunet_v10.xml'
HD = 8
AUGMENTED = True
# ['all'] for all
RULESETS_WHITELIST = [1]
RULESETS_BLACKLIST = []
TRAINSPLIT = 0.8
TARGETFOLDER = '../../Datasets/HumanExamples/GeneratedDatasets'

# -------------------------------- #


class DatasetGame:
    def __init__(self):
        self.ID = None
        self.Opening = None
        self.Ruleset = None
        self.Players = {"Black" : None, "White" : None}
        self.Winner = None
        self.Tourney = None
        self.Moves = []
        self.Augmented = False

class GameHandler:
    def __init__(self, Game):
        self.moves = Game.Moves
        self.index = 0
        if Game.Winner == "0":
            self.result = 1.0
        elif Game.Winner == "1":
            self.result = -1.0
        else:
            self.result = 0.0

    def onLastMove(self):
        return self.index >= len(self.moves) - 1
    
    def getMoveHistory(self):
        moveHistory = []
        for iterator, move in enumerate(self.moves[:self.index]):
            moveHistory.append((move, iterator % 2))
        return moveHistory

    def getNLastMoves(self, n):
        lastMoves = []
        for iteratorIndex in range(self.index - n - 1, self.index - 1):
            if iteratorIndex < 0:
                lastMoves.append((None, iteratorIndex % 2))
            else:
                lastMoves.append((self.moves[iteratorIndex], iteratorIndex % 2))
        return lastMoves
            
    def nextMove(self):
        self.index += 1
        self.result *= -1
        if self.index >= len(self.moves):
            return None
        return (self.moves[self.index], self.index % 2)

def formatRanges(numbers):
    ranges = []
    start = end = numbers[0]
    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = num
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ", ".join(ranges)

def MovesToGamestate(moves):
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

def gamestateFromBytes(gamestateBytes):
    return np.frombuffer(gamestateBytes, dtype=bool).reshape((HD + 1, 15, 15))

def MoveToString(move):
    move = move[0]
    return f'{move[0]},{move[1]}'

def StringToMove(s):
    x, y = map(int, s.split(','))
    return x, y

def MoveToGamestateTarget(move):
    target = np.zeros((15, 15), dtype=bool)
    target[move] = True
    target = target.flatten()
    return target

def OutcomeToGamestateTarget(outcome):
    target = np.zeros((1), dtype=np.int8)
    target[0] = outcome
    return target


RULESETS = []
if RULESETS_WHITELIST[0] == 'all':
    RULESETS = [i for i in range(1, 30)]
else:
    RULESETS = RULESETS_WHITELIST[:]
for blacklisted in RULESETS_BLACKLIST:
    if blacklisted in RULESETS:
        RULESETS.remove(blacklisted)
if len(RULESETS) == 0:
    raise Exception("No rulesets selected")

RULESETS_STR = formatRanges(sorted(RULESETS))
PATH = f'{TARGETFOLDER}/HD{HD},{"AUG," if AUGMENTED else ""}TS{TRAINSPLIT},RULESETS({RULESETS_STR})'

print(f'Generating Dataset with HD={HD}, Augmented={AUGMENTED}, TS={TRAINSPLIT}, Rulesets={RULESETS_STR}')

datasetGames = []
xmlTree = ET.parse(DATASET_SOURCE)
xmlRoot = xmlTree.getroot()
xmlGames = xmlRoot.find("./games")


for game in xmlGames:
    datasetGame = DatasetGame()
    datasetGame.ID = game.attrib.get("id", None)
    datasetGame.Opening = game.attrib.get("opening", None)
    datasetGame.Tourney = game.attrib.get("tournament", None)
    datasetGame.Ruleset = game.attrib.get("rule", None)
    datasetGame.Winner = game.attrib.get("bresult", None)
    datasetGame.Players["Black"] = game.attrib.get("black", None)
    datasetGame.Players["White"]  = game.attrib.get("white", None)
    
    if int(datasetGame.Ruleset) not in RULESETS:
        continue

    xmlMoves = game.find("./move").text
    if xmlMoves is not None:
        moveOrder = []
        for move in xmlMoves.split(" "):
            moveOrder.append((ord(move[0]) - 97, int(move[1:]) - 1))
        datasetGame.Moves = moveOrder
        datasetGames.append(datasetGame)
del xmlTree, xmlRoot, xmlGames

print(f'Extracted {len(datasetGames)} Games from: {DATASET_SOURCE} that match provided conditions')

if AUGMENTED:
    print("Augmenting Dataset...")
    augmentedDatasetGames = []
    for game in datasetGames:
        moves = game.Moves
    
        variants = [[] for i in range(5)]
        for move in moves:
            # Mirror Move X
            variants[0].append((14 - move[0], move[1]))
            # Mirror Move Y
            variants[1].append((move[0], 14 - move[1]))
            # 90 Degree Rotation
            variants[2].append((14 - move[1], move[0]))
            # 180 Degree Rotation
            variants[3].append((14 - move[0], 14 - move[1]))
            # 270 Degree Rotation
            variants[4].append((move[1], 14 - move[0]))
    
        augmentedDatasetGames.append(game)
        for i in range(5):
            newGame = DatasetGame()
            newGame.ID = game.ID
            newGame.Opening = game.Opening
            newGame.Tourney = game.Tourney
            newGame.Ruleset = game.Ruleset
            newGame.Winner = game.Winner
            newGame.Players = game.Players
            newGame.Augmented = True
            newGame.Moves = variants[i]
            augmentedDatasetGames.append(newGame)
    datasetGames = augmentedDatasetGames
    del augmentedDatasetGames
    
    print(f'Augmented Dataset to {len(datasetGames)} Games')

print("Generating all unique Gamestates...", end=" ")
gamestateToNextMoves = {}
gamestateToResults = {}
for game in datasetGames:
    gameHandler = GameHandler(game)
    
    while not gameHandler.onLastMove():
        isWinning = gameHandler.result
        nextMove = gameHandler.nextMove()
        nextMoveString = MoveToString(nextMove)
        lastMoves = gameHandler.getMoveHistory()
        gamestate = MovesToGamestate(lastMoves)
        gamestateBytes = gamestate.tobytes()
        
        if gamestateBytes in gamestateToNextMoves.keys():
            gamestateToNextMoves[gamestateBytes].append(nextMoveString)
            gamestateToResults[gamestateBytes].append(isWinning)
        else:
            gamestateToNextMoves[gamestateBytes] = [nextMoveString]
            gamestateToResults[gamestateBytes] = [isWinning]

print(f'({len(gamestateToNextMoves.items())})')

print("Preparing Gamestates...")
gamestateToMoveCountDict = {}
for key, values in gamestateToNextMoves.items():
    gamestateToMoveCountDict[key] = {}
    

    for value in values:
        if value in gamestateToMoveCountDict[key].keys():
            gamestateToMoveCountDict[key][value] += 1
        else:
            gamestateToMoveCountDict[key][value] = 1


gamestateToResultCountDict = {} 
for key, values in gamestateToResults.items():
    gamestateToResultCountDict[key] = {} 
    for value in values:
        if value in gamestateToResultCountDict[key].keys():
            gamestateToResultCountDict[key][value] += 1
        else:
            gamestateToResultCountDict[key][value] = 1

del gamestateToNextMoves
del gamestateToResults

print("Calculating best move for each Gamestate...")
gamestateToNextMove = {}
for key, value in gamestateToMoveCountDict.items():
    mostPlayedMove = max(value, key=lambda k: value[k])
    gamestateToNextMove[key] = mostPlayedMove
del gamestateToMoveCountDict

print("Calculating most likely outcome for each Gamestate...")
gamestateToResult = {}
for key, value in gamestateToResultCountDict.items():
    mostLikelyOutcome = max(value, key=lambda k: value[k])
    gamestateToResult[key] = mostLikelyOutcome
del gamestateToResultCountDict

print("Formating Gamestates for export...")
finishedDataset = []
for key, value in gamestateToNextMove.items():
    x = gamestateFromBytes(key)
    nextMove = StringToMove(value)
    yPolicy = MoveToGamestateTarget(nextMove)
    yValue = OutcomeToGamestateTarget(gamestateToResult[key])
    finishedDataset.append((x, yPolicy, yValue))
del gamestateToNextMove
del gamestateToResult

trainSizePercentile = TRAINSPLIT
outputPath = PATH
os.makedirs(outputPath, exist_ok=True)

np.random.shuffle(finishedDataset)
index = int(trainSizePercentile * len(finishedDataset))

print(f'Writing {len(finishedDataset)} datapoints to disk (Train:{index}|Test:{len(finishedDataset) - index})...')

np.array([x[0] for x in finishedDataset[:index]], dtype=bool).tofile(f'{outputPath}/XTrain.bin')
np.array([x[1] for x in finishedDataset[:index]], dtype=bool).tofile(f'{outputPath}/YTrainPol.bin')
np.array([x[2] for x in finishedDataset[:index]], dtype=np.int8).tofile(f'{outputPath}/YTrainVal.bin')
np.array([x[0] for x in finishedDataset[index:]], dtype=bool).tofile(f'{outputPath}/XTest.bin')
np.array([x[1] for x in finishedDataset[index:]], dtype=bool).tofile(f'{outputPath}/YTestPol.bin')
np.array([x[2] for x in finishedDataset[index:]], dtype=np.int8).tofile(f'{outputPath}/YTestVal.bin')

print(f'Saved Dataset to {outputPath}')