import torch
import subprocess
import numpy as np
from Utilities import Utilities as Utils
from Config import Config as Conf


initialModel = "test2.pt"
MODEL_PATH = "../../Models/Human"

loadedModel = Utils.modelLoader(MODEL_PATH, initialModel)