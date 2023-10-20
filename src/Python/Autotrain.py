import torch
import numpy as np
from Utilities import Utilities as Utils
from Config import Config as Conf


# Initialization Model
BaselineModelName = "test1"
BaselineModel = Utils.modelLoader(BaselineModelName, Type='Human')

