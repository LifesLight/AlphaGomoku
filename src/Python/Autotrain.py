import torch
import subprocess
import numpy as np
from Utilities import Utilities as Utils
from Config import Config as Conf


## Initialization Model
#BaselineModelName = "test1"
#BaselineModel = Utils.modelLoader(BaselineModelName, Type='Human')

command = '../../build/AlphaGomoku'
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

output = result.stdout
error = result.stderr

print("Output:")
print(output)

if error:
    print("Error:")
    print(error)