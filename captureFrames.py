import numpy as np
import os

framesPerSetting = 300
bitRates = ["1M", "2M", "5M", "10M", "100M"]

rootPath = '/home/sc/streamingPipeline'

# recompile the code 
os.system('clear && cd {:}/build && make'.format(rootPath))
os.system('mkdir {:}/analysisData/ref/'.format(rootPath))
# for frame in framesPerSetting:
os.system('cd {:}/build && ./reconstruction {:}'.format(rootPath, framesPerSetting))

for bitRate in bitRates:
    os.system('cd {:}/build && ./reconstruction {:}'.format(rootPath, framesPerSetting))
