import numpy as np
import os

myfile = open("settings.txt", "r")
framesPerSetting = int(myfile.readline().split("=")[-1].strip())
# print(framesPerSetting)
voxelSizes = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(voxelSizes)
bitRates = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(bitRates)
compressRates = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(compressRate)
myfile.close()

rootPath = '/home/sc/streamingPipeline'

os.system('clear && cd {:}/build && make'.format(rootPath))

os.system('cd {:}/build && ./mergeBin {:}'.format(rootPath,framesPerSetting))