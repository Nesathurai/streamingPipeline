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
refPath = '/home/sc/streamingPipeline/analysisData/ref'

CHANGE_THRESHOLD = 10
INVALIDATION_THRESHOLD = 2


# delete existing csv file
os.system("rm /home/sc/streamingPipeline/analysisData/temporal-rvl-data/result.csv")

run = 0
os.system('''export DISPLAY=":0.0"''')
os.system('clear && cd {:}/build && make'.format(rootPath))

# for frame in range(framesPerSetting):

# meshIn = "{:}/analysisData/frame_{:}_camera_{:}_vx_{:.5f}.obj".format(rootPath, frame, camera, float(voxelSize))
# overwrite file and include column names

cmd = 'cd {:}/build && ./trvl {:}/ allDepthBin {:} {:} 1 0'.format(rootPath, refPath, CHANGE_THRESHOLD, INVALIDATION_THRESHOLD)
print(cmd)
os.system(cmd)
cmd = 'cd {:}/build && ./trvl {:}/ allDepthBin {:} {:} 0 1'.format(rootPath, refPath, CHANGE_THRESHOLD, INVALIDATION_THRESHOLD)
print(cmd)
os.system(cmd)
