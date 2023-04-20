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
trvlParams = myfile.readline().split("=")[-1].split("|")

# my_list = list(map(str.strip, value.split(',')))
CHANGE_THRESHOLD = list(map(str.strip, trvlParams[0].replace("[","").replace("]","").split(",")))
INVALIDATION_THRESHOLD = list(map(str.strip, trvlParams[1].replace("[","").replace("]","").split(",")))
# print(CHANGE_THRESHOLD)
# print(INVALIDATION_THRESHOLD)
myfile.close()

rootPath = '/home/sc/streamingPipeline'
refPath = '/home/sc/streamingPipeline/analysisData/ref'

# delete existing csv file
os.system("rm /home/sc/streamingPipeline/analysisData/temporal-rvl-data/result.csv")

run = 0
os.system('''export DISPLAY=":0.0"''')
os.system('clear && cd {:}/build && make'.format(rootPath))

# run rvl once, and overwrite file 
cmd = 'cd {:}/build && ./trvl {:}/ allDepthBin {:} {:} 0 0'.format(rootPath, refPath, 0, 0)
# cmd = 'cd {:}/build && ./trvl /home/sc/streamingPipeline/analysisData/temporal-rvl-data/ ppt2-sitting {:} {:} 0 0'.format(rootPath, 0, 0)
# cmd = 'cd {:}/build && ./trvl /home/sc/streamingPipeline/analysisData/ref/ frame_93_camera_0_depthBin {:} {:} 0 0'.format(rootPath, 0, 0)
print(cmd)
os.system(cmd)

for ct in CHANGE_THRESHOLD:
    for it in INVALIDATION_THRESHOLD:
        cmd = 'cd {:}/build && ./trvl {:}/ allDepthBin {:} {:} 1 1'.format(rootPath, refPath, ct, it)
        # cmd = 'cd {:}/build && ./trvl {:}/analysisData/temporal-rvl-data/ allDepthBin {:} {:} 1 1'.format(rootPath, rootPath, ct, it)
        # cmd = 'cd {:}/build && ./trvl /home/sc/streamingPipeline/analysisData/ref/ frame_93_camera_0_depthBin {:} {:} 1 1'.format(rootPath, ct, it)
        # cmd = 'cd {:}/build && ./trvl /home/sc/streamingPipeline/analysisData/temporal-rvl-data/ ppt2-sitting {:} {:} 1 1'.format(rootPath, 0, 0)
        print(cmd)
        os.system(cmd)
        
