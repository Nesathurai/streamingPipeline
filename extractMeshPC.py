import numpy as np
import os

cameras = 4
rootPath = '/home/sc/streamingPipeline'
filePath = '../analysisData/ref'
os.system('clear && cd {:}/build && make'.format(rootPath))

myfile = open("settings.txt", "r")
framesPerSetting = int(myfile.readline().split("=")[-1].strip())
# print(framesPerSetting)
voxelSizes = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(voxelSizes)
myfile.close()

for voxelSize in voxelSizes:
    for camera in range(cameras):
        for frame in range(framesPerSetting):
            # for each setting, compress textures and compute SSIM
            calib = "{:}/frame_{:}_camera_{:}_color.png".format(filePath,frame,camera)
            color = "{:}/frame_{:}_camera_{:}_color.png".format(filePath,frame,camera)
            depth = "{:}/frame_{:}_camera_{:}_depth.png".format(filePath,frame,camera)
            os.system('cd {:}/build/ && ./extractMesh {:} {:} {:} {:}'.format(rootPath, calib, color, depth, float(voxelSize)))
            os.system('cd {:}/build/ && ./extractPC {:} {:} {:}'.format(rootPath, calib, color, depth))