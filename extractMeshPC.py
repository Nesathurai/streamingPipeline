import numpy as np
import os

framesPerSetting = 300 
voxelSizes = [0.01, 0.02, 0.1]
cameras = 4
rootPath = '/home/sc/streamingPipeline'
filePath = '../analysisData/ref'

os.system('clear && cd {:}/build && make'.format(rootPath))

for voxelSize in voxelSizes:
    for camera in range(cameras):
        for frame in range(framesPerSetting):
            # for each setting, compress textures and compute SSIM
            calib = "{:}/frame_{:}_camera_{:}_color.png".format(filePath,frame,camera)
            color = "{:}/frame_{:}_camera_{:}_color.png".format(filePath,frame,camera)
            depth = "{:}/frame_{:}_camera_{:}_depth.png".format(filePath,frame,camera)
            os.system('cd {:}/build/ && ./extractMesh {:} {:} {:} {:}'.format(rootPath, calib, color, depth, voxelSize))
            os.system('cd {:}/build/ && ./extractPC {:} {:} {:}'.format(rootPath, calib, color, depth))