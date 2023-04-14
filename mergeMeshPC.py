import numpy as np
import os

# framesPerSetting = 10 
# voxelSizes = [0.01, 0.02, 0.1]
cameras = 4
rootPath = '/home/sc/streamingPipeline'
filePath = '../analysisData/meshPCs'

os.system('clear && cd {:}/build && make'.format(rootPath))


myfile = open("settings.txt", "r")
framesPerSetting = int(myfile.readline().split("=")[-1].strip())
# print(framesPerSetting)
voxelSizes = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(voxelSizes)
myfile.close()

# merge meshes
for voxelSize in voxelSizes:
    for frame in range(framesPerSetting):
        # for each setting, compress textures and compute SSIM
        f0 = "{:}/frame_{:}_camera_0_vx_{:.5f}.obj".format(filePath,frame,float(voxelSize))
        f1 = "{:}/frame_{:}_camera_1_vx_{:.5f}.obj".format(filePath,frame,float(voxelSize))
        f2 = "{:}/frame_{:}_camera_2_vx_{:.5f}.obj".format(filePath,frame,float(voxelSize))
        f3 = "{:}/frame_{:}_camera_3_vx_{:.5f}.obj".format(filePath,frame,float(voxelSize))
        # print(f0)
        os.system('cd {:}/build/ && ./mergeMesh {:} {:} {:} {:}'.format(rootPath, f0, f1, f2, f3))

# merge PCs
for frame in range(framesPerSetting):
    # for each setting, compress textures and compute SSIM
    f0 = "{:}/frame_{:}_camera_0.ply".format(filePath,frame)
    f1 = "{:}/frame_{:}_camera_1.ply".format(filePath,frame)
    f2 = "{:}/frame_{:}_camera_2.ply".format(filePath,frame)
    f3 = "{:}/frame_{:}_camera_3.ply".format(filePath,frame)
    # print(f0)
    os.system('cd {:}/build/ && ./mergePC {:} {:} {:} {:}'.format(rootPath, f0, f1, f2, f3))
