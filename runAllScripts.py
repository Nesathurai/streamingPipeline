import os

myfile = open("settings.txt", "r")
framesPerSetting = int(myfile.readline().split("=")[-1].strip())
print(framesPerSetting)
voxelSizes = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
print(voxelSizes)
bitRates = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
print(bitRates)
myfile.close()

rootPath = '/home/sc/streamingPipeline'
os.system('cd {:} && python3 captureFrames.py'.format(rootPath))
os.system('cd {:} && python3 framesToVideoSSIM.py'.format(rootPath))
os.system('cd {:} && python3 extractMeshPC.py'.format(rootPath))
os.system('cd {:} && python3 mergeMeshPC.py'.format(rootPath))
os.system('cd {:} && python3 compressMeshPC.py'.format(rootPath))