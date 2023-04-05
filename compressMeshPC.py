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

cameras = 4
framesPerSetting = 10

# framesPerSetting = 300
# bitRates = ["1M", "2M", "5M", "10M", "100M"]
rootPath = '/home/sc/streamingPipeline'
# filePath = '../analysisData'

# for each setting, compress textures and compute SSIM
# os.system('cd {:}/analysisData/ref/ && ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*.png" -vcodec libx264 -crf 0 -pix_fmt yuv420p bit_rate_ref.mp4'.format(rootPath))

# compress and decompress all meshes and PCs 

for compressRate in compressRates:
    for voxelSize in voxelSizes:
        for camera in range(cameras):
            for frame in range(framesPerSetting):
                meshIn = "{:}/analysisData/frame_{:}_camera_{:}_vx_{:.5f}.obj".format(rootPath, frame, camera, float(voxelSize))
                # print(meshIn)
                meshOut = "{:}/analysisData/frame_{:}_camera_{:}_vx_{:.5f}_mesh.drc".format(rootPath, frame, camera, float(voxelSize))
                # print(meshOut)
                meshDecompressed = "{:}/analysisData/frame_{:}_camera_{:}_vx_{:.5f}_decomp.obj".format(rootPath, frame, camera, float(voxelSize))
                print('cd {:}/ && draco_encoder -i {:} -o {:} -cl {:}'.format(rootPath, meshIn, meshOut, compressRate))
                os.system('cd {:}/ && draco_encoder -i {:} -o {:} -cl {:}'.format(rootPath, meshIn, meshOut, compressRate))
                print('cd {:}/ && draco_decoder -i {:} -o {:}'.format(rootPath, meshOut, meshDecompressed))
                os.system('cd {:}/ && draco_decoder -i {:} -o {:}'.format(rootPath, meshOut, meshDecompressed))


for compressRate in compressRates:
    for camera in range(cameras):
        for frame in range(framesPerSetting):
            PCIn = "{:}/analysisData/frame_{:}_camera_{:}.ply".format(rootPath, frame, camera)
            # print(PCIn)
            PCOut = "{:}/analysisData/frame_{:}_camera_{:}_PC.drc".format(rootPath, frame, camera)
            # print(PCOut)
            PCDecompressed = "{:}/analysisData/frame_{:}_camera_{:}_decomp.ply".format(rootPath, frame, camera)
            print('cd {:}/ && draco_encoder -point_cloud -i {:} -o {:} -cl {:}'.format(rootPath, PCIn, PCOut, compressRate))
            os.system('cd {:}/ && draco_encoder -point_cloud -i {:} -o {:} -cl {:}'.format(rootPath, PCIn, PCOut, compressRate))
            print('cd {:}/ && draco_decoder -point_cloud -i {:} -o {:}'.format(rootPath, PCOut, PCDecompressed))
            os.system('cd {:}/ && draco_decoder -point_cloud -i {:} -o {:}'.format(rootPath, PCOut, PCDecompressed))
                
                

#     os.system('cd {:}/analysisData/ref/ && ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*.png" -vcodec libx264 -pix_fmt yuv420p -b:v {:} bit_rate_{:}.mp4'.format(rootPath, bitRate, bitRate))
#     os.system('cd {:}/analysisData/ref/ && ffmpeg -i bit_rate_{:}.mp4 -i bit_rate_ref.mp4 -lavfi ssim=stats_file=ssim_bit_rate_{:}.txt -f null -'.format(rootPath,bitRate,bitRate))

# ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*color.png"  -vcodec libx264 -pix_fmt yuv420p -b:v 2M bit_rate_.mp4
# os.system('draco_encoder -i frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_1.obj -o frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_cl_'+str(cl)+'.drc -cl '+str(cl))
