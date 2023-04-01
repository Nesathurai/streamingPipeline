import numpy as np
import os


myfile = open("settings.txt", "r")
framesPerSetting = int(myfile.readline().split("=")[-1].strip())
# print(framesPerSetting)
voxelSizes = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(voxelSizes)
bitRates = myfile.readline().split("=")[-1].replace("[","").replace("]","").replace(",","").replace("\n","").split(" ")[1:]
# print(bitRates)
myfile.close()

# framesPerSetting = 300
# bitRates = ["1M", "2M", "5M", "10M", "100M"]
rootPath = '/home/sc/streamingPipeline'

# for each setting, compress textures and compute SSIM
os.system('cd {:}/analysisData/ref/ && ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*.png" -vcodec libx264 -crf 0 -pix_fmt yuv420p bit_rate_ref.mp4'.format(rootPath))

for bitRate in bitRates:
    os.system('cd {:}/analysisData/ref/ && ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*.png" -vcodec libx264 -pix_fmt yuv420p -b:v {:} bit_rate_{:}.mp4'.format(rootPath, bitRate, bitRate))
    os.system('cd {:}/analysisData/ref/ && ffmpeg -i bit_rate_{:}.mp4 -i bit_rate_ref.mp4 -lavfi ssim=stats_file=ssim_bit_rate_{:}.txt -f null -'.format(rootPath,bitRate,bitRate))

# ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "*color.png"  -vcodec libx264 -pix_fmt yuv420p -b:v 2M bit_rate_.mp4
# os.system('draco_encoder -i frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_1.obj -o frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_cl_'+str(cl)+'.drc -cl '+str(cl))
