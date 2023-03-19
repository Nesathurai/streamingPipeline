import numpy as np
import os
# vxs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#vxs = [5, 10, 15, 20]
# dcl = [1, 5, 10, 20, 30, 40, 50]
# cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
framesPerSetting = [1]
voxelSizes = [0.02, 0.015, 0.01, 0.009, 0.005]

# recompile the code 
os.system('cd /home/sc/streamingPipeline/build && make')

for vx in voxelSizes:
    for frame in framesPerSetting:
        os.system('cd /home/sc/streamingPipeline/build && ./reconstruction 1 ' + str(vx))
# os.system('draco_encoder -i frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_1.obj -o frames/mesh_frame_vxs_'+str(vx)+'_dcl_'+str(dc)+'_cl_'+str(cl)+'.drc -cl '+str(cl))
