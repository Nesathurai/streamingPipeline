# streamingPipeline

cmake -DCMAKE_PREFIX_PATH=/home/sc/open3d_install_0.16.0 ..
export DISPLAY=":1.0"
ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i "frame_*.jpg" -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4

./extractMesh ../analysisData/ref/frame_0_camera_0_color.png /home/sc/streamingPipeline/analysisData/ref/frame_0_camera_0_color.png /home/sc/streamingPipeline/analysisData/ref/frame_0_camera_0_depth.png 0.01
