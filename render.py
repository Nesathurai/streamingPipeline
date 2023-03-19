import open3d as o3d
import numpy as np
import cv2
import os

isExist = os.path.exists('./render/')
if not isExist:
   os.makedirs('./render/')
   print("The new directory is created!")

mesh_model1 = o3d.io.read_triangle_model(f'/home/sc/streamingPipeline/analysisData/isometric_office/scene.gltf', print_progress=True)
mesh_model2 = o3d.io.read_triangle_model(f'/home/sc/streamingPipeline/analysisData/isometric_office/scene.gltf', print_progress=True)

h_res = 1920
v_res = 1080

intrinsic = np.array([[933, 0, 954],
                        [0, 933, 551],
                        [0, 0, 1]])

ToGLCamera = np.array([
    [1,  0,  0,  0],
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])
FromGLGamera = np.linalg.inv(ToGLCamera)

screenshot_num = 0

def screenshot(window):
    global screenshot_num

    render = o3d.visualization.rendering.OffscreenRenderer(h_res, v_res)
    
    extrinsics = ToGLCamera@window.scene.camera.get_view_matrix()
    render.setup_camera(intrinsic, extrinsics, h_res, v_res)
    # render.scene.scene.enable_sun_light(True)
    render.scene.scene.enable_indirect_light(True)
    render.scene.scene.set_indirect_light_intensity(55000)

    render.scene.add_model('mesh1', mesh_model1)
    render.scene.add_model('mesh2', mesh_model2)

    color_image = np.asarray(render.render_to_image())
    depth_image = (np.asarray(render.render_to_depth_image(z_in_view_space=True)) * 1000.0).astype('uint16')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./render/' + 'color_' + str(screenshot_num) + '.png', color_image)
    cv2.imwrite('./render/' + 'depth_' + str(screenshot_num) + '.png', depth_image)
    np.savetxt('./render/' + 'extrinsic_' + str(screenshot_num) + '.txt', extrinsics)

    screenshot_num += 1


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    window = o3d.visualization.O3DVisualizer('Synthetic scene', width=h_res, height=v_res)
    
    window.setup_camera(intrinsic, np.identity(4), h_res, v_res)
    window.add_geometry(f'mesh1', mesh_model1)
    window.add_geometry(f'mesh2', mesh_model2)
    window.reset_camera_to_default()
    window.show_menu(True)
    window.show_settings = True
    window.add_action("screenshot", screenshot)
    app.add_window(window)
    app.run()

if __name__ == "__main__":
    main()
