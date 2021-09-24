import pyrealsense2 as rs
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from PIL import Image
import math
#%%

def find_device_json_input_interface() :
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B64"]
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices();
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No product line device that has json input interface")


def get_pointcloud():

    jsonDict = json.load(open("640X480_L_short_default.json"))
    jsonString= str(jsonDict).replace("'", '\"')


    #%%
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        dev = find_device_json_input_interface()
        ser_dev = rs.serializable_device(dev)
        ser_dev.load_json(jsonString)
        print("loaded json")


    except Exception as e:
        print(e)
        pass

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    #    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



    print("[INFO] start streaming...")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print("width is: ",intr.width)
    print("height is: ",intr.height)
    print("ppx is: ",intr.ppx)
    print("ppy is: ",intr.ppy)
    print("fx is: ",intr.fx)
    print("fy is: ",intr.fy)
    HFOV = math.degrees(2*math.atan(intr.width/(intr.fx+intr.fy)))
    print('HFOV is',HFOV)
    VFOV = math.degrees(2*math.atan(intr.height/(intr.fx+intr.fy)))
    print('VFOV is',VFOV)

    #aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
    point_cloud = rs.pointcloud()

    #while True:
#    for _ in range(10):
    frames = pipeline.wait_for_frames()
    #frames = aligned_stream.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
#    points = point_cloud.calculate(depth_frame)
#    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,640,3) # xyz
#
    decimation = rs.decimation_filter(2)
    depth_frame = decimation.process(depth_frame)

    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,320,3)# xyz

    depth_image = np.asanyarray(depth_frame.get_data())*depth_scale
    depth_image = depth_image[:,40:280]

    color = np.asanyarray(color_frame.get_data())
    color = color[:,210:750,:]

    pipeline.stop()
    verts = verts[:,40:280,:]
    return verts, color#depth_image

#verts, depth_image = get_pointcloud()
##cx = verts[:,:,0]
##cy = verts[:,:,1]
#cz = verts[:,:,2]

#depth = np.floor(1/np.max(cz)*cz*255)
#depth = np.expand_dims(depth, axis=2)
#img_d= np.concatenate((depth, depth, depth), axis=-1).astype(np.uint8)
#plt.imshow(img_d)
