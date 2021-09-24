import os
import cv2
import math3d as m3d
import math
from scipy.spatial.transform import Rotation as R
import requests
from tqdm import tqdm

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def draw_circle(image, radius):
    """Draw a circle at the center of the given image."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Evaluate radius
#    radius = max(3, radius)
    inner_radius = radius
    cv2.circle(image, center, inner_radius, (255, 255, 255), -1)
#    cv2.circle(image, center, inner_radius, (0, 0, 0), 2)
    return image


def world_to_gripper_orn(pitch, roll, yaw):
    grip_rot = m3d.Transform()
    grip_rot.pos = (0,0,0)
    grip_rot.orient.rotate_yb(math.radians(roll)) # roll
    grip_rot.orient.rotate_xb(math.radians(pitch)) #pitch
    grip_rot.orient.rotate_zb(math.radians(yaw)) #yaw
    grip_matrix = grip_rot.get_matrix()
    robot_Orn = R.from_matrix(grip_matrix[:3,:3]).as_quat()
    return robot_Orn

def change_urdf_fingerlength(robot_path, tmp_z):
    #read input file
    fin = open(robot_path, "rt")
    #read file contents to string
    data = fin.readlines()
    # get z of joint _ short finger
    tmp = data[178].split(' ')
    #tmp_z, 0.02 向下?， -0.02，向上?
    tmp[7] = str(tmp_z)
    tmp = ' '.join(tmp)
    data[178] = tmp
    #close the input file
    fin.close()
    #open the input file in write mode
    fin = open(robot_path, "wt")
    #overrite the input file with the resulting data
    fin.writelines(data)
    #close the file
    fin.close()

####################
#download file from google drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
###################