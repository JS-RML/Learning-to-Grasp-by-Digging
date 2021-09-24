import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import numpy as np
import random
import math



class Sim:

    def __init__(self, state_save_path, num_obj, GUI, image_pixel, 
                 EyePosition,TargetPosition,fov,far,
                 near,robotStartOrn,object_path,mass_random,lateralFriction_random,globalScaling_random):
        """Creates a robot.
        Args:
          urdfRootPath: The path to the root URDF directory.
          timeStep: The Pybullet timestep to use for simulation.
          clientId: The Pybullet client's ID.
        """

        self.state_save_path = state_save_path
        self.num_obj = num_obj
        self.object_path = object_path
        self.robotStartOrn = robotStartOrn

        self.image_pixel = image_pixel
        self.EyePosition = EyePosition
        self.TargetPosition = TargetPosition
        self.fov = fov
        self.far = far
        self.near = near


        self.GUI = GUI
        self.numSolverIterations = 50
        self.plane_pos = [0,0,0]
        self.bowl_pos = [0,0,0]
        self.robotStartPos = [0, 0, 0.6]
        self.image_pixel_after = 240
        
        self.mass_random =  mass_random
        self.lateralFriction_random = lateralFriction_random
        self.globalScaling_random = globalScaling_random
        


        # GUI/DIRECT
        if self.GUI ==True:
            p.connect(p.GUI)
#            physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
#            physicsClient = p.connect(p.DIRECT)

    def render(self):
        view_matrix = p.computeViewMatrix(cameraEyePosition = self.EyePosition,
                                          cameraTargetPosition = self.TargetPosition,
                                          cameraUpVector = [0,1,0])
        proj_matrix = p.computeProjectionMatrixFOV(fov=self.fov,aspect =1,nearVal=self.near,farVal=self.far)
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=self.image_pixel,height=self.image_pixel,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)


        rgbImg = np.array(rgbImg).reshape(self.image_pixel,self.image_pixel,4)
        depthImg = np.array(depthImg).reshape(self.image_pixel,self.image_pixel)
        segImg = np.array(segImg).reshape(self.image_pixel,self.image_pixel)

        self.rgbImg = rgbImg[40:280,40:280]
        self.depthImg = depthImg[40:280,40:280]
        self.segImg= segImg[40:280,40:280]
        return self.rgbImg, self.depthImg, self.segImg

    def after_render(self):
        "depth"

        float_depth = self.far * self.near / (self.far - (self.far - self.near) * self.depthImg)

        #5mm max noise
        noise = np.random.normal(0,1,[self.image_pixel_after,self.image_pixel_after])
        noise = noise/(np.max(noise) - np.min(noise))/100
        float_depth = float_depth + noise

        move_depth = self.EyePosition[2]-float_depth #move point 

        depth = float_depth.copy()
        depth = np.floor(1/self.EyePosition[2]*depth*255).astype(np.int)
        depth = np.expand_dims(depth, axis=2)
        img_d= np.concatenate((depth, depth, depth), axis=-1).astype(np.uint8)

        fov_p = math.radians(self.fov)
        "480/640"
        came_can_len = self.EyePosition[2]*math.tan(fov_p/2)*self.image_pixel_after/self.image_pixel

        poke_pos_map=np.zeros((self.image_pixel_after,self.image_pixel_after),dtype=list)
        step = came_can_len*1000/(self.image_pixel_after/2)
        wh= step*self.image_pixel_after/2-1

        x=-wh
        for i in range(self.image_pixel_after):
            y=wh
            for j in range(self.image_pixel_after):
                'switch from 200mm to m, /1000'
                ratio = float_depth[j][i]/self.EyePosition[2]

                xy_tmp=[x/1000*ratio,y/1000*ratio,move_depth[j][i]]
                poke_pos_map[j][i]=xy_tmp
                y = y-step
            x = x+step
        return img_d, float_depth, poke_pos_map

    def build_e(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
#        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0.2])
        # define world
        p.setGravity(0, 0, -9.8) # NOTE
        self.planeID = p.loadURDF("plane.urdf",self.plane_pos ,useFixedBase = True)
        #######################################
        ###    define and setup object     ###
        #######################################
        self.bowlID = p.loadURDF('./objurdf/wbn/wbin.urdf',self.bowl_pos,p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase = True)

        # load Object

#        mass_random =  random.uniform(0.005, 0.006)
#        lateralFriction_random = random.uniform(0.25,0.35)
#        globalScaling_random = random.uniform(0.9,1)

        self.objectUid=[]
        for i in range(self.num_obj):
            xran =  random.uniform(-1,1)*0.08
            yran =  random.uniform(-1,1)*0.08
            pitch_ran =  random.uniform(-1,1)*3.14
            roll_ran =  random.uniform(-1,1)*3.14
            yaw_ran =  random.uniform(-1,1)*3.14
            Uid = p.loadURDF(self.object_path, [xran,yran,0.3],p.getQuaternionFromEuler([pitch_ran,roll_ran,yaw_ran]),globalScaling = self.globalScaling_random)
            # p.changeVisualShape(Uid, -1, rgbaColor=[0.1, 1, 0.1, 1.0])
#            p.changeVisualShape(Uid, -1, rgbaColor=[np.random.random(),np.random.random(), np.random.random(), 1.0])
            p.changeDynamics(Uid, -1, mass = self.mass_random, lateralFriction = self.lateralFriction_random)
            self.objectUid.append(Uid)
            for loop in range(20):
                p.stepSimulation()
        for loop in range(400):
            p.stepSimulation()
        #######################################
        ###    define and setup robot       ###
        #######################################

        robotUrdfPath = './gripper_urdf/00.urdf'
        self.robotID = p.loadURDF(robotUrdfPath, self.robotStartPos, self.robotStartOrn,
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.robotID)
        jointInfo = namedtuple("jointInfo",
                               ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

        self.joints = AttrDict()
        self.dummy_center_indicator_link_index = 0

        # get jointInfo and index of dummy_center_indicator_link
        for i in range(numJoints):
            info = p.getJointInfo(self.robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
            self.joints[singleInfo.name] = singleInfo
            # register index of dummy center link
            if jointName == "gripper_roll":
                self.dummy_center_indicator_link_index = i


        # position control
        position_control_group = []
        position_control_group.append(p.addUserDebugParameter('x', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('y', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('z', -0.25, 1, 0.2))
        position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, 0))
        position_control_group.append(p.addUserDebugParameter('pitch', 0, 3.14, 1.57))
        position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 5, 0))

        self.position_control_joint_name = ["center_x",
                                       "center_y",
                                       "center_z",
                                       "gripper_roll",
                                       "gripper_pitch",
                                       "gripper_yaw"]
        p.saveBullet(self.state_save_path)
        return self.objectUid, self.robotID, self.joints, self.dummy_center_indicator_link_index,self.position_control_joint_name
    def restore_env(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
#        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0.2])
        # define world
        p.setGravity(0, 0, -9.8) # NOTE
        self.planeID = p.loadURDF("plane.urdf",self.plane_pos ,useFixedBase = True)
        #######################################
        ###    define and setup robot       ###
        #######################################
        self.bowlID = p.loadURDF('./objurdf/wbin/wbin.urdf',self.bowl_pos,p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase = True)

        # load Object
        self.objectUid=[]
        for i in range(self.num_obj):
            xran =  random.uniform(-1,1)*0.08
            yran =  random.uniform(-1,1)*0.08
            pitch_ran =  random.uniform(-1,1)*3.14
            roll_ran =  random.uniform(-1,1)*3.14
            yaw_ran =  random.uniform(-1,1)*3.14
            Uid = p.loadURDF(self.object_path, [xran,yran,0.3],p.getQuaternionFromEuler([pitch_ran,roll_ran,yaw_ran]),globalScaling = self.globalScaling_random)
            # p.changeVisualShape(Uid, -1, rgbaColor=[0.1, 1, 0.1, 1.0])
#            p.changeVisualShape(Uid, -1, rgbaColor=[np.random.random(),np.random.random(), np.random.random(), 1.0])
            p.changeDynamics(Uid, -1, mass = self.mass_random, lateralFriction = self.lateralFriction_random)
            self.objectUid.append(Uid)
        #######################################
        ###    define and setup robot       ###
        #######################################

    #    p.restoreState(fileName="state_more_obj1.bullet")
        # load robot

        robotUrdfPath = './gripper_urdf/00.urdf'
        self.robotID = p.loadURDF(robotUrdfPath, self.robotStartPos, self.robotStartOrn,
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.robotID)
        jointInfo = namedtuple("jointInfo",
                               ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

        self.joints = AttrDict()
        self.dummy_center_indicator_link_index = 0

        # get jointInfo and index of dummy_center_indicator_link
        for i in range(numJoints):
            info = p.getJointInfo(self.robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
            self.joints[singleInfo.name] = singleInfo
            # register index of dummy center link
            if jointName == "gripper_roll":
                self.dummy_center_indicator_link_index = i


        # position control
        position_control_group = []
        position_control_group.append(p.addUserDebugParameter('x', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('y', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('z', -0.25, 1, 0.2))
        position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, 0))
        position_control_group.append(p.addUserDebugParameter('pitch', 0, 3.14, 1.57))
        position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 5, 0))

        self.position_control_joint_name = ["center_x",
                                       "center_y",
                                       "center_z",
                                       "gripper_roll",
                                       "gripper_pitch",
                                       "gripper_yaw"]
#        p.restoreState(self.state_save_path)
        p.restoreState(fileName=self.state_save_path)
        return self.objectUid, self.robotID, self.joints, self.dummy_center_indicator_link_index,self.position_control_joint_name

    def reset(self):
        p.restoreState(fileName=self.state_save_path)


    def step_action(self,xyz,ori,force=50,veclocity=.1):
        parameter_orientation = p.getQuaternionFromEuler([ori[0], ori[1], ori[2]])
        jointPose = p.calculateInverseKinematics(self.robotID,
                                                 self.dummy_center_indicator_link_index,
                                                 [xyz[0], xyz[1], xyz[2]],
                                                 parameter_orientation)
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                    targetPosition=jointPose[joint.id], force=force,
                                    maxVelocity=veclocity)


    def reset_and_poke(self,robot_sp,target_pos_orn,robot_orn,robot_path):

        robotStartOrn = robot_orn
        p.removeBody(self.robotID)
        robotUrdfPath = robot_path
        self.robotID = p.loadURDF(robotUrdfPath, robot_sp, robotStartOrn,flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        #p.resetBasePositionAndOrientation(robotID,robot_start_pos,robotStartOrn)


        self.step_action(target_pos_orn[0],target_pos_orn[1],force=50,veclocity=.05)
        for i in range(200):
            p.stepSimulation()

#        time.sleep(2)
        "close gripper"
        p.setJointMotorControl2(self.robotID,7,p.POSITION_CONTROL,targetPosition=0.1,force=30,maxVelocity=.05)
        for loop in range(200):
            p.stepSimulation()
#        time.sleep(2)
        "lift 10cm"
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robotID)
        self.step_action([robot_pos[0],robot_pos[1],robot_pos[2]+0.2],target_pos_orn[1],force=50,veclocity=.1)
        for i in range(200):
            p.stepSimulation()

        "cal condition, label"
        up_duomi = 0
        for duomi_uid in range(len(self.objectUid)):
            dp, _ = p.getBasePositionAndOrientation(duomi_uid)
            if dp[2]>0.27:
                up_duomi+=1
        if  up_duomi ==1:
           # print('true')
            label_at_pixel =128
        else:
            label_at_pixel =0
        return label_at_pixel
