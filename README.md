# Learning to Pick by Digging: Data-Driven Dig-Grasping

## 1. Overview
In this work, we propsoe a learning framework for a manipulation technique to bin picking, named **Dig-Grasping**, which aims at singulating and simultaneously picking the objects one by one from a random clutter. Dig-grasping illustrates a way of grasping through a physical interaction between the robot's gripper and object clutter, realized as a digging operation along a straight line. A gripper designed for this technique is capable of changing relative digit lengths such that the object being digged will not collide with the other finger. This repository provides the PyTorch implementation for training (in simulation) and testing (in both simulation and real world) the action primitives of dig-grasping. 

The following figure shows an overall process. Given a depth image, the robot learns a way of interaction (encoded in  the  score  map)  with  the  target  object  to  pick  up. The object (blue  block)  is rotated by the finger pushing it down to the clutter.
<p align = "center">
<img src="files/fg1.jpg" width="485" height="348"> 
</p>

## 2. Prerequisites
### 2.1 Hardware
- [**Universal Robot UR10**](https://www.universal-robots.com/products/ur10-robot/)
- [**Robotiq 140mm Adaptive parallel-jaw gripper**](https://robotiq.com/products/2f85-140-adaptive-robot-gripper)
- [**RealSense Camera SR300**](https://www.intelrealsense.com/lidar-camera-l515/)
- [**Extendable Finger**](https://github.com/HKUST-RML/extendable_finger) for realizing finger length differences during digging

### 2.2 Software
#### 2.2.1 Enviroment requirements
The code is built with Python 3.6. Libraries are listed in [[requirements.yaml](https://github.com/HKUST-RML/Learning-to-Grasp-by-Digging_v2/blob/main/requirements.yaml "requirements.yaml")] and can be installed with conds by:

    conda  env create -n learn_dig -f requirements.yaml
## 3 Training
To train the models presented in the paper:

    python trainer.py 

## 4 Test
### 4.1 Test in simulation
Here we provide testing script to reproduce the results in paper. The saved models will download automatically.

    python test_in_sim.py

### 4.2 Test in real with UR10 and Robotiq140 gripper
