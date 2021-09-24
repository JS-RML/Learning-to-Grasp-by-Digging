
# Learning to Pick by Digging: Data-Driven Dig-Grasping

## 1. Overview
In this work, we propose a learning framework for a manipulation technique to bin picking, named **Dig-Grasping**, which aims at singulating and simultaneously picking the objects one by one from a random clutter. Dig-grasping illustrates a way of grasping through a physical interaction between the robot's gripper and object clutter, realized as a digging operation along a straight line. A gripper designed for this technique is capable of changing relative digit lengths such that the object being digged will not collide with the other finger. This repository provides the PyTorch implementation for training (in simulation) and testing (in both simulation and real world) the action primitives of dig-grasping. 

The following figure shows an overall process. Given a depth image, the robot learns a way of interaction (encoded in  the  score  map)  with  the  target  object  to  pick  up. The object (blue  block)  is rotated by the finger pushing it down to the clutter.
<p align = "center">
<img src="files/fg1.jpg" width="485" height="348"> 
</p>

Video demonstration:
<p align = "center">
<img src="files/hg.gif" width="640" height="360"> 
</p>
<p align = "center">
<img src="files/tube.gif" width="320" height="180"> 
<img src="files/key.gif" width="320" height="180"> 
</p>

**Full Video** can be seen from this [link](https://youtu.be/3zgnn5pVX9c).


## 2. Prerequisites
### 2.1 Hardware
- [**Universal Robot UR10**](https://www.universal-robots.com/products/ur10-robot/)
- [**Robotiq 140mm Adaptive parallel-jaw gripper**](https://robotiq.com/products/2f85-140-adaptive-robot-gripper)
- [**RealSense Camera L515**](https://www.intelrealsense.com/lidar-camera-l515/)
- [**Extendable Finger**](https://github.com/HKUST-RML/extendable_finger) for realizing finger length differences during digging

### 2.2 Software
The code is built with Python 3.6. Libraries are listed in [[requirements.yaml](https://github.com/HKUST-RML/Learning-to-Grasp-by-Digging_v2/blob/main/requirements.yaml "requirements.yaml")] and can be installed with conds by:

    conda env create -n learn_dig -f requirements.yaml
    
## 3. A Quick Start (Demo in Simulation)
This demo runs with our trained model in simulation, which depicts the bin picking of domino blocks from a cluttered bin.

**Instruction**
1. Download this repository:
```
https://github.com/HKUST-RML/Learning-to-Grasp-by-Digging_v2.git
```
2. Run the script (the trained model will be downloaded automatically)
```
    python demo.py
```

<p align = "center">
<img src="files/sim_demo.gif" width="640" height="480"> 
</p>
    
## 4. Training
The entire training process is repeated 7 times in a self-supervised manner. At each time, we recollect a dataset that contains 5000 scenes (bins of objects).

### 4.1 Dataset
We provide [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/czhaobb_connect_ust_hk/EXnUmbbMxzFOhFbPu0U12f8BZcG52E8plFfe4K3j_b_lSQ?e=mbeOMj) the dataset in our last training process. 

You can train the model with this provided dataset by running:
```
python train_last_model.py
```

### 4.2 Train
If you want to create your own dataset, and start training the models from scratch, please run the following code:
```
python trainer.py 
```

## 5. Test
### 5.1 Test in simulation
Here we provide a testing script to reproduce the simulation results in the paper. Our saved model will download automatically. The model is tested on three objects which are used during training.

    python test_in_sim.py

### 5.2 Test in real with UR10 and Robotiq140 gripper
Here we provide the script to test our method in a real robot. The saved models will download automatically.

    cd real
    python test_in_real.py
