# DHNN_BodyRepresentation
This repository includes the experiment code and trained model of paper "Disentangled Human Body Embedding Based on Deep Hierarchical Neural Network", TVCG 2020, [https://arxiv.org/abs/1905.05622](https://arxiv.org/abs/1905.05622).
Authors: Boyi Jiang, [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/), Jianfei Cai and Jianmin Zheng
## Dataset
The transformed connectivity consistent human body dataset, including the ACAP feature and mesh file, can be downloaded [here](https://drive.google.com/file/d/1tq5pZT369rf1vv6aKHCzcT49eHLp3s0Z/view?usp=sharing). 
Note that these meshes are transformed from partial data of other datasets, including [SCAPE](http://robotics.stanford.edu/~drago/Projects/scape/scape.html), [FAUST](http://faust.is.tue.mpg.de/), [Dyna](http://dyna.is.tue.mpg.de/downloads), [MANO](https://mano.is.tue.mpg.de/), [Hasler et al](http://gvvperfcapeva.mpi-inf.mpg.de/public/ScanDB/) and [SPRING](https://graphics.soe.ucsc.edu/data/BodyModels/index.html). No commercial usage of the data is allowed.

## Requirements
### 1. Basic Environment
- Ubuntu, TITAN Xp,  CUDA-10.0, Python 3.7
- Python packages: PyTorch(1.3.0), PyTorch geometric(1.3.2), Numpy, openmesh, opencv, [batch_knn](https://github.com/jby1993/gpu_batch_knn)(some experiment codes need).
### 2. Install
- Make a folder called "models" in the repository directory.
- Download the [trained model](https://drive.google.com/open?id=1VMCK86OPVjO5wp1YpP13I8adWfQ061zG) and unzip to the "models" folder..
- Download the [trained extension model](https://drive.google.com/open?id=1FpHrKr9_2Hsno63Ox_T529bVa_tzcXh2) and unzip to the "models" folder.(Optional)
## Usage
The "exps" folder includes sample codes with test data of several experiments described in the paper.
- PoseTransfer：A sample code of pose transfer, running this code should produce the results of figure 13 in the paper.
- point2point: A sample code for registering meshes with same connectivity.
- DFaustFit: A sample code of the DFaust registration experiment. (about 10 minutes)
- FaustAlign: A sample code of Faust alignment experiment.(about 20 minutes)
- BodyFrom2DJoints: A sample code of 3D body pose estimation from 2D joints.
## Citation
Please cite the following paper if it helps your research: 
 ```
 @article{Jiang2020HumanBody,
      title={Disentangled Human Body Embedding Based on Deep Hierarchical Neural Network},
      author={Jiang, Boyi and Zhang, Juyong and Cai, Jianfei and Zheng, Jianmin},
      booktitle={IEEE Transactions on Visualization and Computer Graphics},
      year={2020}
}
 ```
