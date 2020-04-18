# DHNN_BodyRepresentation
This repository includes the experiment code and trained model of paper "Disentangled Human Body Embedding Based on Deep Hierarchical Neural Network", TVCG 2020, [https://arxiv.org/abs/1905.05622](https://arxiv.org/abs/1905.05622).
Authors: Boyi Jiang, [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/), Jianfei Cai and Jianmin Zheng
## Dataset
The transformed connectivity consistent human body dataset can be downloaded [here](https://github.com/Juyong/DHNN_BodyRepresentation).
## Requirements
### 1. Basic Environment
- Ubuntu,  CUDA-10.0, Python 3.7
- Python packages: PyTorch(1.3.0), PyTorch geometric(1.3.2), Numpy, openmesh, opencv, [batch_knn](https://github.com/jby1993/gpu_batch_knn)(some experiment codes need).
### 2. Install
- Make a folder called "models" in the repository directory.
- Download the [trained model](https://github.com/Juyong/DHNN_BodyRepresentation) and unzip to the "models" folder..
- Download the [trained extension model](https://github.com/Juyong/DHNN_BodyRepresentation) and unzip to the "models" folder.(Optional)
## Usage
The "exps" folder includes sample codes with test data of several experiments described in the paper.
## Citation
Please cite the following papers if it helps your research:
 @inproceedings{Jiang2020HumanBody,
      title={Disentangled Human Body Embedding Based on Deep Hierarchical Neural Network},
      author={Jiang, Boyi and Zhang, Juyong and Cai, Jianfei and Zheng, Jianmin},
      booktitle={IEEE Transactions on Visualization and Computer Graphics},
      year={2020}
}
