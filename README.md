# CYM
this is modifying code for Cell detect base Yolo Mamba(CYM)

# Main environment
The environment installation procedure can be followed by VM-UNet, or by following the steps below:

```bash
conda create -n exp python=3.8
conda activate exp
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 # --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
pip install psutil seaborn pandas ipython tensorboard
```
# 2024.11.5

## add the mamba's ss2d module, and fusion the H-ss2d module!(Although the memory usage has increased and the batch-size has reduce to 2, the model is still able to perform comparable to no pretrian yolov11-l and batch-size with 8)

# 2024.11.20

## add the LDConv module, modified the backbone, the LDConv (linear Deformable Convolution) can improve the performance of the model, batch-size with 2 and ephoch with 300,the Recall can from 0.86~0.87 reach to 0.9+(comparable with the cym with ss2d and cpam)

# Updating ...