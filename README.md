# DSSL
Physics guided dual self-supervised learning for materials property prediction

Nihang Fu, Lai Wei, and Jianjun Hu* <br>

Machine Learning and Evolution Laboratory <br>
Department of Computer Science and Engineering <br>
University of South Carolina

## Table of Contents
- [Installations](#Installations)

- [Datasets](#Datasets)

- [Usage](#Usage)

- [Performance](#Performance)

- [Acknowledgement](#Acknowledgement)

## Installations

0. Set up a virtual environment using the following commands.
```
conda create -n dssl
conda activate dssl
```

1. **PyTorch**
Our code is tested on Pytorch 1.8.1 with cuda 11.1. Use the following command to install (or you can also install the other version using the command from the PyTorch website):
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
 
2. **Pytorch Geometric**
Replace the `${TORCH}` and `${CUDA}` with your corresponding pytorch and cuda version. For example, if your PyTorch version is 1.8.1 and your CUDA version is 11.1, then replace `${TORCH}` with `1.8.1` and `${CUDA}` with cu111.
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

3. Other packagess
```
pip install -r requirements.txt
```  

## Datasets  
| Datasets  | Description | Property | Sample No.|
| :---: | :---: | :---: | :---: | 
|  MP  | Pretraining  | - | 138, 614|
|  HOIP  | finetuning  | Band Gap (eV) | 1,345|
|  Lanthanides  | finetuning  | Formation Energ (eV/atom) | 4,166 | 
|  Fermi Energy  | finetuning  | Fermi Energy (eV) | 26,447 |
|  Formation Energy  | finetuning  | Formation Energy (eV/atom) | 26,741 |
|  Band Gap  | finetuning  | Band Gap (eV) | 27,110 |
|  JDFT2D  | finetuning  | Exfoliation Energy (meV/atom) | 636 |
|  Phonons  | finetuning  | Last Phdos Peak (1/cm) | 1,265 |
|  Dielectric  | finetuning  | Refractive Index (no unit) | 4,764 |
|  GVRH  | finetuning  | Shear Modulus (GPa)  | 10,987 |
|  KVRH  | finetuning  | Bulk Modulus (GPa) | 10,987|

## Usage
### A Quick Run

### Training

### Predict


## Performance
![Performance](performances.png)

## Acknowledgement
We use DeeperGATGNN as the backbone.
```
@article{omee2022scalable,
  title={Scalable deeper graph neural networks for high-performance materials property prediction},
  author={Omee, Sadman Sadeed and Louis, Steph-Yves and Fu, Nihang and Wei, Lai and Dey, Sourin and Dong, Rongzhi and Li, Qinyang and Hu, Jianjun},
  journal={Patterns},
  publisher={Elsevier}
}
```

## Cite our work


# Contact
If you have any problem using BERTOS, feel free to contact via [funihang@gmail.com](mailto:funihang@gmail.com).

