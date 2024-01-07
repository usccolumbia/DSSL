# DSSL
Physics guided dual self-supervised learning for materials property prediction

Nihang Fu, Jianjun Hu* <br>

Machine Learning and Evolution Laboratory <br>
Department of Computer Science and Engineering <br>
University of South Carolina

## Table of Contents
- [Installations](#Installations)

- [Datasets](#Datasets)

- [Usage](#Usage)

- [Pretrained Models](#Pretrained-models)

- [Performance](#Performance)

- [Acknowledgement](#Acknowledgement)

## Installations

0. Set up virtual environment
```
conda create -n dssl
conda activate dssl
```

1. PyTorch for computers with Nvidia GPU.
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

2. Other packagess
```
pip install -r requirements.txt
```  

## Datasets  


## Usage
### A Quick Run

### Training

### Predict


## Pretrained Models
Our trained models can be downloaded from figshare [BERTOS models](https://figshare.com/articles/online_resource/BERTOS_model/21554823), and you can use it as a test or prediction model.


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

