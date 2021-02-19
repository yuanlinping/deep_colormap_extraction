# Deep Colormap Extraction from Visualizations

This repository contains code and materials for the paper _Deep Colormap Extraction from Visualizations,_ contributed by Lin-Ping Yuan, Wei Zeng, Siwei Fu, Zhiliang Zeng, Haotian Li, Chi-Wing Fu, and Huamin Qu. All rights reserved by authors.

-----
## Introduction
This work presents a new approach based on deep learning to automatically extract colormaps from visualizations. After summarizing colors in an input visualization image as a Lab color histogram, we pass the histogram to a pre-trained deep neural network, which learns to predict the colormap that produces the visualization. To train the network, we create a new dataset of ∼64K visualizations that cover various data distributions, chart types, and colormaps. The network adopts an atrous spatial pyramid pooling module to capture color features at multiple scales in the input color histograms. We then classify the predicted colormap as discrete or continuous, and refine the predicted colormap based on its color histogram.

![The method pipeline.](assets/pipeline.png)

## Datasets

The synthetic visualization datasets, the colormaps used for producing the visualizations, and real-world images used for evaluation can be downloaded from [here](https://bit.ly/2rOJTNw).

## Usage
### Split datasets
- Download the datasets from [here](https://bit.ly/2rOJTNw) and put them in the folder ./dataset
- Get training, testing, evaluating file lists by running:
```
python utils/gen_file_lists.py 63956 *.csv *.png. # 63956 is the number of total pairs of visualization and colormaps
```

### Training
```
python train.py --with_aspp True --backbone resnet18 --mode 1
```

### Testing
```
python inference.py --with_aspp True --backbone resnet18 --mode 1  --trained_model_config xxx # xxx is the trained model file name
```
