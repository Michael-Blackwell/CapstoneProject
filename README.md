# Data 6910: Attention Weighted Fully Convolutional Networks for Segmentation and Classification of Dermatoscopic Images 

**Author:** Michael Blackwell

**objective:** To enhance the performance of an FCNN by using attention weighting on the extracted features. 

**Data Set:** 
 
2017 ISIC Dataset - https://challenge.isic-archive.com/data/#2017

- Dataset Home Page: https://challenge.isic-archive.com/landing/2017/



## Proposed Model
***

["CrissCross"](https://github.com/speedinghzl/CCNet) Attention Weighted U-Net

## Results
***
### Metrics
| Model                                              | Dice    | mIoU   | Recall     | Precision | 
|----------------------------------------------------|---------|--------|------------|-----------|
| W/ Attention                                       | 0.8106  | 0.8029 |  0.7674    |   0.8893  |
| W/O Attention                                      | 0.7596  | 0.7612 |  0.7211    |   0.8486  |
| [**Intel U-Net**](https://github.com/IntelAI/unet) | 0.7753  | 0.7623 |  0.665     |  0.8946   |

### Model Information
| Model         | Flops          | Trainable Params | Non-Trainable Params | Total Params | 
|---------------|----------------|------------------|----------------------|--------------|
| W/ Attention  | 8,614,412,288  | 3,521,901        | 2,500                | 3,524,401    |
| W/O Attention | 8,413,184,000  | 2,181,481        | 1,668                | 2,183,149    |
| [**Intel U-Net**](https://github.com/IntelAI/unet)     | 12,153,913,344 | 1,941,105        | 0                    | 1,941,105    |

## Primary References
***
Criss Cross Attention |  https://github.com/speedinghzl/CCNet

Attention is All You Need | https://arxiv.org/abs/1706.03762

ISIC 2017 Challenge | https://arxiv.org/abs/1710.05006
