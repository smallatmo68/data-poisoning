# Data Glitches Discovery using Influence-based Model Explanation

This repository contains the code of the paper Data Glitches Discovery using Influence-based Model Explanation. For the additional experiements, go at the end of the file.

## Installation and Running Instructions

**Note**: The experiments run on a 16-core Intel Xeon cpu, 64GB ram and no gpu. In our experiments we have used Python 3.9.

Install the necessary packages by running `pip install -r requirements.txt`

Run `make demo_single` that runs CNCI (for uniform-class noise, class-dependent noise)  and PCID (anomalies) for ResNet-20 on the Fashion MNIST dataset. The output is a barplot showing the F1-score of our signals CNCI (or PCID) w.r.t. the existing influence-based signals (SI, MAI, MI and GD-class).

Run `make demo_mixed` that runs CFRank (the proposed mixed signal), CNCI, and PCID for ResNet-20 on the MNIST dataset with mixed errors, both mislabeled and anomalous samples. The output (printed in the console) is the F1-scores of the three signals and the error characterization accuracy, i.e., how accurately a detected error is being characterized.

**Note**: In case of memory error, consider *decreasing* the "batch_size" in the json file "configs/resnet/tracin_resnet.json". 

## Pipeline Overview

![pipeline](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/f6af369b-c00e-4602-bd9d-f2956560b061)

## Training Settings 
Subsequently we report the datasets information:

| **Dataset**   | **Training Size** | **Validation Size** | **#Classes** |
|---------------|-------------------|---------------------|--------------|
| MNIST         | 60K               | 10K                 | 10           |
| Fashion-MNIST | 60K               | 10K                 | 10           |
| CIFAR-10      | 60K               | 10K                 | 10           |
| Forest Cover  | 116K              | 37K                 | 7            |
| Jannis        | 320K              | 80K                 | 2            |
| Epsilon       | 53K               | 13K                 | 4            |

Subsequently we report per dataset the learning rate, batch size and epochs that we train each foundational model. 

| Dataset       | Model    | Learning Rate | Batch Size | Epochs |
|---------------|----------|---------------|------------|--------|
|     MNIST     | ResNet   |          0.01 |        128 |      7 |
|     MNIST     | ViT      |           0.1 |        128 |      2 |
|     MNIST     | ConvNeXt |          0.05 |        128 |      2 |
| Fashion MNIST | ResNet   |          0.01 |        128 |      7 |
| Fashion MNIST | ViT      |           0.1 |        128 |      2 |
| Fashion MNIST | ConvNeXt |           0.1 |        128 |      4 |
|    CIFAR-10   | ResNet   |          0.01 |        128 |      7 |
|    CIFAR-10   | ViT      |           0.1 |        128 |      4 |
|    CIFAR-10   | ConvNeXt |           0.1 |        128 |      3 |

## Validation Performance 

The validation performances (accuracy on a validation set) of each model for a data glitch are reported in the table below. Note that the errors are presented in the training sets.

| Dataset       | Model    | Uniform Class Noise | Class-based Noise | Anomalies       |
|---------------|----------|---------------------|-------------------|-----------------|
|     MNIST     | ResNet   | 0.94 $\pm$ 0.01     | 0.95 $\pm$ 0.01   | 0.96 $\pm$ 0.01 |
|     MNIST     | ViT      | 0.83 $\pm$ 0.03     | 0.81 $\pm$ 0.03   | 0.90 $\pm$ 0.01 |
|     MNIST     | ConvNeXt | 0.83 $\pm$ 0.01     | 0.82 $\pm$ 0.02   | 0.87 $\pm$ 0.01 |
| Fashion MNIST | ResNet   | 0.82 $\pm$ 0.01     | 0.80 $\pm$ 0.01   | 0.82 $\pm$ 0.01 |
| Fashion MNIST | ViT      | 0.81 $\pm$ 0.01     | 0.81 $\pm$ 0.01   | 0.83 $\pm$ 0.01 |
| Fashion MNIST | ConvNeXt | 0.81 $\pm$ 0.01     | 0.80 $\pm$ 0.01   | 0.82 $\pm$ 0.01 |
|    CIFAR-10   | ResNet   | 0.89 $\pm$ 0.01     | 0.88 $\pm$ 0.00   | 0.90 $\pm$ 0.00 |
|    CIFAR-10   | ViT      | 0.90 $\pm$ 0.01     | 0.90 $\pm$ 0.01   | 0.91 $\pm$ 0.01 |
|    CIFAR-10   | ConvNeXt | 0.85 $\pm$ 0.01     | 0.85 $\pm$ 0.00   | 0.87 $\pm$ 0.00 |


## Class-Based Detection for 5 runs with different random seeds
Comparison of F1-Score on 10% class-dependent noise detection between CNCI and existing influence signals. Note that in this case, 10% of the samples of each class are relabeled to another random class. CNCI is on par or better on detecting class-based mislabeled samples.

![mcb_sigs_cont_all](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fa0ce8b0-a9a7-44e3-a11e-c23d6c1b2f06)

## Unreduced F1-Score for Data/Model pairs for 5 runs with different random seeds

We report the unreduced performances from the comparative plots for the different models and datasets reported in the paper.

### Uniform Class Noise (Performance per dataset)
![raw_mu_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/63666b97-9230-409b-b926-09077075e1ec)

### Class-based Noise (Performance per dataset) 
In the following plots, we contaminated one class at random (selected class changes with random seed) with 10% mislabeled samples of another class.

![raw_mcb_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/73b5e95e-ba1a-4406-b55c-f6d53d142c30)

### Anomalies (Performance per dataset)
![raw_anom_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fc048f55-9a5e-40a2-855d-0f61ae3adb39)

## Additional Experiments 

### Experiments on Tabular Data 

In this experiment we used three deep learning models namely MLP, Resnet and FT-Transformer, that have been proven to be effective for various tabular datasets [NIPS '21](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf). We employed three diverse datasets with different sample and feature size, namely Forest Cover, Jannis and Epsilon. We injected 10% uniform mislabeled samples in the training set of each dataset. Note that we followed the same evaluation pipeline described in our manuscript. In the figure below, we observe that the proposed CNCI signal outperforms all prior influence-based signals for the three models and datasets, detecting uniform mislabeled samples with 0.65 F1-score on average.

![additional_experiments](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/ee7733bf-abfa-4b24-beed-97fd026251df)

### Ablation Study on Glitch Ratio

In this experiment we tried different anomaly and mislabeled ratios ranging from 1% up to 30%. 

Subsequently we present the influence-based signals F1-score for increasing mislabeled ratio. CNCI outperforms the influence-based signals in every dataset for both models, especially for low class-noise ratios. Specifically, for a 1% ratio CNCI achieves 36% better F1-score on average than SI (second-best signal).

![mislabelled_uniform_noise_exp](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/82cec48e-4f45-4dec-a586-b21a7ca9ae46)

As depicted in the figure below, for the anomalous samples, the performance of PCID increases as the anomaly ratio increases. Note that the performance significantly decreases for smaller anomaly ratios. 

![ood_far_cl_noise_exp](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/f199878d-7123-49ca-b425-df6f3399b877)

### K&L vs TracIn IF Approximation Quality to LOOR 
We run apprixmation experiments using both IFs. LOOR is considered the gold standard of influence and each IF is considered qualitative if it provides better LOOR approximation. For these experiments TracIn has been proven to provide more accurate approximations and thus has been chosen in our experimental evaluation. Specifically, in the plots below we show the ECDF for TracIn and K&L IFs on approximating the LOOR, which is a standard experiment in each influence function work. We have used two tabular (Breast cancer, wine), and one image dataset (MNIST). A good approximation is when the area under an ECDF curve is as small as possible, indicating for the majority of the samples, the correlation their train-to-validation influence is positively correlated with LOOR. Note that we performed this experiment for each training sample in the dataset. In the tables below the two Figures, we observe that TracIn approximates better the LOOR than K&L exhibiting a higher positive correlation.

<img width="992" alt="koh_vs_tracin" src="https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/11f80a7a-4135-4998-8882-fbac2878743a">

### Analysis of Different Glitches in Training Set

Subsequently we report a table showing the average loss and accuracy on clean, mislabeled and anomalous samples. The glitch ratio is 10%. We observe that ResNet learns the anomalies acheiving 100% accuracy and a smaller average loss than in clean samples. On the other hand, the model does not learn the mislabeled samples.

| Dataset       | Model    | Avg. Loss / Acc. of Clean Train Samples | Avg. Loss / Acc. of Mislabeled Samples| Avg. Loss / Acc. of Anomalies|
|---------------|----------|:---------------------:|:-------------------:|:-----------------:|
|     MNIST     | ResNet-20   |  1.60 / 97%          |     2.35 / 5%    |    1.47 / 100% |
|     F-MNIST   | ResNet-20      | 1.67 / 91%          |     2.35 / 9%    |     1.48 / 100% |
|     CIFAR-10  | ResNet-20 | 1.76 / 95%          |     2.38 / 6%    |     1.59 / 100% |

The next table illustrates the validation performance difference (accuracy) w.r.t. clean model performance, i.e., no glitches in the training set. Specifically, each cell represents the glitched_train_validation_accuracy - clean_train_validation_accuracy. Note that in the case of uniform class noise, the performance is mostly negative (the uniform mislabeled samples affect the performance). Interestingly enough, when trained with anomalies, ResNet has mostly better performance w.r.t. the clean train set. This is a behavior that may be attributed to these samples; thus, the existence of such anomalies should be reported to the model expert. 

| Dataset       | Model    | Uniform Class Noise | Anomalies |
|---------------|----------|:---------------------:|:-----------:|
| MNIST         | ResNet-20   | -0.2             | -0.006    |
| Fashion MNIST | ResNet-20   | -0.01               | +0.31     |
| CIFAR-10      | ResNet-20   | 0.00               | +0.19     |

The next figure shows the relation of increasing self influence (SI) values and prediction accuracy. We can observe that the clean samples with higher SI values exhibit a substantially smaller prediction accuracy than the ones with low SI. These high SI samples are hard-but-clean samples that confuse SI to detect them as mislabeled. 

![hard_samples](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/577924df-de0f-49bd-8e50-c145a36b1404)

### Anomaly Detection using MNIST-C and CIFAR-10-C

MNIST-C: https://zenodo.org/records/3239543
CIFAR-C: https://zenodo.org/records/2535967

In the subsequent plots PCID is compared with the rest influence-based signals on detecting images with corruptions in MNIST and CIFAR-10 training sets. We followed the same evaluation pipeline as described in the paper. Specifically, we injected 10% brightness and stripe corruptions generated using [mnist-c](https://github.com/google-research/mnist-c). We chose these two corruptions based on their impact on the models' generalisation according to \[Mu et al.\]. The experiments were run for ResNet-20 and ConvNeXt. As we can see from the plots, PCID is the only signal that detects corrupted images in MNIST-C for both ResNet-20 and ConvNeXt, while all existing signals do not detect any anomaly (corrupted image) in the top 10% (F1-Score = 0) which is the anomaly ratio. In total, PCID outperforms all existing influence signals in all but one out of the 8 dataset/model pairs. Moreover, PCID exhibits robust detection behavior on both datasets and corruption types.

Mu, Norman, and Justin Gilmer. "Mnist-c: A robustness benchmark for computer vision." arXiv preprint arXiv:1906.02337 (2019).

![corruptions](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fad008a9-dbcb-4c43-88df-9150930e58eb)

### Glitch Detection in Training using a Partially Cleaned Validation Set

In the following group of plots we show the detection performance of CNCI when both train and test sets contain mislabeled samples (uniform and class-based label noise). The mislabeled ratio is 10% on training and 20% on the validation set. The validation set is 15% w.r.t. the training set. The results for tabular and image datasets are presented below. Each circle denotes a different combination of model and dataset. The diagonal like means no detection performance difference (in terms of F1) when the label noise exists in the validation set. *The dirty validation set slightly affects the detection efficacy of CNCI signal, making it robust to practical situations where the access to a totally clean validation set is not possible.*

![dirty_val_ablation](https://github.com/user-attachments/assets/56bea8dd-07c2-4e9a-b66d-673599934411)

### Comparative analysis of label noise detection and repair on tabular data

The following group of plots depicts the detection performance of CNCI vs the dedicated detector CleanLab along with the label repair accuracy, i.e., how accurate are the repair suggestions by each method. In plots (a) and (b) CNCI outperforms CleanLab for both label noise types across the three tabular and vision datasets averaged by the corresponding ML model. Regarding the label repair, CNCI suggests more accurate labels as repairs than CleanLab except for the FT-Transformer in the class-based noise. To ensure a fair comparison the experiment considers all the mislabeled samples (detected or not) and measures how accurately the methods propose label fixes. It is important to note that if a method detects many samples incorrectly as mislabeled, then, the labels of many clean samples will be altered. *CNCI reduces this phenomenon by performing a more accurate detection of both label noise types than CleanLab in tabular data, accompanied by accurate label repair suggestions*.

![tab-cnci-cleanlab](https://github.com/user-attachments/assets/8f67f8ef-613b-40ed-9523-4a793e62cb4d)

### Execution time of the influence signals 

The following group of plots depicts the execution time of the CNCI (for image and tabular data) and PCID (for image data). 

The time of the method to run end-to-end is decomposed as follows:

* T1. Model training: in this stage the model is trained for few epochs **T** and the gradients of the influence layers (the layers that will be considered for the infleunce computations) are stored in the disk for every sample.

* T2. To compute the train-to-validation influence matrix **O(*Tnmd*)** computations are required, where the training samples' partial derivatives are **n** x ***d** (**n**: training sample size and **d**: number of parameters of infleunce layers) and the validation samples' partial derivates results in a matrix **m** x **d** (**m**: validation sample size and m << n).

* T3. To compute the proposed the counterfactual influence matrix an additional infleunce step, based only on the weights of the last epoch, is required resulting to **O(*nmd*)**. 
  
* T4. Finally, to compute CNCI and PCID **O(*nm|C|*)** computations are required, where |C| is the number of classes.

From the previous steps, *T1* is the cost to train the target model and *T2* is the cost to calculate any joint signal (such as AAI, MI and GD-Class included in our work). The proposed counterfactual signals (CNCI and PCID) add a small additional overhead with the extra infleunce step using the counterfactual class of each sample resulting to *T3*. Note that for the mislabeled samples this has an additional benefit of repairing a potential label issue, as shown in the experiments of the paper. It is important to note that the execution time of *T1*, *T2* and *T3* are <ins>significantly affected</ins> by the (i) model size, (ii) the resources such as  GPU, CPU, and main memory, and (iii) the library implementation. The final step *T4* is not dependent on the model size or the available resources.

*Note that in the reported execution times we do not make use of a GPU; such utilization would have accelerated further the signals’ execution time, especially for models with many parameters such as transformer architectures*. 

In the following figure, both signals take only a small fraction ~8% of the total model training time. For a meaningful comparison we train the models on MNIST keeping 10% of the samples for the same number of epochs. Both signals are shown to be an efficient and effective option, even for large models such as the Vision Transformer (ViT). 

![img](https://github.com/user-attachments/assets/4529951e-532f-4ead-8c63-b0ba4578f9b6)

### Mislabeled Samples Detection on ImageNet Dogs

We have employed a challenging version of a subset of imagenet that contains 10 different dog breeds, taken from [FastAI](https://github.com/fastai/imagenette?tab=readme-ov-file#imagewoof). This dataset contains 320x333 high quality images. We injected 10% mislabeled samples in train set and run the influence-based signals to measure the detection performance. CNCI outperforms all existing influence-based signals.  It is worth noting that for both ConvNeXt and ViT, CNCI achieves the highest detection performance in ImageNet-Dogs (resized to 224x224) and the lowest in MNIST, which contains 28x28 images. We also observe a substantial boost in detection performance for all influence signals in Imagenet-Dogs. This is a justification of our previous claim, i.e., that with higher quality datasets, we expect more accurate influence estimates and therefore better detection using the influence signals.

![imagewoof](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/f86b1550-57a7-4284-ba5b-3b8a9c625536)
