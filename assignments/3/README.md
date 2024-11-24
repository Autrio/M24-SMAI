# Assignment - 3 MultiLayer Perceptron and AutoEncoders

To run the entire assignemnt subparts one after the other in the home folder run,

```console
python3 -m assignments.3.a3
```

## 2. Multi Layer Perceptron

### 2.1 Dataset Analysis and Preprocessing

#### 2.1.1 Statistics Summary

Preprocessing and analysing the data we get the following distribution

| Feature                  | Maximum  | Minimum  | Mean       | Std Dev    |
|--------------------------|----------|----------|------------|------------|
| Fixed Acidity            | 15.90000 | 4.60000  | 8.311111   | 1.746830   |
| Volatile Acidity         | 1.58000  | 0.12000  | 0.531339   | 0.179555   |
| Citric Acid              | 1.00000  | 0.00000  | 0.268364   | 0.196600   |
| Residual Sugar           | 15.50000 | 0.90000  | 2.532152   | 1.355324   |
| Chlorides                | 0.61100  | 0.01200  | 0.086933   | 0.047247   |
| Free Sulfur Dioxide      | 68.00000 | 1.00000  | 15.615486  | 10.246001  |
| Total Sulfur Dioxide     | 289.00000| 6.00000  | 45.914698  | 32.767787  |
| Density                  | 1.00369  | 0.99007  | 0.996730   | 0.001924   |
| pH                       | 4.01000  | 2.74000  | 3.311015   | 0.156596   |
| Sulphates                | 2.00000  | 0.33000  | 0.657708   | 0.170324   |
| Alcohol                  | 14.90000 | 8.40000  | 10.442111  | 1.081722   |
| Quality                  | 8.00000  | 3.00000  | 5.657043   | 0.805472   |


#### Distribution Visualised
![Fixed Acidity](figures/2/alcohol-wineQT.png)
![Volatile Acidity](figures/2/chlorides-wineQT.png)
![Volatile Acidity](figures/2/citric_acid-wineQT.png)
![Volatile Acidity](figures/2/density-wineQT.png)
![Volatile Acidity](figures/2/fixed_acidity-wineQT.png)
![Volatile Acidity](figures/2/pH-wineQT.png)
![Volatile Acidity](figures/2/quality-wineQT.png)
![Volatile Acidity](figures/2/residual_sugar-wineQT.png)
![Volatile Acidity](figures/2/sulphates-wineQT.png)
![Volatile Acidity](figures/2/total_sulfur_dioxide-wineQT.png)
![Volatile Acidity](figures/2/volatile_acidity-wineQT.png)

#### code 
```console
python3 -m assignments.3.2.1
```

#### 2.1.3
Used Label Encoder and Standard Scaler for the data and also handled missing data by filing the mean.

### 2.2 Model Building From Scratch

**BackProp Details**

Cost Function Derivative for backprop

$$J\left(a^{[3]} \mid y\right)=-y \log \left(a^{[3]}\right)-(1-y) \log \left(1-a^{[3]}\right)$$

BackProp Gradient equations

$$\begin{aligned} \frac{\partial J}{\partial \mathbf{W}^{[3]}} & =\frac{d J}{d a^{[3]}} \frac{d a^{[3]}}{d z^{[3]}} \frac{\partial z^{[3]}}{\partial \mathbf{W}^{[3]}} \\ \frac{\partial J}{\partial b^{[3]}} & =\frac{d J}{d a^{[3]}} \frac{d a^{[3]}}{d z^{[3]}} \frac{\partial z^{[3]}}{\partial b^{[3]}}\end{aligned}$$

Example of a three layered multi layer perceptron neural network.

Third Layer
$$
\begin{gathered}
d \mathbf{W}^{[3]}=\delta^{[3]} a^{[2] T} \\
d b^{[3]}=\delta^{[3]} \\
\delta^{[3]}=a^{[3]}-y
\end{gathered}
$$

Second Layer
$$
\begin{gathered}
d \mathbf{W}^{[2]}=\boldsymbol{\delta}^{[2]} \mathbf{a}^{[1] T} \\
d \mathbf{b}^{[2]}=\boldsymbol{\delta}^{[2]} \\
\boldsymbol{\delta}^{[2]}=\mathbf{W}^{[3] T} \delta^{[3]} * g^{\prime}\left(\mathbf{z}^{[2]}\right)
\end{gathered}
$$

First Layer
$$
\begin{gathered}
d \mathbf{W}^{[1]}=\boldsymbol{\delta}^{[1]} \mathbf{x}^T \\
d \mathbf{b}^{[1]}=\boldsymbol{\delta}^{[1]} \\
\boldsymbol{\delta}^{[1]}=\mathbf{W}^{[2] T} \delta^{[2]} * g^{\prime}\left(\mathbf{z}^{[1]}\right)
\end{gathered}
$$

Another important thing to note is that the output is a softmax function becuase we need to output probabilities of and then classify to the one which has the highest probabilities and the sum of all these should be 1 necessitating the use of a softmax function.

An interesting thing to note is that taking the derivative of the loss as both mse and binary cross entropy have given very similar results which is not surprising at the very least.

### code
```console
python3 -m assignments.3.2.2
```
The Performance of this arbitraty MLP classifier can be found in this WandB run :

[MLP Classifier on WineQT](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP/runs/82zs0iah)
### 2.3 Hyperparameter Tuning

The Metrics Logged are as follows
---
| Metric       | Value                  |
|--------------|------------------------|
| Accuracy     | 0.8699                 |
| Precision    | 0.3374                 |
| Recall       | 0.2778                 |
| F1-score     | 0.3047                 |
---

### 2.3 Hyperparameter Tuning

The Hyperparameter tuning for our mlp classifier has been done by initialising a WandB sweep which sweeps over the various hyperparameters so as to maximize validation accuracy

for the sake of simplicity the exact sweep dashboard is linked here,

[Classification Sweep Dashboard](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP/sweeps/cbud6h4k/workspace)

The various tested hyperparameters and the resulting validation accuracy can be visualised by this parallel axis chart :

![classification sweep](figures/classification_sweep.png)

### code
```console
python3 -m assignments.3.2.3
```

### 2.4 Best Model Evaluation
The best model from the hyperparameter tuning section is now ran and evaluated here:

### code
```console
python3 -m assignments.3.2.4
```
the performance of this run can be visualised in this WandB dashboard

[best classifier](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP/runs/x0k3kdeg)

| Parameter         | Value                           |
|-------------------|---------------------------------|
| **lr**            | 0.036847                        |
| **batch_size**    | 512                             |
| **epoch**         | 2000                            |
| **optimizer**     | mini-batch                      |
| **loss_fn**       | mlp.loss.CELoss()              |
| **activation**    | mlp.activations.Linear()        |
| **type**          | single_label_classification      |
| **early_stopping**| True                            |
| **patience**      | 512                             |


The Performance of the model is as follows:

## Test Set Metrics

- **Accuracy:** 0.8655
- **Precision:** 0.3076
- **Recall:** 0.2611
- **F1-score:** 0.2824


### 2.5 Hyperparameter vs loss testing

In this section we have swept over each individual hyperparameter and have observed its effects on the loss of the model

The various parameters tested are:

| Parameter            | Values / Range                                  |
|----------------------|-------------------------------------------------|
| **batch_size**       | [32, 64, 128, 256]                            |
| **epoch**            | [1000]                                         |
| **loss_fn**          | ["CELoss"]                                     |
| **activations**      | ["Relu", "Sigmoid", "Tanh"]                   |
| **type**             | ["single_label_classification"]                 |
| **lr**               | [0.1, 0.01, 0.001, 0.0001]                     |
| **model_architecture**| ["arch1"]                                      |

The loss logged by each of these 64 runs can be visualised in this WandB dashboard:

[hyperparam-tuning](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP/sweeps/jcu1wj7v/workspace)

Since we are concerned with the loss in the validation set (and by extension the test set) 
the chart has been included here
![val loss](figures/val_loss.png)

### code
```console
python3 -m assignments.3.2.4
```

### Multilabel Classification

In this section we use the advertisements.csv dataset to perform multilabel classification

### code
```console
python3 -m assignments.3.2.6
```

The performance of the run can be visualised in this WandB dashboard:

[multilabel](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP-MultiLabel/runs/9o5agp16)

The performance on the test set of our multilabel classifier is as follows:

### Test Set Metrics

- **Accuracy:** 0.6638
- **Precision:** 0.0851
- **Recall:** 0.1444
- **F1-score:** 0.1071
- **Hamming Loss:** 0.3363

### 2.7 Analysis
Most of the labels are between 2 3 and 4 as obserevd in the distribution The model does well in classifying between 2 3 and 4 but it's perfomance is poor for classifying into the other classes as the data available for those classes is very less


## 3. MLP Regression

### 3.1 Data Preprocessing 

#### 3.1.1 Statistics
```console
python3 -m assignments.3.3.1
```
The HousingData.csv dataset can be described as follows:

| Feature   | Maximum   | Minimum  | Mean       | Std Dev   |
|-----------|-----------|----------|------------|-----------|
| **CRIM**  | 88.9762   | 0.00632  | 3.611874   | 8.537322  |
| **ZN**    | 100.0000  | 0.00000  | 11.211934  | 22.898391 |
| **INDUS** | 27.7400   | 0.46000  | 11.083992  | 6.692542  |
| **CHAS**  | 1.0000    | 0.00000  | 0.069959   | 0.249986  |
| **NOX**   | 0.8710    | 0.38500  | 0.554695   | 0.115763  |
| **RM**    | 8.7800    | 3.56100  | 6.284634   | 0.701923  |
| **AGE**   | 100.0000  | 2.90000  | 68.518519  | 27.412339 |
| **DIS**   | 12.1265   | 1.12960  | 3.795043   | 2.103628  |
| **RAD**   | 24.0000   | 1.00000  | 9.549407   | 8.698651  |
| **TAX**   | 711.0000  | 187.00000| 408.237154 | 168.370495|
| **PTRATIO**| 22.0000  | 12.60000 | 18.455534  | 2.162805  |
| **B**     | 396.9000  | 0.32000  | 356.674032 | 91.204607 |
| **LSTAT** | 37.9700   | 1.73000  | 12.715432  | 7.005806  |
| **MEDV**  | 50.0000   | 5.00000  | 22.532806  | 9.188012  |

#### Visualising Distributions:
![age](figures/3/AGE-HousingData.png)
![age](figures/3/B-HousingData.png)
![age](figures/3/CHAS-HousingData.png)
![age](figures/3/CRIM-HousingData.png)
![age](figures/3/DIS-HousingData.png)
![age](figures/3/INDUS-HousingData.png)
![age](figures/3/LSTAT-HousingData.png)
![age](figures/3/MEDV-HousingData.png)
![age](figures/3/NOX-HousingData.png)
![age](figures/3/PTRATIO-HousingData.png)
![age](figures/3/RAD-HousingData.png)
![age](figures/3/RM-HousingData.png)
![age](figures/3/TAX-HousingData.png)
![age](figures/3/ZN-HousingData.png)

### 3.2 Model Building From Scratch

For performing the regresion task the main differences from the single label classification are as follow:
The Final Activation layer isnt softmax as the output is regression and shouldn't be probabilities, so it is a linear function thus changing one derivative term in the
backprop algorithm and the loss function is the Mean Square Error function.

### code
```console
python3 -m assignments.3.3.2
```
The Performance of this arbitraty MLP regressor can be found in this WandB run :

[MLP regressor on housing data](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP-regression/runs/ql3p14zo)

#### Test Set Metrics

- **R² Score:** 0.8400
- **MSE:** 10.1705
- **RMSE:** 3.1891
- **MAE:** 2.3978


### 3.3 Hyperparameter Tuning

The Hyperparameter tuning for our mlp regressor has been done by initialising a WandB sweep which sweeps over the various hyperparameters so as to maximize validation accuracy

for the sake of simplicity the exact sweep dashboard is linked here,

[regression Sweep Dashboard](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP-regression/sweeps/oty2r5m9/workspace)

The various tested hyperparameters and the resulting validation accuracy can be visualised by this parallel axis chart :

![classification sweep](figures/regrssion_sweep.png)

### code
```console
python3 -m assignments.3.3.3
```

### 3.4 Best Model Evaluation
The best model from the hyperparameter tuning section is now ran and evaluated here:

### code
```console
python3 -m assignments.3.3.4
```
the performance of this run can be visualised in this WandB dashboard

[best regressor](https://wandb.ai/autrio-das-international-institute-of-information-techno/SMAI-A3-MLP-regression/runs/en9mm4js)

and the test set metrics are as follows :

#### Test set Metrics 
| Metric      | Value               |
|-------------|---------------------|
| **R² Score**| 0.5540              |
| **MSE**     | 28.3602             |
| **RMSE**    | 5.3254              |
| **MAE**     | 3.4622              |


### 3.5 Comparision of MSE and BCE

### code
```console
python3 -m assignments.3.3.5
```

the loss vs epoch plot of MSE vs BCE loss in a binary classification task using an MLP regressor is as follows 
![MSEBCE](figures/3/MSEvsBCE.png)

Note here that the last layer of our regressor is a sigmoid fucntion

### 3.6 Analysis

There is difference we can see in the rate of convergence the mse convergence is much quicker and decays faster as compared to that of the bce loss which looks
more logarithmic than that of the the BCE loss plot.
The mse is high when the data stops following a certain patern Also the mse appears shoots up when the peaks are too close together in the sequence of data

## 4. AutoEncoders

### 4.1 Implementation
For the AutoEncoder we make a multi layer output mlp regressor with the input and output dimensions equal to the dimensions of the present dataset and the central
layer is the latent layer which learns the reduced dataset In out case it is 17 -> 16 -> 8 -> 7 -> 8 -> 16 -> 17 where 7 is latent layer containing reduced data. 

### 4.2 Training

### code
```console
python3 -m assignments.3.4.2
```

The model has been trained on the spotify.csv dataset and the model is stored in autoencoder.pkl

![autoencoedr](figures/autoencoder.png)

The reconstruction error is about 28%

### 4.3 Autoencoder with KNN:
### code
```console
python3 -m assignments.3.4.3
```
The model performs as such with an encoded dataset

#### Metrics

- **Accuracy:** 0.0673
- **Precision:** 0.0509
- **Recall:** 0.0536
- **F1-score:** 0.0518

Barebones KNN performs as such 
## Metrics

- **Accuracy:** 0.3675
- **Precision:** 0.3625
- **Recall:** 0.3582
- **F1-score:** 0.3579

This shows the same trend that knn showed with dimentionality reduction using PCA



### 4.3 Autoencoder with MLP:
### code
```console
python3 -m assignments.3.4.4
```

With the encoded dataset the mlp classifier performs as such
![mlp encoded](figures/mlp_encoded.png)

