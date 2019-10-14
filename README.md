# bovs2
The aim of bovs2 package is to implement the **Bagging on the Validation Set** to deep learning models trained in [Keras R](https://keras.rstudio.com/).

## Main idea of the algorithm
The main idea behind **bovs** algorithm is instead of doing bagging on the training data which is a common practice in modern statistics, such as random forest; **bovs** is applied to the validation data, greatly improving the speed of the algorithm, as the time required to do the validation is much less than the time to do the training in neural network. By making use of keras R package, we develop bovs2, can be used together with multiple models trained in Keras R.

## Key Functions
`baggingVal`: Function to do the bagging on the validation set

`baggingTest`: Function to implement return value of `baggingVal` to new data to study how the test result will look like

`performanceEvaluate`: Function to evaluate the performance of the ensembled model trained by `baggingVal` and tested by `baggingTest`

## Installation
To obatin the latest version of the `bovs2` package, access the following site
```
#install.packages("devtools")
devtools::install_github("CHuanSite/bovs2")
```

