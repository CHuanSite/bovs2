# bovs2
The aim of bovs2 package is to implement the **Bagging on the Validation Set(bovs)** to deep learning models trained in [Keras R](https://keras.rstudio.com/).

## Main idea of the algorithm
The main idea behind **bovs** algorithm is instead of doing bagging on the training data which is a common practice in modern statistics, such as random forest; **bovs** is applied to the validation data, greatly improving the speed of the algorithm, as the time required to do the validation is much less than the time to do the training in neural network. By making use of keras R package, we develop bovs2, can be used together with multiple models trained in Keras R.

## Key Functions
`baggingVal`: Function to do the bagging on the validation set;

`baggingTest`: Function to implement return value of `baggingVal` to new data to study how the test result will look like;

`performanceEvaluate`: Function to evaluate the performance of the ensembled model trained by `baggingVal` and tested by `baggingTest`;

`sortMaxFequency`: Function to sort the label and find the one with highest frequency of appearence.

## Installation
To obatin the latest version of the `bovs2` package, access the following site
```
devtools::install_github("CHuanSite/bovs2")
```
## Example
Load packages `keras`, `bovs2`, `tiyverse`
```
library(keras)
library(bovs2)
library(tidyverse)
```

Load **mnistData** data into R, **mnistData** contains all the data needed to run this example, which is from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset
```
data(mnistData)
x_train = mnistData$x_train
y_train = mnistData$y_train
x_val = mnistData$x_val
y_val = mnistData$y_val
x_test = mnistData$x_test
y_test = mnistData$y_test
```

Train models in keras with different model architectures, a small amount of models is used in this example,
```
model_list = list()
## The structure of the model
unitsNumber = c(128, 256, 512)
unitsHiddenNumber = c(64, 32, 16)

index = 1
for(i in 1 : length(unitsNumber)){
  
  ## Train the model
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = unitsNumber[i], activation = "relu", input_shape = c(784)) %>% 
    layer_dense(units = unitsHiddenNumber[i], activation = "relu") %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(),
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(
    x_train, y_train, 
    epochs = 30, batch_size = 128
  )
  
  model_list[[index]] = model
  index = index + 1
}

```

Apply **bovs** to the data
```
valResult = baggingVal(model_list, x_val, y_val)
```

Output the test result after the `valResult`
```
testResult = baggingTest(valResult, model_list, x_test, y_test, iteration = 1000)
```

Compare the results among three different models
```
y_test = apply(y_test,1, which.max) - 1
sum(testResult == y_test)
sum(model_list[[1]] %>% predict_classes(x_test) == y_test)
sum(model_list[[2]] %>% predict_classes(x_test) == y_test)
sum(model_list[[2]] %>% predict_classes(x_test) == y_test)
```
