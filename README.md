# bovs2
The aim of bovs2 package is to implement the **Bagging on the Validation Set(bovs)** to deep learning models trained in [Keras R](https://keras.rstudio.com/).

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
## Example
Load **mnistData** data into R 
```
load(mnistData)
x_train = mnistData$x_train
y_train = mnistData$y_train
x_val = mnistData$x_val
y_val = mnistData$y_val
x_test = mnistData$x_test
y_test = mnistData$y_test
```

Reshape and rescale data
```
# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_val <- array_reshape(x_val, c(nrow(x_val), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_val <- x_val / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_val <- to_categorical(y_val, 10)
y_test <- to_categorical(y_test, 10)
```

Train models
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

Apply bovs to the data
```
valResult = baggingVal(list(model1, model2, model3), x_val, y_val)
testResult = baggingTest(valResult, list(model1, model2, model3), x_test, y_test, iteration = 1000)
```

Compare the results among three different models
```
y_test = apply(y_test,1, which.max) - 1
sum(testResult == y_test)
sum(model1 %>% predict_class(x_test) == testResult)
sum(model2 %>% predict_class(x_test) == testResult)
```

