# bovs2
The aim of bovs2 package is to implement the Bagging on the Validation Set to deep learning models trained in [Rkeras](https://keras.rstudio.com/).

## Key Functions
`baggingVal`: Function to do the bagging on the validation set

`baggingTest`: Function to implement return value of `baggingVal` to new data to study how the test result will look like

`performanceEvaluate`: Function to evaluate the performance of the ensembled model trained by `baggingVal` and tested by `baggingTest`

## Installation
To obatin the latest version of the `bovs2` package, access the following site

` install.packages("devtools")
  devtools::install_github("CHuanSite/bovs2")
`

