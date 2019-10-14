#' baggingVal function
#'
#' This function allows you to do bagging on the validation set
#' @param model the list of deep learning models object returned by model in keras R
#' @param x_val a dataframe of features to be used as validation set
#' @param y_val a vector indiates what is the label
#' @param iteration the total number of bootstrap to be used, the default value is 1000
#' @import keras tidyverse
#' @export

baggingVal <- function(model_list, x_val, y_val, iteration = 1000){
  #How many samples are used in the validation dataset
  val_size = nrow(x_val)

  #Number of candidate models
  model_num = length(model_list)

  #A list to store the prediction result for each model on the validation set
  val_result = list()

  #To store the result of the prediciton on the validation set temporarily
  temp = NULL

  #The performance of each model on the validation set
  for(i in 1 : model_num){
    temp = (model_list[[i]] %>% predict(x_val))

    #Deal with the binary classification problem seperately
    if(ncol(temp) == 1){
      temp = cbind(1 - temp, temp)
    }

    #Store the validation result in the val_result list
    val_result[[i]] = apply(temp, 1, which.max) - 1
  }

  #Store the result for every bootstrap sample
  res = NULL
  for(i in 1 : iteration){
    #The index of data being bootstrapped
    index = sample(val_size, val_size, replace = TRUE, prob = NULL)

    #To store the validation result on the bootstrapped dataset
    eva_result = NULL
    for(j in 1 : model_num){
      eva_result[j] = sum(val_result[[j]][index] == y_val[index])
    }
    res[i] = which.max(eva_result)
  }
  return(res)
}




#'baggingTest function
#'
#'This function allows you to make use of the training result obtained by bagginVal to test the performance of new model
#'
#' @param res A vector obtained by the baggingVal function,  the length is equal to the number of bootstrap done by the baggingVal function
#' @param model_list The list is same as the one used in baggingVal function to produce res
#' @param x_test A dataframe contains the features used by the bagginTest function
#' @param y_test A vector contains the labels used by baggingTest function
#' @import keras tidyverse
#' @export

baggingTest <- function(res, model_list, x_test, y_test){
  ## The start point and end point of the res used in baggingVal
  start_pos = 1
  end_pos = length(res)

  #The matrix format to store all the test result for the candidate models
  result = NULL

  #Store the test result on all the candidate models
  result_test = NULL

  #Number of candidate models
  model_num = length(model_list)

  #To store the validation result temporarily
  temp = NULL

  #The number of samples in the test set
  test_size = nrow(x_test)

  for(i in 1 : model_num){
    #The ith model's performance on the test dataset
    temp = model_list[[i]] %>% predict(x_test)

    #Deal with the binary classification problem seperately
    if(ncol(temp) == 1){
      temp = cbind(1 - temp, temp)
    }

    result_test[[i]] = apply(temp, 1, which.max) - 1
  }

  #Compute the predict result for the all the model obtained from the bootstrap samples
  for(i in 1 : iteration){
    result = cbind(result, result_test[[res[i]]])
  }

  #Store the prediction result for each test sample on all the models
  result_freq = NULL

  #Store the accuracy for each iteration of sample
  bagging_store = NULL

  #Count how many bagging have been done
  count = 1

  #Initialize the list to store the cumulative result
  list_table_temp = list()
  for(i in 1 : test_size){
    list_table_temp[[i]] = table(c(0 : (max(y_test)) ))
  }

  #The loop to study the performance of bagging models with respect to different numbers of Bootstrap samples
  for(k in start_pos : end_pos){

    #Update the table of the models chosen
    for(i in 1 : test_size){
      list_table_temp[[i]][result[i, k] + 1] = list_table_temp[[i]][result[i, k] + 1] + 1
      result_freq[i] = names(list_table_temp[[i]])[which.max(list_table_temp[[i]])]
      #print(i)
    }

    #The result of the ensemble when count number of individual models are trained
    bagging_store[count] = sum((as.numeric(result_freq)) == y_test) / nrow(x_test)

    #Update the number of candidate models used
    count = count + 1
  }
  return(bagging_store)
}

#' Performance evaluation function
#'
#' This function is used to evaluate the performance of a series of models trained by R keras
#' @param model_list A list of neural network models object returned by keras R
#' @param x_test A dataframe contains the features used by the performanceEvaluate function
#' @param y_test A vector contains the labels used by baggingTest function
#' @import keras tidyverse
#' @export

performanceEvaluate <- function(model_list, x_test, y_test){
  ## The start point and end poitn of the evaluation procedure
  start_pos = 1
  end_pos = length(model_list)

  #Number of the candidate models
  model_num = length(model_list)

  #List to store the individual performance of each individual model
  result_comp = NULL

  #The number of model being tested
  count = 1
  for(r in start_pos : end_pos){
    result_comp[count] = (model_list[[r]] %>% evaluate(x_test, y_test, verbose = 0))$acc
    count = count + 1
  }
  return(result_comp)
}
