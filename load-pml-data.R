library(psych)
naStrings = na.strings = c("NA", "", "NaN", "#DIV/0!")


loadCleanPmlData = function(type) {
    filename = ifelse(type == "testing", "pml-testing.csv", "pml-training.csv")
    pml.data = read.csv(filename, na.strings = naStrings)
    overviewData(pml.data)
    pml.data$cvtd_timestamp = as.POSIXct(pml.data$cvtd_timestamp, format =
                                             "%m/%d/%Y %H:%M")
    pml.data[, getNotCompleteMissingColumns(pml.data)]
}

getColMissingRatio = function(dataInput) {
    missingRatioColsRatio = apply(is.na(dataInput), 2, mean)
    missingRatioColsRatio
}

#Get columns in data framework where data is NOT completely missing
getNotCompleteMissingColumns = function(dataInput) {
    missingRatioColsRatio = getColMissingRatio(dataInput)
    names(missingRatioColsRatio[missingRatioColsRatio < 1])
}

#Find columns that have missing values
getColumnsWithMissingValues = function(dataInput) {
    missingRatioColsRatio = getColMissingRatio(dataInput)
    names(missingRatioColsRatio[missingRatioColsRatio > 0])
}

#Get predictors cols, which is intersect of training and testing data and remove non-predictors 
getPredictorFeatures = function(trainingData, testingData) {
    predictorColsTest = intersect(names(pml.data.testing), names(pml.data.training.complete))
    nonPredictiveFeatures = c("X","user_name","cvtd_timestamp","raw_timestamp_part_1","raw_timestamp_part_2","new_window","num_window")
    setdiff(predictorColsTest, nonPredictiveFeatures)
}



#Quick view of data
overviewData = function(data) {
    dim(data)
    summary(data)
    str(data)
}

library(dplyr)
library(caret)
library(mice)
library(Hmisc)
#Handle missing data, which are critical for later model training
imputeData = function(dataInput) {
    #there is no need to impute timestamp
    dataInput = select(dataInput, -cvtd_timestamp)
    dataInput = dataInput[,getColumnsWithMissingValues(dataInput)]
    
    #first we notice there are some all 0 vars, these can simply by impute with mean
    dataInput$amplitude_yaw_belt = impute(dataInput$amplitude_yaw_belt, mean)
    dataInput$amplitude_yaw_dumbbell = impute(dataInput$amplitude_yaw_dumbbell, mean)
    dataInput$amplitude_yaw_forearm = impute(dataInput$amplitude_yaw_forearm, mean)
    
    mice_impute = mice(dataInput, m = 1, maxit = 2)
    dataComplete = complete(mice_impute, 1)
    #we notice there are still 6 colums with missing value
    dataInput[,names(dataComplete)] = dataComplete
    
    mice_impute = mice(dataComplete[,getColumnsWithMissingValues(dataComplete)], m = 1, maxit = 3)
    dataComplete = complete(mice_impute, 1)
    dataInput[,names(dataComplete)] = dataComplete
    
    #the last 3 vars, the hardest to crack, we need to leverage know, simply using mean
    dataInput$min_yaw_belt = impute(dataInput$min_yaw_belt, min)
    dataInput$min_yaw_dumbbell = impute(dataInput$min_yaw_dumbbell, min)
    dataInput$min_yaw_forearm = impute(dataInput$min_yaw_forearm, min)
    
    dataInput
}
