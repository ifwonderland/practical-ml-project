#linear model training related functions

#Get suitable features/vars for linear model fitting
getLowMissingDataCols  = function(data) {
    #Handling missing values, all analysis below are sensitive to missing values
    colMissingRatio = apply(is.na(data), 2, mean)
    #for linear analysis, if most of the values are missing, the feature won't be helpful, so we can simply filter out these
    names(colMissingRatio[colMissingRatio < 0.05])
}


#preprocess for linear model based 
getLinearData = function(dataInput) {
    #first, get rid of some vars we know are not helpful for linear modeling
    data.numeric = dataInput[,sapply(dataInput, is.numeric)]
    data.linear = data.numeric
    data.linear$classe = dataInput$classe
    data.linear
}

#quick helper method for sampling
sampleData = function(dataInput, sampleSize = 1000) {
    dataInput[sample(nrow(dataInput), sampleSize),]
}

#Transform to PC components
transformPCData = function(preproc, dataInput) {
    dataInput.processed  = predict(preproc, dataInput)
    dataInput.processed = dataInput.processed[,grepl("PC", names(dataInput.processed))]
    dataInput.processed$classe = dataInput$classe
    dataInput.processed
}

tc = trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10
)


#Naive Bayesian model
trainNativeBayes = function(trainingData) {
    #NB is very expensive to run, with large data set, it will take huge amount of time,
    train(
        classe ~ .,
        data = trainingData,
        method = "nb",
        trControl = tc
    )
}


#LDA : Linear Discrimitive Model
trainLDA = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "lda",
        trControl = tc
    )
}

#SVM: Linear
trainSVM = function(trainingData) {
    train(
        classe~.,
        data = trainingData,
        method = "svmLinear",
        trControl = tc
    )
}


#QDA: Quadratic Discrimitive Analysis
trainQDA = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "qda",
        trControl = tc
    )
}


#Classification Tree
trainCART = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "rpart",
        tuneLength = 30,
        trControl = tc
    )
}

#AdBoost classification tree
trainAdaBag = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "AdaBag",
        tuneLength = 3,
        trControl = trainControl(
            method = "repeatedcv",
            number = 5,
            repeats = 5
        )
    )
}

#Random Forest
trainRF = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "rf",
        trControl = trainControl(
            method = "repeatedcv",
            number = 3
        )
    )
}

#Neutral Network
trainNN = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "nnet",
        trace = TRUE,
        trControl = trainControl(
            method = "repeatedcv",
            number = 3,
            repeats = 3
        )
    )
}

#GBM
trainGBM = function(trainingData) {
    train(
        classe ~ .,
        data = trainingData,
        method = "gbm",
        trControl = tc
    )
}

