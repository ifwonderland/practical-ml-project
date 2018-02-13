#libraries
library(caret)
library(dplyr)

#load and some basic cleaning of data
source("load-pml-data.R")
pml.data.training.complete = loadCleanPmlData("training")
overviewData(pml.data.training.complete)
pml.data.testing = loadCleanPmlData("testing")
overviewData(pml.data.testing)

getColMissingRatio(pml.data.training.complete)

#Get only predictor features 
predictorCols = getPredictorFeatures(pml.data.training.complete, pml.data.testing)
pml.data.training.raw = pml.data.training.complete[, predictorCols]
pml.data.training.raw$classe = pml.data.training.complete$classe
pml.data.testing = pml.data.testing[,predictorCols]

#handling missing data
#first we look at how many data is missing
getColMissingRatio(pml.data.training.raw)


#data splitting
set.seed(32455) #for reproduction
inTrain = createDataPartition(pml.data.training.raw$classe,
                              p = 3 / 4,
                              list = FALSE)
pml.data.training = pml.data.training.raw[inTrain, ]
pml.data.validation = pml.data.training.raw[-inTrain, ]

## Model fitting, cross validation and testing
source("model-training.R")

### Stats model based fitting
pml.data.training.linear = getLinearData(pml.data.training)
pml.data.validation.linear = getLinearData(pml.data.validation)
pml.data.testing.linear = getLinearData(pml.data.testing)

#preprocess, now we can impute since we have very low missing data in these features, imputatin also center and scale data, making analysis easier as well, we also transform these features with BoxCox for normality
preproc = preProcess(
    select(pml.data.training.linear,-classe),
    method = c("pca", "center", "scale")
)

pml.data.training.linear = transformPCData(preproc, pml.data.training.linear)
pml.data.validation.linear = transformPCData(preproc, pml.data.validation.linear)
pml.data.testing.linear = transformPCData(preproc, pml.data.testing.linear)

#Naive Bayesian model
pml.model.nb = trainNativeBayes(pml.data.training.linear)
#evaluation of models
confusionMatrix(
    predict(pml.model.nb, pml.data.validation.linear),
    pml.data.validation.linear$classe
)
#accuracy 0.6366, Kappa: 0.5412
predict(pml.model.nb, newdata = pml.data.testing.linear)
#C A A A C C D D A A A C B A E B A E B B

#Linear Discrimitive Analysis
pml.model.lda = trainLDA(pml.data.training.linear)
confusionMatrix(
    predict(pml.model.lda, pml.data.validation.linear),
    pml.data.validation.linear$classe
)
#accuracy 0.5275, Kappa: 0.4019
predict(pml.model.lda, newdata = pml.data.testing.linear)
#D A A A A C D D A C A A E A E B A D A B

#SVM
pml.model.svm = trainSVM(pml.data.training)
confusionMatrix(predict(pml.model.svm, pml.data.validation),
                pml.data.validation$classe)
#Accuracy : 0.7804, Kappa : 0.7207
predict(pml.model.svm, newdata = pml.data.testing)
#C A B C A E D D A A C A B A E E A B B B

###Non-linear models
#Quadratic Discrimptive Analysis
pml.model.qda = trainQDA(pml.data.training.linear)
confusionMatrix(
    predict(pml.model.qda, pml.data.validation.linear),
    pml.data.validation.linear$classe
)
# Accuracy : 0.7412, Kappa : 0.6762
predict(pml.model.qda, newdata = pml.data.testing.linear)
#C C C A A B D B A A A C B A E E A B B B

### Tree models
#tree based models does not require too much preprocessing, we are just going to use raw training data
#Decision Tree
pml.model.rpart = trainCART(pml.data.training)
confusionMatrix(
    predict(pml.model.rpart, newdata = pml.data.validation),
    pml.data.validation$classe
)
#Accuracy 0.8336 and Kappa is 0.7896 as well
predict(pml.model.rpart, newdata = pml.data.testing)
#B A C A A E D A A A C C B A E E A B B B

#Boostings 
#Random Forest
pml.model.rf = trainRF(pml.data.training)
confusionMatrix(
    predict(pml.model.rf, newdata = pml.data.validation),
    pml.data.validation$classe
)
#accuracy 0.9923, kappa 0.9902
predict(pml.model.rf, newdata = pml.data.testing)
#B A B A A E D B A A B C B A E E A B B B

### Neural Network
pml.model.nn = trainNN(pml.data.training)
confusionMatrix(
    predict(pml.model.nn, newdata = pml.data.validation),
    pml.data.validation$classe
)
#accuracy 0.4144, kappa 0.2718
predict(pml.model.nn, newdata = pml.data.testing)
#C A B C A C D B A A C C D A D E C B B B

### Ensembling
#Adaboost
pml.model.adaboost = trainAdaBag(pml.data.training)
confusionMatrix(
    predict(pml.model.adaboost, newdata = pml.data.validation),
    pml.data.validation$classe
)
#accuracy 0.4323, kappa: 0.2288
predict(pml.model.adaboost, newdata = pml.data.testing)


#GBM
pml.model.gbm = trainGBM(pml.data.training)
confusionMatrix(predict(pml.model.gbm, newdata = pml.data.validation),
                pml.data.validation$classe)
#accurary 0.9615, kappa 0.9512
predict(pml.model.gbm, newdata = pml.data.testing)
#B A B A A E D B A A B C B A E E A B B B

## Predictions and error discussion


## Conclusion
#Overall, Random Forest and GBM seems to be giving the best prediction. 

