library(doMC)
registerDoMC(cores = detectCores() - 1)
library(plyr)
library(caret)
library(nnet)

source('./R/downsize.R')
source('./R/load.R')

data.train <- data.raw
data.train$label <- factor(make.names(data.train$label))

# REMOVE LATER: limit while working through syntax
data.train <- data.train[sample(nrow(data.train), 1e4),]

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

X <- matrix.downsize(data.train[, -1])
X <- data.matrix(X)
y <- data.train[, 1]

print('training model...')
tg <- expand.grid(.size = c(4),
                  .decay = c(4.75))

tr.ctrl <- trainControl(method = "cv",
                        number = 3,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)

model <- train(X,
               y,
               method = "nnet",
               tuneGrid = tg,
               preProcess = c('center', 'scale'),
               trControl = tr.ctrl)

print(model)

print('applying model...')
XT <- matrix.downsize(data.matrix(data.test[, -1]))
data.test$output <- predict(model, XT)

print('summarizing results...')
results <- caret::confusionMatrix(data.test$output, data.test$label)
print(results)
