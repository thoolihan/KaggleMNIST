library(doMC)
registerDoMC(cores = detectCores() - 1)
library(caret)
library(nnet)

source('./R/downsize.R')
source('./R/load.R')
data.train <- data.raw
data.train$label <- factor(make.names(data.train$label))

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

X <- data.matrix(data.train[, -1])
y <- data.train[, 1]

print('training model...')
tg <- expand.grid(.size = c(4),
                  .decay = c(4.75))

tr.ctrl <- trainControl(method = "cv",
                        number = 2,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)

model <- train(X,
               y,
               method = "nnet",
               tuneGrid = tg,
               preProcess = c('center', 'scale'),
               trControl = tr.ctrl,
               softmax = TRUE)

print(model)

print('applying model...')
XT <- data.matrix(data.test[, -1])
data.test$output <- predict(model, XT)

print('summarizing results...')
results <- caret::confusionMatrix(data.test$output, data.test$label)
print(results)
