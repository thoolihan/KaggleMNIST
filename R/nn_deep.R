library(doMC)
registerDoMC(cores = detectCores() - 1)
library(caret)
library(nnet)

source('./R/downsize.R')
source('./R/load.R')

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

X <- data.matrix(data.train[, -1])
y <- data.train[, 1]

print('training model...')
tg <- data.frame(.size = c(4),
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
XT <- data.matrix(data.test[, -1])
data.test$output <- predict(model, XT)

print('summarizing results...')
results <- caret::confusionMatrix(data.test$output, data.test$label)
print(results)
