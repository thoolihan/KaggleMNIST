library(doMC)
registerDoMC(cores = detectCores() - 1)
library(caret)
library(nnet)

source('./R/downsize.R')
source('./R/load.R')
data.train <- data.raw

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

nToMatrix <- function(v) {
  m <- matrix(0, nrow = length(v), ncol = length(unique(v)))
  for(i in 1:length(v)) { m[i, v[i]] = 1 }
  m
}

X <- data.matrix(data.train[, -1])
y <- nToMatrix(data.train[, 1])

print('training model...')

model <- nnet(x = X,
              y = y,
              data = data.train,
              decay = 4.75,
              size = 4,
              MaxNWts = 5000,
              softmax = TRUE)

print('applying model...')
XT <- data.matrix(data.test[, -1])
data.test$output <- max.col(predict(model, XT))

print('summarizing results...')
results <- caret::confusionMatrix(data.test$output, data.test$label)
print(results)
