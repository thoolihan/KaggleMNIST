library(doMC)
registerDoMC(cores = 6)
library(plyr)
library(caret)

print('loading data...')
n_pixels <- 784
col.types <- c('character', rep('integer', n_pixels))
data.train <- read.csv('data/train.csv',
                       colClasses = col.types)
data.train$label <- factor(make.names(data.train$label))
data.train <- data.train[1:1000,]

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

# REMOVE LATER: limit while working through syntax
print('trimming down data size while in dev...')
data.train <- data.train[1:200,] 
data.test <- data.test[1:100,]

print('training model...')
tg <- expand.grid(.size = c(n_pixels))

tr.ctrl <- trainControl(method = "repeatedcv",
                        repeats = 3,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)

X <- data.train[, -1]
y <- data.train[, 1]

model <- train(X,
               y,
               method = "mlp",
               metric = "ROC",
               tuneGrid = tg,
               trControl = tr.ctrl)

print(model)

print('applying model...')
data.test$output <- predict(model, data.test)

print('summarizing results...')
print(confusionMatrix(data.test$output, data.test$label))
