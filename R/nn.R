library(doMC)
registerDoMC(cores = detectCores() - 1)
library(plyr)
library(caret)

print('loading data...')
n_pixels <- 784
col.types <- c('character', rep('integer', n_pixels))
data.trainraw <- read.csv('data/train.csv',
                       colClasses = col.types)
data.train <- data.trainraw

data.train$label <- factor(make.names(data.train$label))

# REMOVE LATER: limit while working through syntax
#print('trimming down to 5000 records while in dev...')
#data.train <- data.train[1:2500,]

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

print('training model...')
tg <- expand.grid(.size = c(floor(25)))

tr.ctrl <- trainControl(method = "repeatedcv",
                        repeats = 3,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)

X <- data.matrix(data.train[, -1])
y <- data.train[, 1]

model <- train(X,
               y,
               method = "mlp",
               metric = "ROC",
               tuneGrid = tg,
               trControl = tr.ctrl)

print(model)

print('applying model...')
XT <- data.matrix(data.test[, -1])
data.test$output <- predict(model, XT)

print('summarizing results...')
results <- caret::confusionMatrix(data.test$output, data.test$label)
print(results)
write(paste('Accuracy', results$overall[['Accuracy']]), file = "R/output/results.txt")
