library(doMC)
registerDoMC(cores = detectCores() - 1)
library(plyr)
library(caret)
library(nnet)

n_pixels <- 784
col.types <- c('character', rep('integer', n_pixels))
raw <- read.csv('data/train.csv', colClasses = col.types)
raw$label <- factor(make.names(raw$label))

# REMOVE LATER: limit while working through syntax
data.train <- raw[sample(nrow(raw), 21000),]

print('splitting data...')
tri <- createDataPartition(data.train$label, p = .75, list = FALSE)
data.test <- data.train[-tri,]
data.train <- data.train[tri,]

matrix.downsize <- function(M, factor = 2) {
  len <- as.integer(sqrt(dim(M)[2]))
  steps <- seq(1, len, factor)
  pixels <- c()
  for(row in steps) {
    for (col in steps) {
      pixel_n <- row * len + col
      pixels <- c(pixels, pixel_n)
    }
  }
  M[, pixels]
}

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
