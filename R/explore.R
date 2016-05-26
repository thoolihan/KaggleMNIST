library(ggplot2)

n_pixels <- 784
col.types <- c('factor', rep('integer', n_pixels))
data.train <- read.csv('data/train.csv',
                       colClasses = col.types)

g <- ggplot(data = data.train, aes(x = label)) +
      geom_bar(fill = '#33bbff')

print(g)