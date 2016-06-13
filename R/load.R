# create an .RData file that is ready for use

if(!exists('data.raw')) { 
  if(file.exists(".RData")) {
    load(".RData") 
  } else {
    n_pixels <- 784
    col.types <- c('character', rep('integer', n_pixels))
    data.raw <- read.csv('data/train.csv', colClasses = col.types)
    data.raw$label <- factor(make.names(data.raw$label))
    save(data.raw, file = ".RData", compress = TRUE)
  }
}