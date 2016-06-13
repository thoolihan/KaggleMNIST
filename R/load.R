# create an .RData file that is ready for use

if(!exists('raw')) { 
  if(file.exists(".RData")) {
    load(".RData") 
  } else {
    n_pixels <- 784
    col.types <- c('character', rep('integer', n_pixels))
    raw <- read.csv('data/train.csv', colClasses = col.types)
    raw$label <- factor(make.names(raw$label))
    save(raw, file = ".RData", compress = TRUE)
  }
}