# create an .RData file that is ready for use

if(!exists('data.raw')) { 
  if(file.exists(".RData")) {
    load(".RData") 
  } else {
    n_pixels <- 784
    col.types <- c('integer', rep('integer', n_pixels))
    data.raw <- read.csv('data/train.csv', colClasses = col.types)
    save(data.raw, file = ".RData", compress = TRUE)
  }
}