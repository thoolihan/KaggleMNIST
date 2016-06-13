# create an .RData file that is ready for use

if(!exists('data.raw')) { 
  if(file.exists(".RData")) {
    load(".RData") 
    print('loaded data.raw from .RData')
  } else {
    n_pixels <- 784
    col.types <- c('integer', rep('integer', n_pixels))
    data.raw <- read.csv('data/train.csv', colClasses = col.types)
    print('read data.raw from csv')
    save(data.raw, file = ".RData", compress = TRUE)
    print('saved data.raw to .RData')
  }
} else {
  print('data.raw already exists')
}