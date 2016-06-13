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