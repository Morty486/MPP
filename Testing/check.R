setwd("/Users/ayakaouyang/Documents/GLMpp/2025/GLMpp/GLMpp")
Rcpp::compileAttributes()
devtools::load_all()



# 1
V <- matrix(0, 3, 3)
Lvec <- 1:6
makeLowTriMat(V, Lvec)



# 2

A <- matrix(1:9, 3, 3)

LowTriVec(A)


# 3
mu <- c(10,20,30,40,50)
p_z_vec <- c(2,3)

vec_to_field(mu, p_z_vec)




