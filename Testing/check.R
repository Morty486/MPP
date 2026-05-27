

setwd("~/Documents/MPP")

# save all files first in RStudio
Rcpp::compileAttributes()

devtools::clean_dll()
devtools::load_all(reset = TRUE, recompile = TRUE)
library(MPP)





##############
##### 1 ######
##############
V <- matrix(0, 3, 3)
Lvec <- 1:6
makeLowTriMat(V, Lvec)



##############
##### 2 ######
##############

A <- matrix(1:9, 3, 3)

LowTriVec(A)




##############
##### 3 ######
##############


# Create two matrices
A <- matrix(c(1, 2,
              3, 4), nrow = 2, byrow = TRUE)

B <- matrix(c(5, 6, 7), nrow = 1)

# Create a list (R will convert to arma::field<arma::mat>)
M <- list(A, B)

# Run Bdiag
Bdiag(M)



##############
##### 4 ######
##############

# =========================
# Example 1: symmetric positive definite matrix
# =========================

A <- matrix(c(4, 1,
              1, 3), nrow = 2, byrow = TRUE)

myinvCpp(A)

# Compare with R
solve(A)


# =========================
# Example 2: non-symmetric matrix
# inv_sympd() will fail,
# then arma::inv() will be used
# =========================

B <- matrix(c(1, 2,
              3, 4), nrow = 2, byrow = TRUE)

myinvCpp(B)
myinvCpp2(B)

# Compare with R
solve(B)


# =========================
# Example 3: singular matrix
# inv() fails,
# then pseudoinverse pinv() is used
# =========================

C <- matrix(c(1, 2,
              2, 4), nrow = 2, byrow = TRUE)

myinvCpp(C)

# Compare with MASS::ginv()
MASS::ginv(C)



##############
##### 5 ######
##############


# =========================
# Example 1:
# Positive definite matrix
# Normal Cholesky works
# =========================

A <- matrix(c(4, 1,
              1, 3), nrow = 2, byrow = TRUE)

myCholCpp(A)

myCholCpp2(A)

# Compare with R
chol(A)


# =========================
# Example 2:
# Not positive definite
# First chol() fails
# Then diagonal perturbation is added
# =========================

B <- matrix(c(1, 2,
              2, 1), nrow = 2, byrow = TRUE)

eigen(B)$values

myCholCpp(B)
myCholCpp2(B)

# =========================
# Example 3:
# Singular matrix
# Even perturbed chol may fail
# =========================

C <- matrix(c(0, 0,
              0, 0), nrow = 2)

# myCholCpp(C)
print("myCholCpp fails")

myCholCpp2(C)




##############
##### 6 ######
##############


library(MPP)

set.seed(123)

# -------------------------
# Simulate simple LME data
# y_ij = beta0 + beta1 * time_ij + b_i + error_ij
# -------------------------

n <- 5          # subjects
m <- 4          # observations per subject

beta_true <- c(2, 0.5)
sigma_b <- 1
sigma_e <- 0.3

weights <- rep(1, n)

Y <- vector("list", n)
X <- vector("list", n)
Z <- vector("list", n)

for(i in 1:n){
  time <- 0:(m - 1)

  X_i <- cbind(1, time)        # fixed effects: intercept + time
  Z_i <- matrix(1, m, 1)       # random intercept only

  b_i <- rnorm(1, 0, sigma_b)
  y_i <- X_i %*% beta_true + Z_i * b_i + rnorm(m, 0, sigma_e)

  Y[[i]] <- as.vector(y_i)
  X[[i]] <- X_i
  Z[[i]] <- Z_i
}

fit <- init_LME(
  weights = weights,
  Y = Y,
  X = X,
  Z = Z,
  maxiter = 100,
  eps = 1e-4
)

fit

fit$beta      # estimated fixed effects
fit$sig2      # estimated residual variance
fit$Sigma     # estimated random-effect covariance
fit$mu        # subject-specific random-effect means
fit$V         # subject-specific random-effect uncertainty

fit$Sigma
fit$V[[1]]



set.seed(123)

library(MASS)

# -----------------------------
# 1. Simulate toy data
# -----------------------------

n <- 200                     # number of subjects
weights <- rep(1, n)

delta_true <- c(2, 1.5)      # fixed effects: intercept and time slope
sig_true <- 0.2              # residual SD
sig2_true <- sig_true^2

Sigma_b_true <- matrix(
  c(0.50, 0.10,
    0.10, 0.30),
  nrow = 2,
  byrow = TRUE
)

W <- vector("list", n)
U <- vector("list", n)
Vdesign <- vector("list", n)

b_true <- MASS::mvrnorm(
  n = n,
  mu = c(0, 0),
  Sigma = Sigma_b_true
)

for (i in 1:n) {

  # random number of observations per subject
  mi <- sample(3:8, size = 1)

  # irregular observation times
  time_i <- sort(runif(mi, 0, 1))

  # fixed-effect design matrix U_i
  # columns: intercept, time
  U_i <- cbind(1, time_i)

  # random-effect design matrix V_i
  # also random intercept and random slope
  V_i <- cbind(1, time_i)

  # generate biomarker values
  mean_i <- U_i %*% delta_true + V_i %*% b_true[i, ]
  W_i <- as.numeric(mean_i + rnorm(mi, mean = 0, sd = sig_true))

  W[[i]] <- W_i
  U[[i]] <- U_i
  Vdesign[[i]] <- V_i
}

# -----------------------------
# 2. Add one subject with no observations
# -----------------------------
# This checks whether your empty-vector case works.

W[[1]] <- numeric(0)
U[[1]] <- matrix(numeric(0), nrow = 0, ncol = 2)
Vdesign[[1]] <- matrix(numeric(0), nrow = 0, ncol = 2)

# -----------------------------
# 3. Run init_LME2_one_marker()
# -----------------------------

fit <- init_LME2_one_marker(
  weights = weights,
  W = W,
  U = U,
  Vdesign = Vdesign,
  maxiter = 100,
  eps = 1e-5
)

# -----------------------------
# 4. Check results
# -----------------------------

fit$delta_k
delta_true

fit$sig2_k
sig2_true

fit$Sigma_bk
Sigma_b_true

# Check dimensions of variational/posterior quantities
length(fit$mu_b_ik)
length(fit$S_bb_ik)

fit$mu_b_ik[[2]]
fit$S_bb_ik[[2]]


fit$delta_k
# close to c(2, 1.5)

fit$sig2_k
# close to 0.04

fit$Sigma_bb_k
# roughly close to:
#      [,1] [,2]
# [1,] 0.50 0.10
# [2,] 0.10 0.30



# More patients have no observations on one biomarker

# -----------------------------
# 1. Simulate toy data
# -----------------------------

n <- 200
weights <- rep(1, n)

delta_true <- c(2, 1.5)
sig_true <- 0.2
sig2_true <- sig_true^2

Sigma_b_true <- matrix(
  c(0.50, 0.10,
    0.10, 0.30),
  nrow = 2,
  byrow = TRUE
)

W <- vector("list", n)
U <- vector("list", n)
Vdesign <- vector("list", n)

b_true <- MASS::mvrnorm(
  n = n,
  mu = c(0, 0),
  Sigma = Sigma_b_true
)

# proportion of subjects with no observations
missing_prob <- 0.35

missing_id <- rep(FALSE, n)

for (i in 1:n) {

  # Some subjects have no observations for this biomarker
  if (runif(1) < missing_prob) {

    W[[i]] <- numeric(0)
    U[[i]] <- matrix(numeric(0), nrow = 0, ncol = 2)
    Vdesign[[i]] <- matrix(numeric(0), nrow = 0, ncol = 2)

    missing_id[i] <- TRUE

  } else {

    # random number of observations per observed subject
    mi <- sample(3:8, size = 1)

    # irregular observation times
    time_i <- sort(runif(mi, 0, 1))

    # fixed-effect design: intercept + time
    U_i <- cbind(1, time_i)

    # random-effect design: random intercept + random slope
    V_i <- cbind(1, time_i)

    # generate observed biomarker values
    mean_i <- U_i %*% delta_true + V_i %*% b_true[i, ]
    W_i <- as.numeric(mean_i + rnorm(mi, mean = 0, sd = sig_true))

    W[[i]] <- W_i
    U[[i]] <- U_i
    Vdesign[[i]] <- V_i
  }
}

# Check how many subjects have no observations
table(missing_id)
mean(missing_id)

fit <- init_LME2_one_marker(
  weights = weights,
  W = W,
  U = U,
  Vdesign = Vdesign,
  maxiter = 100,
  eps = 1e-5
)


fit$delta_k
delta_true

fit$sig2_k
sig2_true

fit$Sigma_bb_k
Sigma_b_true


empty_subjects <- which(missing_id)

# First subject with no observations
i0 <- empty_subjects[1]


fit$mu_b_ik[[i0]]
# approximately c(0, 0)

fit$S_bb_ik[[i0]]
# equal or close to the initialized/estimated Sigma_bk




##############
##### 7 ######
##############

out <- test_MPP_para_t_basic()

names(out)

out$n
out$K

out$beta_time
out$delta
out$sig2
out$gamma
out$alpha_c
out$beta_c

out$Sigma_k
out$Sigma
out$invSigma
out$Sigma_times_invSigma
out$invSigma_k
out$Lmat

out$mu_ik
out$Z_ik
out$mu_i
out$Z_i
out$Lvec_i

out$mu_i_1_length
out$Z_i_1_dim
out$Lvec_i_1_length


out$beta_time[[1]]
out$beta_time[[2]]

out$delta[[1]]
out$delta[[2]]

out$Sigma_k[[1]]
out$Sigma_k[[2]]

out$mu_ik[[1, 1]]
out$mu_ik[[1, 2]]
out$mu_ik[[2, 1]]

out$Z_ik[[1, 1]]
out$Z_ik[[1, 2]]

out$mu_i[[1]]
out$Z_i[[1]]
out$Lvec_i[[1]]


length(out$beta_time)
length(out$delta)
length(out$alpha_c)
length(out$beta_c)
length(out$Sigma_k)

dim(out$Sigma)
dim(out$invSigma)
dim(out$Lmat)

length(out$mu_i[[1]])
dim(out$Z_i[[1]])
length(out$Lvec_i[[1]])




##############
##### 8 ######
##############


library(dplyr)

# -----------------------------
# 1. Toy baseline data
# -----------------------------
toy_base <- data.frame(
  subject_num = c(101, 102, 103),
  Age = c(54, 76, 65),
  Gender = c(0, 1, 0),
  BMI = c(28.2, 24.7, 31.0),
  In_hospital_death = c(0, 1, 0)
)

# -----------------------------
# 2. Toy long biomarker data
# -----------------------------
toy_long <- data.frame(
  subject_num = c(
    101,101,101,101,101,
    102,102,102,102,
    103,103,103,103,103
  ),
  Time = c(
    "00:07","00:07","00:37","00:37","01:37",
    "00:10","00:40","01:20","01:20",
    "00:05","00:35","00:35","01:10","01:45"
  ),
  Parameter = c(
    "HR","Temp","HR","Temp","HR",
    "HR","HR","Temp","GCS",
    "GCS","HR","Temp","HR","Temp"
  ),
  Value = c(
    73,35.1,77,35.6,60,
    90,95,38.0,12,
    15,80,36.5,85,37.0
  )
)

# -----------------------------
# 3. Create biomarker indicator
# -----------------------------
toy_codes <- unique(toy_long$Parameter)

toy_long <- toy_long %>%
  mutate(
    feature_id = match(Parameter, toy_codes)
  )

toy_long



# -----------------------------
# 4. Convert HH:MM to normalized time
# -----------------------------
toy_long <- toy_long %>%
  mutate(
    hour = as.numeric(substr(Time, 1, 2)),
    minute = as.numeric(substr(Time, 4, 5)),
    time = (hour * 60 + minute) / (48 * 60)
  )

# -----------------------------
# 5. Build nested objects
# -----------------------------
subjects <- toy_base$subject_num
K <- length(toy_codes)
n <- length(subjects)

W <- vector("list", n)
Time_list <- vector("list", n)
U <- vector("list", n)
Vdesign <- vector("list", n)

for (i in seq_len(n)) {

  id <- subjects[i]

  W[[i]] <- vector("list", K)
  Time_list[[i]] <- vector("list", K)
  U[[i]] <- vector("list", K)
  Vdesign[[i]] <- vector("list", K)

  for (k in seq_len(K)) {

    dat_ik <- toy_long %>%
      filter(subject_num == id, feature_id == k) %>%
      arrange(time)

    tvec <- dat_ik$time
    wvec <- dat_ik$Value

    W[[i]][[k]] <- wvec
    Time_list[[i]][[k]] <- tvec

    U[[i]][[k]] <- cbind(1, tvec)
    Vdesign[[i]][[k]] <- cbind(1, tvec)
  }
}

# -----------------------------
# 6. Baseline outcome/covariates
# -----------------------------
Y <- toy_base$In_hospital_death

Z <- as.matrix(
  cbind(
    intercept = 1,
    Age = toy_base$Age,
    Gender = toy_base$Gender,
    BMI = toy_base$BMI
  )
)

weights <- rep(1, n)

# -----------------------------
# 7. Final datalist
# -----------------------------
toy_datalist <- list(
  n = n,
  K = K,
  biomarker_names = toy_codes,

  Y = Y,
  Z = Z,
  weights = weights,

  W = W,
  Time = Time_list,
  U = U,
  Vdesign = Vdesign
)



# flatten nested list into 1D list
flatten_ik <- function(x, n, K) {

  out <- vector("list", n * K)

  iter <- 1

  for (i in seq_len(n)) {
    for (k in seq_len(K)) {

      out[[iter]] <- x[[i]][[k]]

      iter <- iter + 1
    }
  }

  out
}

toy_datalist_cpp <- toy_datalist

toy_datalist_cpp$W <-
  flatten_ik(toy_datalist$W, n, K)

toy_datalist_cpp$Time <-
  flatten_ik(toy_datalist$Time, n, K)

toy_datalist_cpp$U <-
  flatten_ik(toy_datalist$U, n, K)

toy_datalist_cpp$Vdesign <-
  flatten_ik(toy_datalist$Vdesign, n, K)

str(toy_datalist_cpp$W)




# -----------------------------
# 8. Create toy parameter list
# -----------------------------

# dimensions
q_a <- 2   # a_ik: random intercept + slope for measurement-time model
q_b <- 2   # b_ik: random intercept + slope for mark/value model
q_c <- q_a + q_b

p_delta <- 2   # fixed effects in mark model: intercept + time
p_gamma <- ncol(Z)

beta_time <- vector("list", K)
delta <- vector("list", K)
sig2 <- rep(1, K)

alpha_c <- vector("list", K)
beta_c <- vector("list", K)

Sigma_k <- vector("list", K)

for (k in seq_len(K)) {

  # fixed effects in time process
  # just toy values
  beta_time[[k]] <- c(0.1, 0.2)

  # fixed effects in mark/value model
  delta[[k]] <- c(1.0, 0.5)

  # residual variance for mark/value model
  sig2[k] <- 1.0

  # outcome coefficients for a_ik and b_ik effects
  alpha_c[[k]] <- c(0.1, 0.1)
  beta_c[[k]]  <- c(0.1, 0.1)

  # covariance for c_ik = (a_ik, b_ik)
  Sigma_k[[k]] <- diag(q_c)
}

gamma <- rep(0, p_gamma)

# variational parameters mu_ik and Z_ik
mu_ik_nested <- vector("list", n)
Z_ik_nested <- vector("list", n)

for (i in seq_len(n)) {

  mu_ik_nested[[i]] <- vector("list", K)
  Z_ik_nested[[i]] <- vector("list", K)

  for (k in seq_len(K)) {

    mu_ik_nested[[i]][[k]] <- rep(0, q_c)
    Z_ik_nested[[i]][[k]] <- diag(q_c)
  }
}

mu_ik <- flatten_ik(mu_ik_nested, n, K)
Z_ik <- flatten_ik(Z_ik_nested, n, K)

toy_paralist <- list(
  beta_time = beta_time,
  delta = delta,
  sig2 = sig2,
  gamma = gamma,
  alpha_c = alpha_c,
  beta_c = beta_c,
  Sigma_k = Sigma_k,
  mu_ik = mu_ik,
  Z_ik = Z_ik
)



test_model(toy_datalist_cpp, toy_paralist)







