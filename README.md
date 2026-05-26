# MPP

An in-progress R/Rcpp package for developing a multi-marker marked point process joint model.



## Project Goal

The long-term goal is to implement a joint modeling framework for irregular EHR-type biomarker data. For subject `i` and biomarker `k`, the target model contains three linked parts:

1. A measurement-time point process submodel

```text
log lambda_ik(s) = x_i^T alpha_k + psi_1k(s)^T theta_k + psi_2k(s)^T a_ik
```

2. A longitudinal mark/value submodel

```text
w_ik(s) = u_ik(s)^T delta_k + v_ik(s)^T b_ik + error
```

3. A binary outcome submodel linked through shared random effects

```text
logit P(Y_i = 1 | c_i) = z_i^T gamma
                         + sum_k a_ik^T H_ik alpha_c,k
                         + sum_k b_ik^T G_ik beta_c,k
```

where `c_ik = (a_ik, b_ik)` and the full subject-level random effects vector is stacked as

```text
c_i = (c_i1, ..., c_iK).
```

## Installation

It can be installed by:

```r
pak::pak("Morty486/MPP")
library(MPP)
```

## Main Implemented Components

| Component | Function / Structure | Current purpose |
|---|---|---|
| Truncated exponent helper | `my_trunc()` | Avoids numerical overflow by bounding linear predictors. |
| Lower-triangular tools | `makeLowTriMat()`, `LowTriVec()` | Converts between lower-triangular matrices and vectorized Cholesky parameters. |
| Block diagonal matrix | `Bdiag()` | Builds block-diagonal matrices from a list/field of matrices. |
| Matrix inverse | `myinvCpp()`, `myinvCpp2()` | Tries symmetric inverse, general inverse, then pseudoinverse. |
| Cholesky decomposition | `myCholCpp()`, `myCholCpp2()` | Computes lower Cholesky factor with fallback perturbation. |
| Basic LME initializer | `init_LME()` | Initializes fixed effects, random-effect covariance, residual variance, and subject-level random effects. |
| One-marker longitudinal initializer | `init_LME2_one_marker()` | Initializes one continuous biomarker/mark submodel and handles subjects with no observations. |
| Multi-marker parameter container | `MPP_para_t` | Stores model parameters, builds full block covariance, and stacks subject-level variational parameters. |
| Struct test | `test_MPP_para_t_basic()` | Checks dimensions, covariance construction, inverse covariance, and subject-level stacking. |


## Current Data Flow

```text
R objects
  |
  |-- lists of vectors/matrices
  |-- weights
  |-- design matrices
  |
  v
Rcpp / Armadillo objects
  |
  |-- arma::vec
  |-- arma::mat
  |-- arma::field<arma::vec>
  |-- arma::field<arma::mat>
  |
  v
Initialization modules
  |
  |-- init_LME()
  |-- init_LME2_one_marker()
  |
  v
MPP parameter container
  |
  |-- biomarker-specific parameters
  |-- block-diagonal Sigma
  |-- inverse Sigma
  |-- stacked mu_i and Z_i
```

## Development Checks

The current testing script, `check.R`, checks all the function mention before by
using toy examples


