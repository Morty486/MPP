# MPP

An in-progress R/Rcpp package for developing a multi-marker marked point process joint model.

This repository currently focuses on building and testing the computational foundation for the full model: matrix utilities, mixed-effects initialization, one-marker longitudinal/mark initialization, and parameter storage for a future multi-marker marked point process model.

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

## Current Progress

The current code is not yet the final full model. The main progress so far is:

- Implemented core Rcpp/Armadillo helper functions for matrix and field manipulation.
- Implemented stable numerical utilities for inverse and Cholesky calculations.
- Implemented a basic linear mixed-effects initializer, `init_LME()`.
- Implemented a one-marker longitudinal/mark initializer, `init_LME2_one_marker()`, including the case where some subjects have no observations for a biomarker.
- Implemented the first version of the multi-marker parameter structure, `MPP_para_t`.
- Implemented `test_MPP_para_t_basic()` to check whether multi-marker parameters are correctly stored, stacked, and converted into block-diagonal covariance matrices.
- Added `check.R` as a working script for manually testing helper functions and model components.

## Installation

During development, the package can be loaded locally from the project folder:

```r
setwd("~/Documents/MPP")

Rcpp::compileAttributes()
devtools::clean_dll()
devtools::load_all(reset = TRUE, recompile = TRUE)

library(MPP)
```

After pushing the package to GitHub, it can be installed by:

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

The current testing script, `check.R`, manually checks:

- lower-triangular matrix conversion;
- block-diagonal matrix construction;
- inverse and Cholesky fallback behavior;
- basic LME initialization on simulated toy data;
- one-marker longitudinal initialization with irregular observation times;
- one-marker initialization when more subjects have no biomarker observations;
- construction of the multi-marker parameter structure.


## Current Development Notes

This package is under active development. The current code mainly prepares the computational infrastructure for the full model.

Important next steps:

1. Add a formal data structure for the full multi-marker marked point process model.
2. Implement the one-marker measurement-time point process initializer for `a_ik`.
3. Combine measurement-time random effects `a_ik` and mark/value random effects `b_ik` into the full `c_ik` update.
4. Implement the full ELBO for the outcome, measurement-time process, and mark/value process.
5. Add coordinate-ascent or L-BFGS updates for the full variational approximation.
6. Replace manual checks in `check.R` with formal `testthat` unit tests.
7. Add simulation examples and diagnostic plots after the full estimation routine is stable.

One small cleanup item: make sure names used in `check.R` match the names returned by C++ functions. For example, the current C++ return names from `init_LME2_one_marker()` are `Sigma_bb_k`, `mu_b_ik`, and `S_bb_ik`.

## Repository Status

This repository currently represents a development checkpoint rather than a finished release. The main accomplishment is that the low-level Rcpp utilities and the first parameter-storage design for the multi-marker model are now in place. The next milestone is to connect these components into the full marked point process variational estimation algorithm.
