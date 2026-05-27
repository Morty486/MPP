# MPP

An in-progress R/Rcpp package for developing a multi-marker marked point process joint model.


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
| Cholesky decomposition | `myCholCpp()`, `myCholCpp2()` | Computes lower Cholesky factor. |
| Basic LME initializer | `init_LME()` | Initializes fixed effects, random-effect covariance, residual variance, and subject-level random effects. |
| One-marker longitudinal initializer | `init_LME2_one_marker()` | Initializes one continuous biomarker/mark submodel and handles subjects with no observations. |
| Multi-marker parameter container | `MPP_para_t` | Stores model parameters, builds full block covariance, and stacks subject-level variational parameters. |
| Struct test | `test_MPP_para_t_basic()` | Checks dimensions, covariance construction, inverse covariance, and subject-level stacking. |


## Current Data Flow


```text
// ============================================================================
//                         INPUT STRUCTURE (from R)
// ============================================================================
//
//                                  |
//                                  v
//
// ┌────────────────────────────────────────────────────────────────────────────┐
// │                         datalist (List)                                    │
// ├────────────────────────────────────────────────────────────────────────────┤
// │                                                                            │
// │  Basic outcome/covariate data                                              │
// │  ├── Y              [n x 1 binary outcome]                                  │
// │  ├── weights        [n x 1 sampling weights]                                │
// │  ├── Z              [n x p_zz baseline covariates for outcome model]        │
// │  └── K              [number of biomarkers / longitudinal variables]         │
// │                                                                            │
// │  Mark/value longitudinal data: n x K flattened list                         │
// │  ├── W              [observed biomarker values w_ik]                        │
// │  ├── Time           [observed measurement times t_ijk]                      │
// │  ├── U              [fixed-effect design matrices U_ik for mark model]      │
// │  └── Vdesign        [random-effect design matrices V_ik for b_ik]           │
// │                                                                            │
// │  Measurement-time point-process design: n x K flattened list                │
// │  ├── GQ_w_time      [Gauss quadrature weights for ∫ lambda_ik(s) ds]         │
// │  ├── X_T_time       [sum of fixed-effect design at observed event times]    │
// │  ├── Z_T_time       [sum of a_ik random-effect design at observed times]    │
// │  ├── X_gq_time      [fixed-effect design at quadrature nodes]               │
// │  └── Z_gq_time      [a_ik random-effect design at quadrature nodes]         │
// │                                                                            │
// │  Outcome connector matrices                                                 │
// │  ├── H              [K-list; H_k links a_ik to outcome]                     │
// │  └── G              [n x K flattened list; G_ik links b_ik to outcome]      │
// │                                                                            │
// │  Gaussian-Hermite quadrature for binary outcome expectation                 │
// │  ├── GHQ_w          [Gauss-Hermite weights]                                 │
// │  └── GHQ_t          [Gauss-Hermite nodes]                                   │
// │                                                                            │
// └────────────────────────────────────────────────────────────────────────────┘
//
//                                  |
//                                  v
//
// ┌────────────────────────────────────────────────────────────────────────────┐
// │                         paralist (List)                                    │
// ├────────────────────────────────────────────────────────────────────────────┤
// │                                                                            │
// │  Measurement-time submodel                                                  │
// │  └── beta_time      [K-list; beta_time_k = (alpha_k, theta_k)]              │
// │                                                                            │
// │  Mark/value longitudinal submodel                                           │
// │  ├── delta          [K-list; fixed effects delta_k]                         │
// │  └── sig2           [K x 1 residual variances sigma_k^2]                    │
// │                                                                            │
// │  Binary outcome submodel                                                    │
// │  ├── gamma          [p_zz x 1 baseline covariate effects]                   │
// │  ├── alpha_c        [K-list; coefficient basis for a_ik effect]             │
// │  └── beta_c         [K-list; coefficient basis for b_ik effect]             │
// │                                                                            │
// │  Prior covariance of latent random effects                                  │
// │  └── Sigma_k        [K-list; covariance of c_ik = (a_ik, b_ik)]             │
// │                                                                            │
// │  Variational parameters                                                     │
// │  ├── mu_ik          [n x K flattened list; mean of c_ik]                    │
// │  └── Z_ik           [n x K flattened list; covariance of c_ik]              │
// │                                                                            │
// │  Optional safer future version                                              │
// │  ├── q_a_vec        [K x 1 dimension of a_ik]                               │
// │  └── q_b_vec        [K x 1 dimension of b_ik]                               │
// │                                                                            │
// └────────────────────────────────────────────────────────────────────────────┘
//
// ============================================================================
//                       IMPORTANT FLATTENING RULE
// ============================================================================
//
// All n x K list objects must be flattened in subject-major order:
//
//   (i = 1, k = 1)
//   (i = 1, k = 2)
//   ...
//   (i = 2, k = 1)
//   (i = 2, k = 2)
//   ...
//
// Flatten these objects:
//   W, Time, U, Vdesign,
//   GQ_w_time, X_T_time, Z_T_time, X_gq_time, Z_gq_time,
//   G,
//   mu_ik, Z_ik
//
// Do NOT flatten H.
// H is only a K-list, one matrix per biomarker.
//
// ============================================================================
//                     LATENT RANDOM EFFECT STRUCTURE
// ============================================================================
//
// For subject i and biomarker k:
//
//   c_ik = ( a_ik )
//          ( b_ik )
//
// where
//
//   a_ik = random effects for measurement-time process
//   b_ik = random effects for mark/value longitudinal process
//
// For subject i:
//
//   c_i = ( c_i1, c_i2, ..., c_iK )
//
// Variational approximation:
//
//   q_i(c_i) = Normal(mu_i, Z_i)
//
// In C++:
//   mu_ik(i,k)  stores mean of c_ik
//   Z_ik(i,k)   stores covariance of c_ik
//   mu_i(i)     stacks all mu_ik over k
//   Z_i(i)      block-diagonal stack of all Z_ik over k
//
// ============================================================================



// ============================================================================
//                         DATA STRUCTURE INITIALIZATION
// ============================================================================
//
//                                  |
//                                  v
//
//                 ┌─────────────────────────────────────────────┐
//                 │                                             │
//                 v                                             v
//
//      ┌─────────────────────────────┐              ┌─────────────────────────────┐
//      │        MPP_data_t            │              │        MPP_para_t            │
//      │        Constructor           │              │        Constructor           │
//      ├─────────────────────────────┤              ├─────────────────────────────┤
//      │                             │              │                             │
//      │ • Read outcome Y, Z, weights│              │ • Read beta_time, delta     │
//      │ • Read K biomarkers         │              │ • Read sig2, gamma          │
//      │                             │              │ • Read alpha_c, beta_c      │
//      │ • Reshape n x K fields:     │              │ • Read Sigma_k              │
//      │     W, Time                 │              │                             │
//      │     U, Vdesign              │              │ • Read mu_ik, Z_ik          │
//      │                             │              │ • Reshape mu_ik, Z_ik       │
//      │ • Reshape point-process     │              │   into n x K fields         │
//      │   design objects:           │              │                             │
//      │     GQ_w_time               │              │ • Infer dimensions:         │
//      │     X_T_time, Z_T_time      │              │     q_c = dim(Sigma_k)      │
//      │     X_gq_time, Z_gq_time    │              │     q_b = ncol(Vdesign)     │
//      │                             │              │     q_a = q_c - q_b         │
//      │ • Store outcome connectors: │              │                             │
//      │     H       : K-list        │              │ • Build full Sigma:         │
//      │     G       : n x K field   │              │     Sigma = Bdiag(Sigma_k)  │
//      │                             │              │     invSigma = Sigma^{-1}   │
//      │ • Store GHQ_w, GHQ_t        │              │     Lmat = chol(invSigma)   │
//      │                             │              │                             │
//      │ Output: data object         │              │ • Stack variational params: │
//      │                             │              │     mu_i = stack_k mu_ik    │
//      │                             │              │     Z_i  = Bdiag_k Z_ik     │
//      │                             │              │     Lvec_i = LowTriVec(chol)│
//      │                             │              │                             │
//      │                             │              │ Output: para object         │
//      └─────────────────────────────┘              └─────────────────────────────┘
//
//                 │                                             │
//                 └──────────────────────┬──────────────────────┘
//                                        │
//                                        v
//
//                     ┌─────────────────────────────────────┐
//                     │     Ready for VLB / update step      │
//                     ├─────────────────────────────────────┤
//                     │                                     │
//                     │ • test_model() checks construction   │
//                     │ • test_MPP_MuZ_eval() checks one     │
//                     │   subject's variational objective    │
//                     │                                     │
//                     │ Next target: update q_i(c_i)         │
//                     │   q_i(c_i) = N(mu_i, Z_i)            │
//                     │                                     │
//                     └─────────────────────────────────────┘
//
// ============================================================================
```


## Development Checks

The current testing script, `check.R`, checks all the function mention before by
using toy examples

