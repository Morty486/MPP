// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]


#include <math.h>
#include <RcppArmadillo.h>
using namespace Rcpp;



// maximum exponent to avoid numerical instability //


const double MAX_EXP = 15;

// [[Rcpp::export]]
double my_trunc(double x){
  if(x > MAX_EXP){
    x = MAX_EXP;
  }else if(x < -MAX_EXP){
    x = -MAX_EXP;
  }
  return x;
}

















