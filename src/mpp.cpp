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



 // [[Rcpp::export]]
 arma::mat makeLowTriMat(const arma::mat& V,
                         const arma::vec& Lvec){
   arma::uvec lower_indices = arma::trimatl_ind( arma::size(V) );
   arma::mat L( arma::size(V), arma::fill::zeros);
   L(lower_indices) = Lvec;
   return L;
 }


// extract lower-triangular elements of matrix
// [[Rcpp::export]]
arma::vec LowTriVec(const arma::mat& V){
  arma::uvec lower_indices = arma::trimatl_ind( arma::size(V) );
  arma::vec Lvec = V(lower_indices);
  return Lvec;
}


// reshape field of vec to n \times K
void field_reshape_vec(const arma::field<arma::vec>& Y_tmp,
                       arma::field<arma::vec>& Y, int n, int K ){
  int iter = 0;
  for(int k=0; k<K;k++){
    for(int i=0; i<n; i++){
      Y(i,k) = Y_tmp(iter);
      iter++;
    }
  }
}



// reshape field of mat to n \times K
void field_reshape_mat(const arma::field<arma::mat>& Y_tmp,
                       arma::field<arma::mat>& Y, int n, int K ){
  int iter = 0;
  for(int k=0; k<K;k++){
    for(int i=0; i<n; i++){
      Y(i,k) = Y_tmp(iter);
      iter++;
    }
  }
}


// transpose a field of vector
arma::field<arma::vec> trans_field_vec(const arma::field<arma::vec>& Y){
  arma::field<arma::vec> res(1,Y.n_elem);
  field_reshape_vec(Y, res, 1, Y.n_elem);
  return res;
}

// transpose a field of matrix
arma::field<arma::mat> trans_field_mat(const arma::field<arma::mat>& Y){
  arma::field<arma::mat> res(1,Y.n_elem);
  field_reshape_mat(Y, res, 1, Y.n_elem);
  return res;
}

// convert vec to field of vec
arma::field<arma::vec> vec_to_field(const arma::vec& mu,
                                    const arma::uvec& p_z_vec){

  arma::field<arma::vec> mu_f(p_z_vec.n_elem);

  int start = 0;
  for(int k=0; k<p_z_vec.n_elem; k++){
    mu_f(k) = mu.subvec(start,start+p_z_vec(k)-1);
    start = start+p_z_vec(k);
  }

  return mu_f;
}





 // [[Rcpp::export]]
 arma::mat Bdiag(const arma::field<arma::mat>& M){
   int nc=0, nr=0;
   for(int j=0; j<M.n_elem; j++){
     nc += M(j).n_cols;
     nr += M(j).n_rows;
   }
   arma::mat res(nr,nc,arma::fill::zeros);

   int start_r=0, end_r=0, start_c=0, end_c=0;

   for(int j = 0;  j < M.n_elem; j++){
     end_r = start_r +  M(j).n_rows-1;
     end_c = start_c +  M(j).n_cols-1;
     res.submat(start_r, start_c, end_r, end_c) =  M(j);
     start_r = end_r + 1;
     start_c = end_c + 1;
   }

   return res;
 }



// convert  field of vec  to vec
arma::vec field_to_vec(const arma::field<arma::vec>& mu,
                       const arma::uvec& p_z_vec){

  int p_z = arma::accu(p_z_vec);
  arma::vec mu_vec(p_z);

  int start = 0;
  for(int k=0; k<p_z_vec.n_elem; k++){
    mu_vec.subvec(start,start+p_z_vec(k)-1) = mu(k);
    start = start+p_z_vec(k);
  }

  return mu_vec;
}


// matrix inverse //
// [[Rcpp::export]]
arma::mat myinvCpp(const arma::mat& A){
  bool invflag = false;
  arma::mat B = A;
  invflag = arma::inv_sympd(B , A, arma::inv_opts::allow_approx);
  if(!invflag){
    //Rcout << "inv_sympd failed, try inv\n";
    invflag = arma::inv( B, A, arma::inv_opts::allow_approx);
    if(!invflag){
      //Rcout << "inv failed, try pinv\n";
      invflag = arma::pinv( B,A);
      if(!invflag){
        //Rcout << "all inv methods failed!\n";
        //Rcout << A << "\n";
        throw std::runtime_error("error");
      }
    }
  }
  return B;
}

// [[Rcpp::export]]
arma::mat myinvCpp2(const arma::mat& A){
  bool invflag = false;
  arma::mat B;

  invflag = arma::inv_sympd(B, A);

  if(!invflag){
    //Rcout << "inv_sympd failed, try inv\n";
    invflag = arma::inv(B, A);
    if(!invflag){
      //Rcout << "inv failed, try pinv\n";
      invflag = arma::pinv(B, A);

      if(!invflag){
        //Rcout << "all inv methods failed!\n";
        //Rcout << A << "\n";
        throw std::runtime_error("all inverse methods failed");
      }
    }
  }

  return B;
}


// Cholesky decomposition //
// [[Rcpp::export]]
arma::mat myCholCpp(arma::mat A){
  bool flag = false;
  arma::mat B( arma::size(A), arma::fill::zeros);
  flag = arma::chol(B , A, "lower");
  if(!flag){
    arma::vec avec = A.diag();
    arma::vec tmp = avec.elem(arma::find(avec));
    double val = 0.01 * arma::mean(arma::abs(tmp));
    A.diag() += val;
    flag = arma::chol(B , A,"lower");
    if(!flag){
      B.diag().fill(val);
    }
  }
  return B;
}



// Cholesky decomposition //
// [[Rcpp::export]]
arma::mat myCholCpp2(arma::mat A){
  bool flag = false;
  arma::mat B(arma::size(A), arma::fill::zeros);

  flag = arma::chol(B, A, "lower");

  if(!flag){
    arma::vec avec = A.diag();
    arma::vec tmp = avec.elem(arma::find(avec));

    double val;
    if(tmp.n_elem == 0){
      val = 0.01;
    } else {
      val = 0.01 * arma::mean(arma::abs(tmp));
    }

    A.diag() += val;

    flag = arma::chol(B, A, "lower");

    if(!flag){
      B.zeros();
      B.diag().fill(val);
    }
  }

  return B;
}












