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




// [[Rcpp::export]]
List init_LME(arma::vec weights, arma::field<arma::vec> Y,
              arma::field<arma::mat> X, arma::field<arma::mat> Z,
              const int maxiter=100, const double eps=1e-4){

  const int n = Y.n_elem;
  const int px = X(0).n_cols;
  const int pz = Z(0).n_cols;
  const int npara = pz*pz + px + 1;

  arma::field<arma::vec> mu(n);
  arma::field<arma::mat> V(n);
  arma::vec beta(px);
  double sig2 = 1;
  arma::mat Sigma(pz,pz);
  Sigma.zeros();
  Sigma.diag().fill(1.0);
  arma::mat Sigma_inv = Sigma;

  double N=0;
  arma::mat XXinv(px,px, arma::fill::zeros);
  arma::vec XY(px, arma::fill::zeros);
  arma::field<arma::mat> ZZ(n);
  for(int i=0; i<n; i++){
    if(Y(i).n_elem > 0){
      XXinv += weights(i) * X(i).t() * X(i);
      XY += weights(i) * X(i).t() * Y(i);
      N += weights(i) * Y(i).n_elem;
      ZZ(i) =  Z(i).t() * Z(i);
    }
  }
  //XXinv = myinvCpp(XXinv);
  //beta = XXinv*XY;
  beta = arma::solve(XXinv, XY);

  arma::mat Sigma_tmp(pz, pz, arma::fill::zeros);
  int iter = 0;
  arma::vec beta_00 = beta;
  arma::mat Sigma_00 = Sigma;
  double sig2_00 = sig2;

  for(iter=0; iter < maxiter; iter++){

    beta_00 = beta;
    Sigma_00 = Sigma;
    sig2_00 = sig2;
    // update mu_i and V_i
    Sigma_tmp.zeros();
    XY.zeros();
    XXinv.zeros();
    for(int i=0; i<n; i++){
      if(Y(i).n_elem > 0){
        V(i) =  myinvCpp(ZZ(i) + sig2 * Sigma_inv);
        mu(i) =  V(i) * Z(i).t() *(Y(i) - X(i)* beta);
        V(i) *=  sig2;
        Sigma_tmp += weights(i) * (mu(i)*mu(i).t() + V(i));

        arma::mat ZVZ = - Z(i) *V(i)*Z(i).t() / sig2/ sig2;
        ZVZ.diag() += 1/ sig2;

        XXinv += weights(i) * X(i).t() * ZVZ * X(i);
        XY += weights(i) * X(i).t() * ZVZ * Y(i);
      }else{
        mu(i).zeros();
        V(i) = Sigma;
        Sigma_tmp += weights(i) * (mu(i)*mu(i).t() + V(i));
      }
    }

    //beta = XXinv*XY;
    beta = arma::solve(XXinv, XY);
    Sigma = Sigma_tmp/arma::accu(weights);
    Sigma_inv = myinvCpp(Sigma);

    // update sig2;
    sig2 = 0;
    for(int i=0; i<n; i++){
      if(Y(i).n_elem > 0){
        sig2 += weights(i) * (
          arma::trace(V(i) * ZZ(i)) +
            arma::accu(arma::square(
                Y(i) - X(i)*beta - Z(i)*mu(i)))
        );
      }
    }
    sig2 /= N;

    if(iter > 1){

      double err_para = arma::accu( arma::square(beta_00-beta)) +
        arma::accu( arma::square(Sigma_00-Sigma))+
        (sig2_00-sig2)*(sig2_00-sig2);

      err_para = std::sqrt(err_para/npara);
      //Rcout << iter << " err_para=" << err_para <<"\n";

      if(err_para<eps){
        break;
      }

    }
  }

  return List::create(
    _["sig2"] = sig2,
    _["Sigma"] = Sigma,
    _["beta"] = beta,
    _["mu"] = mu,
    _["V"] = V
  );

}




// Initialize one continuous biomarker/mark process k:
//
// W(i)      = vector of observed values w_ik
// U(i)      = fixed-effect design matrix U_ik
// Vdesign(i)= random-effect design matrix V_ik
//
// Model:
// W_i = U_i delta + V_i b_i + e_i
// b_i ~ N(0, Sigma_b)
// e_i ~ N(0, sig2 I)

// [[Rcpp::export]]
List init_LME2_one_marker(const arma::vec& weights,
                          const arma::field<arma::vec>& W,
                          const arma::field<arma::mat>& U,
                          const arma::field<arma::mat>& Vdesign,
                          const int maxiter = 100,
                          const double eps = 1e-4,
                          const double ridge = 1e-8) {

  const int n = W.n_elem;
  const int p_delta = U(0).n_cols;
  const int q_b = Vdesign(0).n_cols;
  const int npara = q_b * q_b + p_delta + 1;

  arma::field<arma::vec> mu_b(n);
  arma::field<arma::mat> S_b(n);

  arma::vec delta(p_delta, arma::fill::zeros);
  double sig2 = 1.0;

  arma::mat Sigma_b(q_b, q_b, arma::fill::eye);
  arma::mat Sigma_b_inv = Sigma_b;

  double Nobs = 0.0;
  arma::mat UU(p_delta, p_delta, arma::fill::zeros);
  arma::vec UW(p_delta, arma::fill::zeros);
  arma::field<arma::mat> VV(n);

  // Initial weighted least squares for delta
  for (int i = 0; i < n; i++) {
    if (W(i).n_elem > 0) {
      UU += weights(i) * U(i).t() * U(i);
      UW += weights(i) * U(i).t() * W(i);
      VV(i) = Vdesign(i).t() * Vdesign(i);
      Nobs += weights(i) * W(i).n_elem;
    }
  }

  UU += ridge * arma::eye(p_delta, p_delta);
  delta = arma::solve(UU, UW);

  arma::vec delta_old = delta;
  arma::mat Sigma_old = Sigma_b;
  double sig2_old = sig2;

  for (int iter = 0; iter < maxiter; iter++) {

    delta_old = delta;
    Sigma_old = Sigma_b;
    sig2_old = sig2;

    arma::mat Sigma_tmp(q_b, q_b, arma::fill::zeros);
    UU.zeros();
    UW.zeros();

    for (int i = 0; i < n; i++) {

      if (W(i).n_elem > 0) {

        // Posterior covariance of b_i
        arma::mat S_unscaled = myinvCpp(VV(i) + sig2 * Sigma_b_inv);

        // Posterior mean of b_i
        mu_b(i) = S_unscaled * Vdesign(i).t() * (W(i) - U(i) * delta);

        // Convert to actual posterior covariance
        S_b(i) = sig2 * S_unscaled;

        Sigma_tmp += weights(i) * (mu_b(i) * mu_b(i).t() + S_b(i));

        // Working inverse marginal covariance for updating delta
        arma::mat Vinv_work =
          - Vdesign(i) * S_b(i) * Vdesign(i).t() / (sig2 * sig2);

          Vinv_work.diag() += 1.0 / sig2;

          UU += weights(i) * U(i).t() * Vinv_work * U(i);
          UW += weights(i) * U(i).t() * Vinv_work * W(i);

      } else {

        // No observed mark values for this biomarker:
        // posterior remains prior
        mu_b(i) = arma::zeros(q_b);
        S_b(i) = Sigma_b;

        Sigma_tmp += weights(i) * (mu_b(i) * mu_b(i).t() + S_b(i));
      }
    }

    UU += ridge * arma::eye(p_delta, p_delta);
    delta = arma::solve(UU, UW);

    Sigma_b = Sigma_tmp / arma::accu(weights);
    Sigma_b_inv = myinvCpp(Sigma_b);

    // Update residual variance sigma_k^2
    sig2 = 0.0;

    for (int i = 0; i < n; i++) {
      if (W(i).n_elem > 0) {
        arma::vec resid = W(i) - U(i) * delta - Vdesign(i) * mu_b(i);

        sig2 += weights(i) * (
          arma::trace(S_b(i) * VV(i)) +
            arma::accu(arma::square(resid))
        );
      }
    }

    if (Nobs > 0) {
      sig2 /= Nobs;
    } else {
      sig2 = 1.0;
    }

    double err_para =
      arma::accu(arma::square(delta_old - delta)) +
      arma::accu(arma::square(Sigma_old - Sigma_b)) +
      std::pow(sig2_old - sig2, 2.0);

    err_para = std::sqrt(err_para / npara);

    if (iter > 1 && err_para < eps) {
      break;
    }
  }

  return List::create(
    _["delta_k"] = delta,
    _["sig2_k"] = sig2,
    _["Sigma_bk"] = Sigma_b,
    _["mu_bik"] = mu_b,
    _["S_bik"] = S_b
  );
}






