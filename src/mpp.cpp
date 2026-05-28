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



// convert  field of vec
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
    _["Sigma_bb_k"] = Sigma_b,
    _["mu_b_ik"] = mu_b,
    _["S_bb_ik"] = S_b
  );
}







// Stack a field of vectors into one long vector:
// input:  field with K vectors
// output: [x_1; x_2; ...; x_K]
arma::vec stack_vec_field(const arma::field<arma::vec>& x) {

  int total_dim = 0;

  for (int k = 0; k < x.n_elem; k++) {
    total_dim += x(k).n_elem;
  }

  arma::vec out(total_dim, arma::fill::zeros);

  int start = 0;

  for (int k = 0; k < x.n_elem; k++) {
    int len = x(k).n_elem;

    if (len > 0) {
      out.subvec(start, start + len - 1) = x(k);
    }

    start += len;
  }

  return out;
}



struct PP2_one_marker_para_t {
  arma::vec beta_time;      // fixed effects in log lambda_ik(s)
  arma::mat Sigma_aa_k;     // covariance of a_ik
  arma::mat invSigma_aa_k;

  arma::field<arma::vec> mu_a_ik;   // n x 1, variational mean of a_ik
  arma::field<arma::mat> S_aa_ik;   // n x 1, variational covariance of a_ik

  arma::field<arma::vec> Lvec_a_ik;
  arma::mat Lmat_aa_k;
};



struct MPP_data_t {

  int n;
  int K;
  int p_zz;

  // -----------------------------
  // Binary outcome data
  // -----------------------------
  arma::vec Y;
  arma::mat Z;
  arma::vec weights;

  // -----------------------------
  // Mark/value longitudinal data
  // -----------------------------
  arma::field<arma::vec> W;        // n x K
  arma::field<arma::vec> Time;     // n x K
  arma::field<arma::mat> U;        // n x K
  arma::field<arma::mat> Vdesign;  // n x K

  // -----------------------------
  // Measurement-time point-process design
  // -----------------------------
  arma::field<arma::vec> GQ_w_time;   // n x K
  arma::field<arma::vec> X_T_time;    // n x K
  arma::field<arma::vec> Z_T_time;    // n x K
  arma::field<arma::mat> X_gq_time;   // n x K
  arma::field<arma::mat> Z_gq_time;   // n x K

  // -----------------------------
  // Binary outcome connector matrices
  // eta_i = z_i^T gamma + sum_k a_ik^T H_k alpha_c,k
  //       + sum_k b_ik^T G_ik beta_c,k
  // -----------------------------
  arma::field<arma::mat> H;           // K x 1
  arma::field<arma::mat> G;           // n x K

  // -----------------------------
  // Gauss-Hermite quadrature for binary outcome expectation
  // -----------------------------
  arma::vec GHQ_w;
  arma::vec GHQ_t;

  MPP_data_t(const List& datalist) {

    Y = as<arma::vec>(datalist["Y"]);
    Z = as<arma::mat>(datalist["Z"]);
    weights = as<arma::vec>(datalist["weights"]);

    n = Y.n_elem;
    p_zz = Z.n_cols;
    K = as<int>(datalist["K"]);

    // -----------------------------
    // Basic mark/value objects
    // -----------------------------
    arma::field<arma::vec> W_tmp =
      as<arma::field<arma::vec>>(datalist["W"]);

    arma::field<arma::vec> Time_tmp =
      as<arma::field<arma::vec>>(datalist["Time"]);

    arma::field<arma::mat> U_tmp =
      as<arma::field<arma::mat>>(datalist["U"]);

    arma::field<arma::mat> V_tmp =
      as<arma::field<arma::mat>>(datalist["Vdesign"]);

    W = arma::field<arma::vec>(n, K);
    Time = arma::field<arma::vec>(n, K);
    U = arma::field<arma::mat>(n, K);
    Vdesign = arma::field<arma::mat>(n, K);

    int iter = 0;
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        W(i, k) = W_tmp(iter);
        Time(i, k) = Time_tmp(iter);
        U(i, k) = U_tmp(iter);
        Vdesign(i, k) = V_tmp(iter);
        iter++;
      }
    }

    // -----------------------------
    // Measurement-time point-process objects
    // -----------------------------
    arma::field<arma::vec> GQ_w_time_tmp =
      as<arma::field<arma::vec>>(datalist["GQ_w_time"]);

    arma::field<arma::vec> X_T_time_tmp =
      as<arma::field<arma::vec>>(datalist["X_T_time"]);

    arma::field<arma::vec> Z_T_time_tmp =
      as<arma::field<arma::vec>>(datalist["Z_T_time"]);

    arma::field<arma::mat> X_gq_time_tmp =
      as<arma::field<arma::mat>>(datalist["X_gq_time"]);

    arma::field<arma::mat> Z_gq_time_tmp =
      as<arma::field<arma::mat>>(datalist["Z_gq_time"]);

    GQ_w_time = arma::field<arma::vec>(n, K);
    X_T_time  = arma::field<arma::vec>(n, K);
    Z_T_time  = arma::field<arma::vec>(n, K);
    X_gq_time = arma::field<arma::mat>(n, K);
    Z_gq_time = arma::field<arma::mat>(n, K);

    iter = 0;
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        GQ_w_time(i, k) = GQ_w_time_tmp(iter);
        X_T_time(i, k)  = X_T_time_tmp(iter);
        Z_T_time(i, k)  = Z_T_time_tmp(iter);
        X_gq_time(i, k) = X_gq_time_tmp(iter);
        Z_gq_time(i, k) = Z_gq_time_tmp(iter);
        iter++;
      }
    }

    // -----------------------------
    // Outcome connector matrices
    // -----------------------------
    arma::field<arma::mat> H_tmp =
      as<arma::field<arma::mat>>(datalist["H"]);

    H = arma::field<arma::mat>(K);

    for (int k = 0; k < K; k++) {
      H(k) = H_tmp(k);
    }

    arma::field<arma::mat> G_tmp =
      as<arma::field<arma::mat>>(datalist["G"]);

    G = arma::field<arma::mat>(n, K);

    iter = 0;
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        G(i, k) = G_tmp(iter);
        iter++;
      }
    }

    // -----------------------------
    // Gauss-Hermite quadrature
    // -----------------------------
    GHQ_w = as<arma::vec>(datalist["GHQ_w"]);
    GHQ_t = as<arma::vec>(datalist["GHQ_t"]);
  }
};





struct MPP_para_t {

  // -----------------------------
  // Basic dimensions
  // -----------------------------
  int n;
  int K;

  arma::uvec q_a_vec;   // dim(a_ik) for each biomarker k
  arma::uvec q_b_vec;   // dim(b_ik) for each biomarker k
  arma::uvec q_c_vec;   // dim(c_ik) = q_a_k + q_b_k

  // -----------------------------
  // Measurement-time submodel
  // log lambda_ik(s) = x_i^T alpha_k + psi_1(s)^T theta_k + psi_2(s)^T a_ik
  // -----------------------------
  arma::field<arma::vec> beta_time;
  // beta_time(k) stacks alpha_k and theta_k

  // -----------------------------
  // Mark/value submodel
  // w_ik(s) = u_ik(s)^T delta_k + v_ik(s)^T b_ik + error
  // -----------------------------
  arma::field<arma::vec> delta;
  // delta(k) = fixed effects for mark/value model of biomarker k

  arma::vec sig2;
  // sig2(k) = sigma_k^2

  // -----------------------------
  // Binary outcome submodel
  // eta_i = z_i^T gamma + sum_k a_ik^T H alpha_c,k
  //       + sum_k b_ik^T G_ik beta_c,k
  // -----------------------------
  arma::vec gamma;

  arma::field<arma::vec> alpha_c;
  // alpha_c(k) = coefficient basis parameters for measurement-time effect

  arma::field<arma::vec> beta_c;
  // beta_c(k) = coefficient basis parameters for mark/value effect

  // -----------------------------
  // Prior covariance of c_i
  // Sigma = Bdiag(Sigma_1, ..., Sigma_K)
  // Sigma_k contains aa, ab, ba, bb blocks
  // -----------------------------
  arma::field<arma::mat> Sigma_k;
  arma::mat Sigma;
  arma::mat invSigma;

  arma::field<arma::mat> invSigma_k;

  // Cholesky factor of invSigma, useful for optimization
  arma::mat Lmat;

  // -----------------------------
  // Variational parameters
  // q_i(c_i) = N(mu_i, Z_i)
  // -----------------------------
  arma::field<arma::vec> mu_i;
  arma::field<arma::mat> Z_i;
  arma::field<arma::vec> Lvec_i;

  // biomarker-specific pieces
  arma::field<arma::vec> mu_ik;  // n x K, each = (mu_a_ik, mu_b_ik)
  arma::field<arma::mat> Z_ik;   // n x K, each block for c_ik



  // -----------------------------
  // Constructor
  // -----------------------------
  MPP_para_t(const int n_,
             const int K_,
             const arma::uvec& q_a_vec_,
             const arma::uvec& q_b_vec_,
             const arma::field<arma::vec>& beta_time_,
             const arma::field<arma::vec>& delta_,
             const arma::vec& sig2_,
             const arma::vec& gamma_,
             const arma::field<arma::vec>& alpha_c_,
             const arma::field<arma::vec>& beta_c_,
             const arma::field<arma::mat>& Sigma_k_,
             const arma::field<arma::vec>& mu_ik_,
             const arma::field<arma::mat>& Z_ik_)
    :
    n(n_),
    K(K_),
    q_a_vec(q_a_vec_),
    q_b_vec(q_b_vec_),
    beta_time(beta_time_),
    delta(delta_),
    sig2(sig2_),
    gamma(gamma_),
    alpha_c(alpha_c_),
    beta_c(beta_c_),
    Sigma_k(Sigma_k_),
    mu_ik(mu_ik_),
    Z_ik(Z_ik_)
  {
    q_c_vec = q_a_vec + q_b_vec;

    for (int k = 0; k < K; k++) {
      if (Sigma_k(k).n_rows != q_c_vec(k)) {
        Rcpp::stop("Sigma_k dimension does not match q_a_vec + q_b_vec.");
      }
    }

    // build full Sigma = blockdiag(Sigma_1, ..., Sigma_K)
    Sigma = Bdiag(Sigma_k);
    invSigma = myinvCpp(Sigma);
    Lmat = myCholCpp(invSigma);

    // build invSigma_k
    invSigma_k = arma::field<arma::mat>(K);
    for (int k = 0; k < K; k++) {
      invSigma_k(k) = myinvCpp(Sigma_k(k));
    }

    // stack mu_ik and Z_ik into mu_i and Z_i
    mu_i = arma::field<arma::vec>(n);
    Z_i = arma::field<arma::mat>(n);
    Lvec_i = arma::field<arma::vec>(n);

    for (int i = 0; i < n; i++) {

      arma::field<arma::vec> mu_blocks(K);
      arma::field<arma::mat> Z_blocks(K);

      for (int k = 0; k < K; k++) {
        mu_blocks(k) = mu_ik(i, k);
        Z_blocks(k) = Z_ik(i, k);
      }

      mu_i(i) = stack_vec_field(mu_blocks);
      Z_i(i) = Bdiag(Z_blocks);

      arma::mat Ltmp = myCholCpp(Z_i(i));
      Lvec_i(i) = LowTriVec(Ltmp);
    }
  }

  void updateInvSigma() {
    invSigma = myinvCpp(Sigma);
    Lmat = myCholCpp(invSigma);

    for (int k = 0; k < K; k++) {
      invSigma_k(k) = myinvCpp(Sigma_k(k));
    }
  }

  // -----------------------------
  // Constructor2
  // -----------------------------

  MPP_para_t(const List& paralist, const MPP_data_t& data) {

    n = data.n;
    K = data.K;

    beta_time = as<arma::field<arma::vec>>(paralist["beta_time"]);
    delta     = as<arma::field<arma::vec>>(paralist["delta"]);
    sig2      = as<arma::vec>(paralist["sig2"]);
    gamma     = as<arma::vec>(paralist["gamma"]);
    alpha_c   = as<arma::field<arma::vec>>(paralist["alpha_c"]);
    beta_c    = as<arma::field<arma::vec>>(paralist["beta_c"]);
    Sigma_k   = as<arma::field<arma::mat>>(paralist["Sigma_k"]);

    arma::field<arma::vec> mu_ik_tmp =
      as<arma::field<arma::vec>>(paralist["mu_ik"]);

    arma::field<arma::mat> Z_ik_tmp =
      as<arma::field<arma::mat>>(paralist["Z_ik"]);

    mu_ik = arma::field<arma::vec>(n, K);
    Z_ik  = arma::field<arma::mat>(n, K);

    int iter = 0;
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        mu_ik(i, k) = mu_ik_tmp(iter);
        Z_ik(i, k)  = Z_ik_tmp(iter);
        iter++;
      }
    }


    q_a_vec = arma::uvec(K);
    q_b_vec = arma::uvec(K);
    q_c_vec = arma::uvec(K);

    for (int k = 0; k < K; k++) {

      int qc = static_cast<int>(Sigma_k(k).n_rows);
      int qb = static_cast<int>(data.Vdesign(0, k).n_cols);

      if (qc <= 0) {
        Rcpp::stop("Sigma_k(k) has zero dimension.");
      }

      if (qb <= 0) {
        Rcpp::stop("q_b_vec(k) must be positive. Check Vdesign.");
      }

      if (qb >= qc) {
        Rcpp::stop("Need q_c_vec(k) > q_b_vec(k), so q_a_vec(k) is positive.");
      }

      q_c_vec(k) = qc;
      q_b_vec(k) = qb;
      q_a_vec(k) = qc - qb;

      if (mu_ik(0, k).n_elem != q_c_vec(k)) {
        Rcpp::stop("mu_ik dimension does not match Sigma_k.");
      }

      if (Z_ik(0, k).n_rows != q_c_vec(k) ||
          Z_ik(0, k).n_cols != q_c_vec(k)) {
        Rcpp::stop("Z_ik dimension does not match Sigma_k.");
      }
    }

    Sigma = Bdiag(Sigma_k);
    invSigma = myinvCpp(Sigma);
    Lmat = myCholCpp(invSigma);

    invSigma_k = arma::field<arma::mat>(K);
    for (int k = 0; k < K; k++) {
      invSigma_k(k) = myinvCpp(Sigma_k(k));
    }

    mu_i = arma::field<arma::vec>(n);
    Z_i = arma::field<arma::mat>(n);
    Lvec_i = arma::field<arma::vec>(n);

    for (int i = 0; i < n; i++) {
      arma::field<arma::vec> mu_blocks(K);
      arma::field<arma::mat> Z_blocks(K);

      for (int k = 0; k < K; k++) {
        mu_blocks(k) = mu_ik(i, k);
        Z_blocks(k) = Z_ik(i, k);
      }

      mu_i(i) = stack_vec_field(mu_blocks);
      Z_i(i) = Bdiag(Z_blocks);

      arma::mat Ltmp = myCholCpp(Z_i(i));
      Lvec_i(i) = LowTriVec(Ltmp);
    }
  }
};



// [[Rcpp::export]]
List test_MPP_para_t_basic() {

  // -----------------------------
  // 1. Toy dimensions
  // -----------------------------
  int n = 3;   // subjects
  int K = 2;   // biomarkers

  // For this toy example:
  // each biomarker has c_ik = (a_ik, b_ik), dimension 2
  int q_c = 2;

  // -----------------------------
  // 2. Create toy model parameters
  // -----------------------------

  arma::field<arma::vec> beta_time(K);
  arma::field<arma::vec> delta(K);
  arma::vec sig2(K);

  arma::vec gamma = {0.5, -0.2};  // baseline outcome coefficients

  arma::field<arma::vec> alpha_c(K);
  arma::field<arma::vec> beta_c(K);

  arma::field<arma::mat> Sigma_k(K);

  for (int k = 0; k < K; k++) {

    // beta_time(k) stacks alpha_k and theta_k
    beta_time(k) = arma::vec({0.1 + 0.1 * k, 0.2 + 0.1 * k});

    // fixed effects for mark/value submodel
    delta(k) = arma::vec({1.0 + k, -0.5});

    // residual variance for mark/value submodel
    sig2(k) = 1.0 + 0.2 * k;

    // coefficients connecting a_ik to outcome
    alpha_c(k) = arma::vec({0.3 + 0.1 * k});

    // coefficients connecting b_ik to outcome
    beta_c(k) = arma::vec({-0.2 - 0.1 * k});

    // covariance matrix for c_ik = (a_ik, b_ik)
    Sigma_k(k) = arma::mat({
      {1.0 + 0.1 * k, 0.2},
      {0.2, 1.5 + 0.1 * k}
    });
  }

  // -----------------------------
  // 3. Create toy variational parameters
  // -----------------------------

  arma::field<arma::vec> mu_ik(n, K);
  arma::field<arma::mat> Z_ik(n, K);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < K; k++) {

      // variational mean for c_ik
      mu_ik(i, k) = arma::vec({
        0.1 * i + 0.05 * k,
        -0.2 * i + 0.03 * k
      });

      // variational covariance for c_ik
      Z_ik(i, k) = arma::mat({
        {1.0 + 0.1 * i, 0.05},
        {0.05, 1.2 + 0.1 * k}
      });
    }
  }

  // -----------------------------
  // 4. Construct the new MPP_para_t struct
  // -----------------------------

  arma::uvec q_a_vec = {1, 1};
  arma::uvec q_b_vec = {1, 1};

  MPP_para_t para(
      n,
      K,
      q_a_vec,
      q_b_vec,
      beta_time,
      delta,
      sig2,
      gamma,
      alpha_c,
      beta_c,
      Sigma_k,
      mu_ik,
      Z_ik
  );

  // -----------------------------
  // 5. Check important quantities
  // -----------------------------

  arma::mat check_identity = para.Sigma * para.invSigma;

  // -----------------------------
  // 6. Return values to R
  // -----------------------------

  return List::create(
    _["n"] = para.n,
    _["K"] = para.K,

    _["beta_time"] = para.beta_time,
    _["delta"] = para.delta,
    _["sig2"] = para.sig2,
    _["gamma"] = para.gamma,
    _["alpha_c"] = para.alpha_c,
    _["beta_c"] = para.beta_c,

    _["Sigma_k"] = para.Sigma_k,
    _["Sigma"] = para.Sigma,
    _["invSigma"] = para.invSigma,
    _["Sigma_times_invSigma"] = check_identity,
    _["invSigma_k"] = para.invSigma_k,
    _["Lmat"] = para.Lmat,

    _["mu_ik"] = para.mu_ik,
    _["Z_ik"] = para.Z_ik,
    _["mu_i"] = para.mu_i,
    _["Z_i"] = para.Z_i,
    _["Lvec_i"] = para.Lvec_i,

    _["mu_i_1_length"] = para.mu_i(0).n_elem,
    _["Z_i_1_dim"] = arma::uvec({para.Z_i(0).n_rows, para.Z_i(0).n_cols}),
    _["Lvec_i_1_length"] = para.Lvec_i(0).n_elem
  );
}



// [[Rcpp::export]]
List test_model(const List& datalist,
                const List& paralist) {

  MPP_data_t data(datalist);
  MPP_para_t para(paralist, data);

  return List::create(

    _["n"] = data.n,
    _["K"] = data.K,

    _["q_a_vec"] = para.q_a_vec,
    _["q_b_vec"] = para.q_b_vec,
    _["q_c_vec"] = para.q_c_vec,

    _["Y"] = data.Y,
    _["weights"] = data.weights,

    _["gamma"] = para.gamma,
    _["sig2"] = para.sig2,

    _["Sigma"] = para.Sigma,
    _["invSigma"] = para.invSigma,

    _["mu_i_1"] = para.mu_i(0),
    _["Z_i_1"] = para.Z_i(0),

    _["W_1_1"] = data.W(0,0),
    _["Time_1_1"] = data.Time(0,0)
  );
}




int c_start_index(const MPP_para_t& para, int k) {
  int start = 0;
  for (int kk = 0; kk < k; kk++) {
    start += para.q_c_vec(kk);
  }
  return start;
}

arma::vec get_mu_a_ik(const MPP_para_t& para, int i, int k) {
  int start = c_start_index(para, k);
  int qa = para.q_a_vec(k);
  return para.mu_i(i).subvec(start, start + qa - 1);
}

arma::vec get_mu_b_ik(const MPP_para_t& para, int i, int k) {
  int start = c_start_index(para, k);
  int qa = para.q_a_vec(k);
  int qb = para.q_b_vec(k);
  return para.mu_i(i).subvec(start + qa, start + qa + qb - 1);
}

arma::mat get_Z_aa_ik(const MPP_para_t& para, int i, int k) {
  int start = c_start_index(para, k);
  int qa = para.q_a_vec(k);
  return para.Z_i(i).submat(start, start, start + qa - 1, start + qa - 1);
}

arma::mat get_Z_bb_ik(const MPP_para_t& para, int i, int k) {
  int start = c_start_index(para, k);
  int qa = para.q_a_vec(k);
  int qb = para.q_b_vec(k);
  return para.Z_i(i).submat(start + qa, start + qa,
                  start + qa + qb - 1, start + qa + qb - 1);
}


arma::vec combineMuZ_MPP(const MPP_para_t& para, int i) {
  return arma::join_cols(para.mu_i(i), para.Lvec_i(i));
}

void storeMuZ_MPP(const arma::vec& muZ,
                  MPP_para_t& para,
                  int i) {

  int q = para.mu_i(i).n_elem;

  arma::vec mu = muZ.subvec(0, q - 1);
  arma::vec Lvec = muZ.subvec(q, muZ.n_elem - 1);

  arma::mat L = makeLowTriMat(para.Z_i(i), Lvec);
  arma::mat Z = L * L.t();

  para.mu_i(i) = mu;
  para.Z_i(i) = Z;
  para.Lvec_i(i) = Lvec;

  // Update biomarker-specific blocks
  int start = 0;
  for (int k = 0; k < para.K; k++) {
    int qc = para.q_c_vec(k);
    para.mu_ik(i, k) = mu.subvec(start, start + qc - 1);
    para.Z_ik(i, k) = Z.submat(start, start,
              start + qc - 1, start + qc - 1);
    start += qc;
  }
}

class MPP_update_MuZ_Fun {
public:
  const MPP_data_t& data;
  MPP_para_t& para;
  int i_now;

  MPP_update_MuZ_Fun(const MPP_data_t& data,
                     MPP_para_t& para)
    : data(data), para(para) {
    i_now = 0;
  }

  double EvaluateWithGradient(const arma::mat& muZ_mat,
                              arma::mat& g) {

    arma::vec muZ = muZ_mat.col(0);
    storeMuZ_MPP(muZ, para, i_now);

    int i = i_now;
    int q = para.mu_i(i).n_elem;

    arma::vec mu = para.mu_i(i);
    arma::mat Z = para.Z_i(i);
    arma::mat L = makeLowTriMat(Z, para.Lvec_i(i));

    double fval = 0.0;

    arma::vec grad_mu(q, arma::fill::zeros);
    arma::mat grad_L(q, q, arma::fill::zeros);

    double wi = data.weights(i);

    // -----------------------------
    // 1. Binary outcome contribution
    // -----------------------------

    arma::vec m_i(q, arma::fill::zeros);

    int start = 0;
    for (int k = 0; k < para.K; k++) {

      int qa = para.q_a_vec(k);
      int qb = para.q_b_vec(k);
      int qc = para.q_c_vec(k);

      arma::vec m_a = data.H(k) * para.alpha_c(k);
      arma::vec m_b = data.G(i, k) * para.beta_c(k);

      m_i.subvec(start, start + qa - 1) = m_a;
      m_i.subvec(start + qa, start + qc - 1) = m_b;

      start += qc;
    }

    double eta_mean = arma::as_scalar(data.Z.row(i) * para.gamma) +
      arma::as_scalar(mu.t() * m_i);

    double eta_var = arma::as_scalar(m_i.t() * Z * m_i);

    eta_mean = my_trunc(eta_mean);

    arma::vec eta_quad;

    if (eta_var > 1e-12) {
      double eta_sd_scaled = std::sqrt(2.0 * eta_var);
      eta_quad = eta_mean + eta_sd_scaled * data.GHQ_t;
    } else {
      eta_quad = arma::vec(data.GHQ_t.n_elem);
      eta_quad.fill(eta_mean);
    }

    eta_quad = arma::clamp(eta_quad, -MAX_EXP, MAX_EXP);

    arma::vec exp_eta = arma::exp(eta_quad);
    arma::vec pi_quad = exp_eta / (1.0 + exp_eta);

    double E_log1pexp = arma::accu(data.GHQ_w % arma::log(1.0 + exp_eta));
    double E_pi = arma::accu(data.GHQ_w % pi_quad);

    fval += wi * (data.Y(i) * eta_mean - E_log1pexp);
    // binary objective

    grad_mu += wi * (data.Y(i) - E_pi) * m_i;

    if (eta_var > 1e-12) {
      double eta_sd_scaled = std::sqrt(2.0 * eta_var);
      double A = arma::accu(data.GHQ_w % pi_quad % data.GHQ_t);

      grad_L -= wi * (2.0 / eta_sd_scaled) * A *
        (m_i * m_i.t() * L);
    }

    // -----------------------------
    // 2. Measurement-time + mark/value contributions
    // -----------------------------

    start = 0;

    for (int k = 0; k < para.K; k++) {

      int qa = para.q_a_vec(k);
      int qb = para.q_b_vec(k);
      int qc = para.q_c_vec(k);

      arma::vec mu_a = mu.subvec(start, start + qa - 1);
      arma::vec mu_b = mu.subvec(start + qa, start + qc - 1);

      arma::mat Z_aa = Z.submat(start, start,
                                start + qa - 1, start + qa - 1);

      // ------------------------------------------------
      // 2a. Measurement-time point-process part
      // ------------------------------------------------

      arma::vec h = data.X_gq_time(i, k) * para.beta_time(k) +
        data.Z_gq_time(i, k) * mu_a;

      for (int r = 0; r < h.n_elem; r++) {
        h(r) += 0.5 * arma::as_scalar(
          data.Z_gq_time(i, k).row(r) *
            Z_aa *
            data.Z_gq_time(i, k).row(r).t()
        );
      }

      h = arma::clamp(h, -MAX_EXP, MAX_EXP);
      h = arma::exp(h);

      fval += wi * (
        arma::accu(data.X_T_time(i, k) % para.beta_time(k)) +
          arma::accu(data.Z_T_time(i, k) % mu_a) -
          arma::accu(data.GQ_w_time(i, k) % h)
      );

      // point process objective

      arma::vec grad_mu_a =
        data.Z_T_time(i, k) -
        data.Z_gq_time(i, k).t() *
        (data.GQ_w_time(i, k) % h);

      grad_mu.subvec(start, start + qa - 1) += wi * grad_mu_a;

      arma::mat A_time =
        data.Z_gq_time(i, k).t() *
        arma::diagmat(data.GQ_w_time(i, k) % h) *
        data.Z_gq_time(i, k);

      arma::mat A_full(q, q, arma::fill::zeros);
      A_full.submat(start, start,
                    start + qa - 1, start + qa - 1) = A_time;

      grad_L -= wi * A_full * L;

      // ------------------------------------------------
      // 2b. Mark/value Gaussian longitudinal part
      // ------------------------------------------------

      int m_obs = data.W(i, k).n_elem;

      if (m_obs > 0) {

        arma::vec resid =
          data.W(i, k) -
          data.U(i, k) * para.delta(k) -
          data.Vdesign(i, k) * mu_b;

        arma::mat Z_bb = Z.submat(start + qa, start + qa,
                                  start + qc - 1, start + qc - 1);

        double sig2_k = para.sig2(k);

        fval += wi * (
          -0.5 * m_obs * std::log(sig2_k)
          -0.5 / sig2_k * (
              arma::accu(arma::square(resid)) +
              arma::trace(data.Vdesign(i, k) * Z_bb *
              data.Vdesign(i, k).t())
          )
        );

        // longitudinal objective

        arma::vec grad_mu_b =
          data.Vdesign(i, k).t() * resid / sig2_k;

        grad_mu.subvec(start + qa, start + qc - 1) += wi * grad_mu_b;

        arma::mat A_mark =
          data.Vdesign(i, k).t() *
          data.Vdesign(i, k) / sig2_k;

        arma::mat A_mark_full(q, q, arma::fill::zeros);
        A_mark_full.submat(start + qa, start + qa,
                           start + qc - 1, start + qc - 1) = A_mark;

        grad_L -= wi * A_mark_full * L;
      }

      start += qc;
    }

    // -----------------------------
    // 3. Prior term
    // -----------------------------

    fval += wi * (
      -0.5 * arma::as_scalar(mu.t() * para.invSigma * mu)
      -0.5 * arma::trace(para.invSigma * Z)
    );

    // prior objective

    grad_mu -= wi * para.invSigma * mu;
    grad_L  -= wi * para.invSigma * L;

    // -----------------------------
    // 4. Entropy term
    // -----------------------------

    double logdet_Z, sign;
    arma::log_det(logdet_Z, sign, Z);

    fval += wi * 0.5 * logdet_Z;

    // entropy objective

    grad_L += wi * arma::trans(arma::inv(arma::trimatl(L)));

    // -----------------------------
    // 5. Turn maximization into minimization
    // -----------------------------

    fval = -fval;

    arma::vec grad_all = arma::join_cols(grad_mu, LowTriVec(grad_L));
    g.col(0) = -grad_all;

    return fval;
  }
};


// [[Rcpp::export]]
List test_MPP_MuZ_eval(const List& datalist,
                       const List& paralist,
                       int i_R = 1) {

  MPP_data_t data(datalist);
  MPP_para_t para(paralist, data);

  int i = i_R - 1;

  MPP_update_MuZ_Fun fun(data, para);
  fun.i_now = i;

  arma::vec muZ = combineMuZ_MPP(para, i);
  arma::mat muZ_mat = muZ;

  arma::mat g(muZ.n_elem, 1, arma::fill::zeros);

  double fval = fun.EvaluateWithGradient(muZ_mat, g);

  return List::create(
    _["fval"] = fval,
    _["gradient"] = g,
    _["gradient_norm"] = arma::norm(g),
    _["muZ"] = muZ,
    _["mu_i"] = para.mu_i(i),
    _["Z_i"] = para.Z_i(i),
    _["q_a_vec"] = para.q_a_vec,
    _["q_b_vec"] = para.q_b_vec,
    _["q_c_vec"] = para.q_c_vec
  );
}


