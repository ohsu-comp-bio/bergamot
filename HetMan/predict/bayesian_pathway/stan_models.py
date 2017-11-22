
from pystan import stan
import numpy as np


model_code = '''
    data {
        int<lower=1> N;     // number of samples
        int<lower=1> G;     // number of genetic features

        real r[N, G];       // observed RNA-seq expression values
        real c[N, G];       // observed copy number GISTIC values
        real p[N, G];       // observed proteomic measurements

        int <lower=G> P;            // number of known pathway interactions
        int <lower=1, upper=G> po[P];   // indices of genes left by edges
        int <lower=1, upper=G> pi[P];   // indices of genes entered by edges
    }
    
    parameters {
        // the weights given to RNA-seq expression when determining
        // transcription levels of each gene, and the prior governing the
        // distribution of these weights
        vector<lower=0, upper=1>[G] tx_wght;
        vector<lower=0.1, upper=5>[2] tx_wght_prior;

        // the inferred accuracy of transcription levels in measuring
        // activity for each gene, and the prior governing the distribution
        // of these accuracies
        vector<lower=0.1>[G] tx_acc;
        vector<lower=0.01, upper=20>[2] tx_acc_prior;

        // inferred activity of each gene in each sample
        matrix[N, G] act;

        // the weights given to pathway edges between genes that interact
        // with one another, and the prior governing the distribution of
        // these weights
        vector<lower=0, upper=1>[P] edge_wght;
        vector<lower=0.1, upper=5>[2] edge_wght_prior;
    }

    transformed parameters{
        matrix[N, G] tx;        // inferred transcription levels of genes
        matrix[N, G] pred_p;    // predicted protein levels of genes

        // calculate the transcription level of each gene in each sample using
        // the corresponding observed expression and copy number levels
        for (g in 1:G) {
            for (n in 1:N) {
                pred_p[n, g] = 0;
                tx[n, g] = (tx_wght[g] * r[n, g]) + ((1.0 - tx_wght[g]) * c[n, g]);
            }
        }

        // calculate the predicted protein levels of each gene in each sample
        // given inferred activity level of the gene itself and of the genes
        // that have pathway edges going into the gene
        for (i in 1:P) {
            for (n in 1:N) {
                pred_p[n, pi[i]] = pred_p[n, pi[i]] + act[n, po[i]] * edge_wght[i];
            }
        }
    }

    model {
        // the weights of expression levels in calculating transcription
        // levels follow a distribution that is to be inferred
        for (g in 1:G) {
            tx_wght[g] ~ beta(tx_wght_prior[1], tx_wght_prior[2]);
        }

        // accuracies of transcription levels in measuring gene activity
        // levels follow a distribution that is to be inferred
        for (g in 1:G) {
            tx_acc[g] ~ gamma(tx_acc_prior[1], tx_acc_prior[2]);
        }
        
        // pathway edge weights follow a distribution that is to be inferred
        for (i in 1:P) {
            edge_wght[i] ~ beta(edge_wght_prior[1], edge_wght_prior[2]);
        }

        // gene activity levels are transcription levels with noise added,
        // observed protein levels are predicted protein levels plus noise
        for (g in 1:G) {
            for (n in 1:N) {
                act[n, g] ~ normal(tx[n, g], pow(tx_acc[g], -1.0));
                p[n, g] ~ normal(pred_p[n, g], 0.01);
            }
        }
    }'''


model_code_ens = '''
    data {
        int<lower=1> N;                 // number of samples
        int<lower=1> G;                 // number of genetic features
        int<lower=1> R;                 // number of known proteomes
        real<lower=0> prec;             // tuning precision

        real r[N, G];       // RNA-seq expression values
        real c[N, G];       // copy number GISTIC values
        real p[N, G];       // proteomic measurements
        real k[N, R];       // inferred proteomic measurements

        int <lower=G> P;                    // number of pathway interactions
        int <lower=1, upper=(G+R)> po[P];   // pathway out-edges
        int <lower=1, upper=(G+R)> pi[P];   // pathway in-edges
    }
    
    parameters {
        vector<lower=0, upper=1>[P] wght;   // pathway interaction weights
        vector<lower=0.1, upper=8>[2] wght_prior;

        vector<lower=0, upper=1>[G] tx_wght;   // RNA-CNA tx_wghtinations
        vector<lower=0.1, upper=8>[2] tx_wght_prior;
    }

    transformed parameters{
        matrix[N, G] tx;
        matrix[N, G] pred_p;

        for (g in 1:G) {
            for (n in 1:N) {
                tx[n, g] = (tx_wght[g] * r[n, g]) + ((1.0 - tx_wght[g]) * c[n, g]);
            }
        }

        for (g in 1:G) {
            for (n in 1:N) {
                pred_p[n, g] = 0;

                for (i in 1:P) {
                    if (pi[i] == g) {
                        
                        if (po[i] <= G) {
                            pred_p[n, g] = pred_p[n, g] + tx[n, po[i]] * wght[i];
                        
                        } else {
                            pred_p[n, g] = pred_p[n, g] + k[n, po[i] - G] * wght[i];
                        }
                    }
                }
            }
        }
    }

    model {
        for (i in 1:P) {
            wght[i] ~ beta(wght_prior[1], wght_prior[2]);
        }

        for (g in 1:G) {
            tx_wght[g] ~ beta(tx_wght_prior[1], tx_wght_prior[2]);
        }
        
        for (g in 1:G) {
            for (n in 1:N) {
                p[n, g] ~ normal(pred_p[n, g], prec);
            }
        }

    }'''

model_code_tf = """
    data {
        int <lower=1> N;        // number of samples
        int <lower=1> G;        // number of genetic features

        // set up expression data structure        
        real e[N, G];           // gene expression (log norm'd rna-seq)

        // set up rppa data structure
        real p[N, G];           // rppa

        // set up regulatory network data structure
        int <lower=G> R;                // number of tf-target edges (relationships) in the .adj file
        int <lower=1, upper=G> tf[R];   // tf nodes (in example above, see regulator vector)
        int <lower=1, upper=G> tg[R];   // target nodes (in example above, see target vector)
        real <lower=-1, upper=1> moa[R]; // mode of regulation (see aracne manual)
        real <lower=0, upper=1> lkly[R]; // likelihood (see aracne manual)

        // provide it with unique tfs and targets
        // todo: unique the tf vector and read that in here
        int <lower=1, upper=G> UTF;      // number of unique tfs
        int <lower=1, upper=G> UTG;      // number of unique tgs
        int <lower=1, upper=G> uniqtf[UTF]; // use this mapping later for UTF to uniqtf in tfa distribution assignment
        int <lower=1, upper=G> uniqtg[UTG]; // ''

    }

    parameters {
        matrix<lower=0, upper=1>[N, UTF] tfa_matrix;      // does this need to be a transformed parameter?

        vector<lower=0, upper=1>[UTF] alpha;              // make more informative? add priors in models?
        vector<lower=0, upper=1>[UTF] beta;               // '', also, "vector" sets type to real

        // other parameters?

    }

    transformed_parameters {
        matrix[N, G] pred_p;

        for (G in 1:G) {                        // for each gene
            for (n in 1:N) {                    // for each sample
                pred_p[n, g]=0;                 // initialize the pred_p's to 0
            }
        }

        for (r in 1:R) {                // for each tf-target relationship
            if (tg[r] == g) {           // if the target in that relationship is the gene of interest
                for (u in 1:UTF) {              // for each index value in the length of the number of unique tfs
                    if uniqtf[u] == tf[r] {     // use that index to grab the tf from uniqtfs (which is a necessary mapping for tfa_matrix below)
                        pred_p[n,g] = pred_p[n,g] + tfa_matrix[n,u] * moa[r] * lkly[r];     // how to handle negative moa?
                    }
                }
            }
        }                                                                                                                                                  }

    model {
        for (n in 1:N) {
            for (u in 1:UTF) {
                tfa_matrix[n, u] ~ beta(alpha[u], beta[u])
            }
        }

        for (n in 1:N) {
            for (g in 1:G) {
                p[n,g] ~ normal(pred_p[n,g], 0.01)

    }
"""


tfa_model_code = """
    data {
        int<lower=1> N;     // number of samples
        int<lower=1> G;     // number of genetic features
        real r[N, G];       // RNA-seq expression values
        real c[N, G];       // copy number GISTIC values
        real p[N, G];       // pseudo TFA measurements
        int <lower=G> P;                // number of pathway interactions
        int <lower=1, upper=G> po[P];   // pathway out-edges
        int <lower=1, upper=G> pi[P];   // pathway in-edges
    }

    parameters {
        vector<lower=0, upper=1>[P] wght;   // pathway interaction weights
        vector<lower=0.01, upper=10>[2] wght_prior;
        vector<lower=0, upper=1>[G] comb;   // RNA-CNA combinations
        vector<lower=0.01, upper=10>[2] comb_prior;
        vector<lower=0>[G] prec;   // precision of activities
        vector<lower=0.01, upper=20>[2] prec_prior;
        matrix[N, G] act;                   // inferred gene activities
    }
    transformed parameters{
        matrix[N, G] act_sum;
        matrix[N, G] pred_p;
        for (g in 1:G) {
            for (n in 1:N) {
                act_sum[n, g] = (comb[g] * r[n, g]) + ((1.0 - comb[g]) * c[n, g]);
                pred_p[n, g] = 0;
                for (i in 1:P) {
                    if (pi[i] == g)
                        pred_p[n, g] = pred_p[n, g] + act[n, po[i]] * wght[i];
                }
                // print(pred_p[n, g]);
            }
        }
    }
    model {
        for (i in 1:P) {
            wght[i] ~ beta(wght_prior[1], wght_prior[2]);
        }
        for (g in 1:G) {
            comb[g] ~ beta(comb_prior[1], comb_prior[2]);
        }

        for (g in 1:G) {
            prec[g] ~ gamma(prec_prior[1], prec_prior[2]);
        }

        for (g in 1:G) {
            for (n in 1:N) {
                act[n, g] ~ normal(act_sum[n, g], pow(prec[g], -1.0));
                p[n, g] ~ normal(pred_p[n, g], 0.01);
            }
        }
    }
"""


tfa_hier_stan_spec = """

    /* Spec for hierarchical regression linear model */

    data {                                 
        int<lower=0> N;                         // count of observations
        int<lower=0> K;                         // count of exog features
        matrix[N, K] X;                         // exog features
        vector[N] y;                            // endog feature 

        int<lower=0> n_patient;                      // count of patient index levels
        int<lower=1, upper=n_patient> patient_enc[N]; // patient index encoding  

        int<lower=0> n_tf;                     // count of tf index levels
        int<lower=1, upper=n_tf> tf_enc[N];   // tf index encoding          

        int<lower=1, upper=n_tf> tf_patient_map[n_tf];
    }
    parameters {
        vector[K] beta;                         // exog coeffs
        real<lower=0> sigma;                    // linear model error       

        real patient_mu;                         // patient mu hyperprior
        real<lower=0> patient_sd;                // patient sd hyperprior

        vector [n_patient] b0_patient;            // tf mu hyperprior (patient prior)
        real<lower=0> tf_sd;                   // tf sd hyperprior

        vector[n_tf] b0_tf;                   // tf prior
    }
    transformed parameters {}
    model {  

        patient_mu ~ normal(0, 10);              // weakly informative
        patient_sd ~ cauchy(0, 10);              // weakly informative
        tf_sd ~ cauchy(0, 10);                 // weakly informative


        for (patient in 1:n_patient) {            // patient priors 
              b0_patient[patient] ~ normal(patient_mu, patient_sd);
        }

        for (tf in 1:n_tf) {                  // tf priors 
            b0_tf[tf] ~ normal(b0_patient[tf_patient_map[tf]], tf_sd);
        }      

        sigma ~ cauchy(0, 10);                  // weakly informative noise
        y ~ student_t(1, b0_tf[tf_enc] + X * beta, sigma);    // likelihood
    }

"""
