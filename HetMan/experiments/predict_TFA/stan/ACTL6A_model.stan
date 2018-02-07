data {
	int N;
	real ACTL6A[N];
	real NCL[N];
	real NME1[N];
	real PTMA[N];
	real GPAM[N];
	real CAD[N];
	real HSPD1[N];
}

parameters {
	real b1;
	real b2;
	real b3;
	real b4;
	real b5;
	real b6;
	real<lower=0> sigma;
}

model {
	for (i in 1:N)
		ACTL6A[i] ~ normal(b1 + b1 * NCL[i] + b2 * NME1[i] + b3 * PTMA[i] + b4 * GPAM[i] + b5 * CAD[i] + b6 * HSPD1[i], sigma);
}