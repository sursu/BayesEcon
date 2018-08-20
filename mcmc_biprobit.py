def mcmc_biprobit(y1, y2, x1, x2, itr=10000, burnin=1000, sample_rho=False, 
                  B0=100, rho_mean0=0, rho_var0=1, rho_s2=None, rho_start=0,
                  beta1_start=None, beta2_start=None):
    """
    Function to perform MCMC from the bivariate Probit model
    
    INPUT: 
        y1, y2 ~ observed outcomes
        x1, x2 ~ explanatory variables
        	x1: (x, y_2)
        	x2:	(x,  z )
        
    OUTPUT: 
        {'beta1', 'beta2', 'rho'}
    """
    # numbers of observations and of covariates
    nobs = len(y1)
    k1 = x1.shape[1]
    k2 = x2.shape[1]
    
    if rho_s2 is None:
        rho_s2 = 1/nobs
    
    # save cross-products to avoid recomputing at each MCMC interation
    x = np.column_stack([x1, x2])
    xx = x.T@x

    # generate useful ingredients for sampling of latent utilities
    # create vector of lower and upper bounds for latent utilities
    y1_lower, y1_upper = np.zeros(nobs), np.zeros(nobs)
    y2_lower, y2_upper = np.zeros(nobs), np.zeros(nobs)
    y1_lower[y1 == 0], y1_upper[y1 == 1] = -np.inf, np.inf
    y2_lower[y2 == 0], y2_upper[y2 == 1] = -np.inf, np.inf
    
    # containers for MCMC draws
    MCMC_beta1 = np.full(shape=(itr, k1), fill_value=np.nan)
    MCMC_beta2 = np.full(shape=(itr, k2), fill_value=np.nan)
    MCMC_rho = np.full(shape=itr, fill_value=np.nan)

    # starting values
    if beta1_start is None:
        beta1 = np.zeros(k1)
    else:
        beta1 = beta1_start
    if beta2_start is None:
        beta2 = np.zeros(k2)
    else:
        beta2 = beta2_start
    ystar1 = abs(np.random.normal(size=nobs)) * np.sign(y1 - 0.5)
    ystar2 = abs(np.random.normal(size=nobs)) * np.sign(y2 - 0.5)
    rho = rho_start
    
    # ----- MCMC sampling
    for rep in range(burnin + itr):
        
        if (rep % 100 == 0):
            print("rep =", rep)

        # ---- sample latent utilities
        y1_mean, y2_mean = x1@beta1, x2@beta2
        
        sd = np.sqrt(1 - rho**2)
        mean = y1_mean + rho * (ystar2 - y2_mean)
        ystar1 = stats.truncnorm.rvs(a = (y1_lower-mean)/sd, b = (y1_upper-mean)/sd, 
                                     loc = mean, scale = sd)
        mean = y2_mean + rho * (ystar1 - y1_mean)
        ystar2 = stats.truncnorm.rvs(a = (y2_lower-mean)/sd, b = (y2_upper-mean)/sd,
                                     loc = mean, scale = sd)

        # ---- sample regression coefficients
        post_var = xx / sd
        post_var[:k1, k1:(k1+k2)] = -rho * post_var[:k1, k1:(k1+k2)]
        post_var[k1:(k1+k2), :k1] = -rho * post_var[k1:(k1+k2), :k1]
        post_var = post_var + np.identity(k1 + k2) / B0
        post_var = np.linalg.inv(post_var)

        post_mean = np.append(
        # np.append(x1.T@(ystar1 - rho*ystar2), x2.T@(ystar2 - rho*ystar1))
        	np.sum(x1.T*(ystar1 - rho*ystar2), axis=-1), 
                              np.sum(x2.T*(ystar2 - rho*ystar1), axis=-1))
        post_mean = post_var@post_mean / sd

        beta = np.random.multivariate_normal(mean = post_mean, cov = post_var)
        beta1 = beta[:k1]
        beta2 = beta[k1:(k1+k2)]

        # ---- sample correlation coefficient
        if (sample_rho):
            # make proposal
            rho_old = rho
            rho_new = rho_old + np.random.normal(scale=np.sqrt(rho_s2))
            if (abs(rho_new) < 1):
                # compute log of likelihoods ratio
                yd = np.column_stack([ystar1 - x1@beta1, ystar2 - x2@beta2])
                sigma_old = np.array([[1, rho_old],[rho_old, 1]])
                sigma_new = np.array([[1, rho_new],[rho_new, 1]])
                ln_ratio_llik = np.sum(np.log(stats.multivariate_normal.pdf(yd, cov=sigma_new))) -\
                                np.sum(np.log(stats.multivariate_normal.pdf(yd, cov=sigma_old)))
                # compute log of prior densities ratio
                ln_ratio_prior = np.log(stats.norm.pdf(rho_new, loc=rho_mean0, scale=np.sqrt(rho_var0))) -\
                                 np.log(stats.norm.pdf(rho_old, loc=rho_mean0, scale=np.sqrt(rho_var0)))
                # compute acceptance ratio
                ln_R = ln_ratio_llik + ln_ratio_prior
                alpha = min(1, np.exp(ln_R))
                # accept or reject proposal
                if (np.random.uniform() <= alpha):
                    rho = rho_new
                    
        # ---- save draws after burnin
        if (rep > burnin-1):
            j = rep - burnin
            MCMC_beta1[j] = beta1
            MCMC_beta2[j] = beta2
            MCMC_rho[j] = rho

    # end of MCMC loop

    # ---- label and return MCMC draws    
    return {'beta1': MCMC_beta1, 'beta2': MCMC_beta2, 'rho': MCMC_rho}