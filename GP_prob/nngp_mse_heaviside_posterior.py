
inference_method = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
lik = GPy.likelihoods.gaussian.Gaussian(variance=0.002)
kernel = CustomMatrix(Xfull.shape[1],Xfull,Kfull)
gp_model = GPy.core.GP(X=X,Y=Y,kernel=kernel,inference_method=inference_method, likelihood=lik)
