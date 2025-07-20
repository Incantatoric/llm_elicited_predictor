import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class HanwhaBayesianModel:
    """
    Bayesian linear regression with LLM-elicited priors for 한화솔루션
    """
    
    def __init__(self, priors_dir: str = "priors"):
        self.priors_dir = priors_dir
        self.priors = self.load_priors()
        self.model = None
        self.trace = None
        logger.info(f"Loaded {len(self.priors)} priors from {priors_dir}")
    
    def load_priors(self) -> np.ndarray:
        """Load all prior arrays and stack them"""
        priors = []
        i = 0
        while True:
            prior_file = f"{self.priors_dir}/hanwha_prior_{i}.npy"
            try:
                prior = np.load(prior_file)
                priors.append(prior)
                i += 1
            except FileNotFoundError:
                break
        
        if len(priors) == 0:
            raise ValueError(f"No prior files found in {self.priors_dir}")
        
        return np.array(priors)  # Shape: (n_priors, n_features+1, 2)
    
    def create_mixture_model(self, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """
        Create Bayesian linear regression with mixture of Gaussian priors
        Following the paper's exact approach
        """
        n_features = X.shape[1]
        n_priors = len(self.priors)
        
        logger.info(f"Creating Bayesian model with {n_priors} mixture components")
        logger.info(f"Features: {n_features}, Observations: {len(y)}")
        
        with pm.Model() as model:
            # Equal mixture weights (as in paper)
            mixture_weights = np.ones(n_priors) / n_priors
            
            # Create mixture priors for each parameter
            betas = []
            
            for param_idx in range(n_features + 1):  # +1 for intercept
                # Extract means and stds for this parameter from all priors
                means = self.priors[:, param_idx, 0]  # All means for this parameter
                stds = self.priors[:, param_idx, 1]   # All stds for this parameter
                
                # Create mixture of normals for this parameter
                beta_param = pm.NormalMixture(
                    f'beta_{param_idx}',
                    w=mixture_weights,
                    mu=means,
                    sigma=stds
                )
                betas.append(beta_param)
            
            # Linear regression equation
            intercept = betas[0]
            feature_betas = pm.math.stack(betas[1:])
            
            # Expected value
            mu = intercept + pm.math.dot(X, feature_betas)
            
            # Noise term (uninformative as in paper)
            sigma = pm.HalfCauchy('sigma', beta=1.0)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
        self.model = model
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   n_samples: int = 5000, n_chains: int = 4) -> az.InferenceData:
        """Train the Bayesian model using MCMC"""
        logger.info(f"Training Bayesian model with {n_samples} samples, {n_chains} chains")
        
        if self.model is None:
            self.create_mixture_model(X, y)
        
        with self.model:
            # Sample from posterior
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                cores=min(4, n_chains),
                return_inferencedata=True,
                random_seed=42
            )
        
        logger.info("Model training completed")
        return self.trace
    
    def predict(self, X_new: np.ndarray, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""
        if self.trace is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(X_new)} data points")
        
        with self.model:
            # Sample from posterior predictive distribution
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace, 
                predictions=True,
                extend_inferencedata=False,
                random_seed=42
            )
        
        predictions = {
            'mean': posterior_predictive['y_obs'].mean(axis=0),
            'std': posterior_predictive['y_obs'].std(axis=0),
            'samples': posterior_predictive['y_obs'],
            'quantiles': {
                '5%': np.percentile(posterior_predictive['y_obs'], 5, axis=0),
                '25%': np.percentile(posterior_predictive['y_obs'], 25, axis=0),
                '75%': np.percentile(posterior_predictive['y_obs'], 75, axis=0),
                '95%': np.percentile(posterior_predictive['y_obs'], 95, axis=0)
            }
        }
        
        return predictions

def create_uninformative_model(X: np.ndarray, y: np.ndarray) -> pm.Model:
    """Create baseline model with uninformative priors for comparison"""
    n_features = X.shape[1]
    
    with pm.Model() as model:
        # Uninformative priors (as in paper)
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=n_features)
        
        # Linear regression
        mu = intercept + pm.math.dot(X, betas)
        sigma = pm.HalfCauchy('sigma', beta=1.0)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    return model