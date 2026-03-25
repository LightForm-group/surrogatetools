from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, uniform
import scipy
from collections.abc import Callable

import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np

class Surrogate:
    def __init__(self,
                 X : np.ndarray, 
                 y : np.ndarray,
                 parameter_names : list[str] | None = None):
        """
        Surrogate initialisation parameters

        Parameters
        ----------
        X : np.ndarray
            X data of dimensionality NxD
        y : np.ndarray
            Y data of dimensionality NxP
        parameter_names : list[str] | None, optional
            Name of each input dimension of X, by default None which automatically 
            generates alphabetical names
        """

        self.X = X
        self.y = y

        self.X_scaler = None
        self.y_scaler = None

        self.parameter_range = np.array([np.min(self.X,axis=0),
                                        np.max(self.X,axis=0)]).T

        self.N, self.D = self.X.shape
        self.N, self.P = self.y.shape

        # automatically generate parameter names if they are not provided
        self.parameter_names = parameter_names if parameter_names is not None else  [chr(i) for i in range(self.D)]


    def scale_data(self, X_scaler = StandardScaler(), y_scaler = StandardScaler()):
        """
        Scale input and output parameters, using Sci-kit scalers by default. Custom
        scalers can be used but must have fit(), transform() and inverse_transform() 
        functionality. The scalers must also have a scale_ property to allow for 
        uncertainty in predictions to be transformed. 

        Parameters
        ----------
        X_scaler : Callable, optional
            Scaler function to use, by default sklearn.preprocessing.StandardScaler()
        y_scaler : Callable, optional
            Scaler function to use, by default sklearn.preprocessing.StandardScaler()
        """

        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

        self.X_scaler.fit(self.X)
        self.y_scaler.fit(self.y)

        self.X = X_scaler.transform(self.X)
        self.y = y_scaler.transform(self.y)
        
        return None
    
    def build_model(self,
                    kernel = 'matern',
                    cross_validate : bool = False,
                    cross_validator : int = 5,
                    scoring : str ='neg_mean_absolute_error',
                    **kwargs):

        # Firstly we define the kernel to be used
        if kernel=='matern':
            kernel = 1.0 * Matern(length_scale=self.D*[1],nu=2.5) 
        else:
            kernel = kernel
        
        # Build the kernel along with any additional parameters

        # Optimise with cross validation
        if cross_validate==True:

            model = GaussianProcessRegressor(kernel=kernel,**kwargs)
            score = cross_val_score(model, self.X, self.y, cv=cross_validator,scoring=scoring)
                
            print('Cross validation score: ', np.round(score,2))

        self.model = GaussianProcessRegressor(kernel=kernel,**kwargs)

        self.model.fit(self.X, self.y)

        return None

    def make_prediction(self,X : np.ndarray,
                        ):
        
        if self.X_scaler is not None:
            X = self.X_scaler.transform(X)

        y_prediction, y_prediction_error = self.model.predict(X,return_std=True)

        if self.y_scaler is not None:
            y_prediction = self.y_scaler.inverse_transform(y_prediction)
            y_prediction_error = self.y_scaler.scale_*y_prediction_error

        return y_prediction,  y_prediction_error
        
    def make_prediction_sobol(self,
                              X : np.ndarray
                              ):
        
        X = np.array(X).T

        y_prediction, _ = self.make_prediction(X)

        return y_prediction.T

    def generate_sobol(self,
                       use_fit : bool = False,
                       **kwargs
                       ):
        """
        Wrapper function for scipy.stats.sobol_indices

        Args:
            n (int): number of samples

            **kwargs: additonal parameters

        Returns:
            sobol: SobolResult
        """

        if use_fit==True:
            distributions = [norm(loc=self.parameter_mean[i], scale=self.parameter_std[i]) for i in range(len(self.parameter_range))]

        else:
            distributions = [uniform(loc=x[0], scale = x[1]-x[0]) for x in self.parameter_range]

        sobol = scipy.stats.sobol_indices(func=self.make_prediction_sobol,
                                          dists=distributions,
                                          **kwargs)
        
        return sobol
    
    def fit(self,
            Y_actual : np.ndarray,
            Y_error : np.ndarray,
            error_scale : float = 0.0,
            loss_func: Callable | None = None,
            **kwargs):

        def _loss(params,Y_actual,Y_error):

            params = np.array(params).reshape(1, -1)

            y_prediction,y_prediction_error = self.make_prediction(X=params)

            loss = (y_prediction - Y_actual)**2/Y_error**2 + error_scale*y_prediction_error**2/Y_error**2
                                
            return loss.sum()
        
        loss = loss_func if loss_func is not None else _loss

        res = scipy.optimize.shgo(  loss,
                                    bounds=self.parameter_range,
                                    args=(Y_actual, Y_error),
                                    **kwargs
                                    )
        return res

    def perfom_inference(self,Y_actual,Y_error,initval=None,error_scale=0.0,loss_func=None,import_prior=None,**kwargs):

        data = [Y_actual,Y_error]

        def my_loglike(params,data):
            params = np.array(params).reshape(1, -1)
            
            y_actual, y_error = data

            if loss_func is None:
                # The surrogate must provide a prediction and uncertainty estimate
                y_prediction, y_prediction_error = self.make_prediction(params)

                loss = (y_prediction - y_actual)**2/y_error**2 + error_scale*y_prediction_error**2/y_error**2

            else: 
                loss = loss_func(params,Y_actual,Y_error)

            f = -1 * loss
            
            return f 

        class LogLike(Op):
            def make_node(self, params, data) -> Apply:
                # Convert inputs to tensor variables
                params = pt.as_tensor(params)
                data = pt.as_tensor(data)

                inputs = [params, data]
                # Define output type, in our case a vector of likelihoods
                outputs = [pt.vector()]

                # Apply is an object that combines inputs, outputs and an Op (self)
                return Apply(self, inputs, outputs)

            def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
                # This is the method that compute numerical output
                # given numerical inputs. Everything here is numpy arrays
                params, data = inputs  # this will contain my variables

                # call our numpy log-likelihood function
                loglike_eval = my_loglike(params, data)

                # Save the result in the outputs list provided by PyTensor
                # pre-populated with a `None` where the result should be saved.
                outputs[0][0] = np.asarray(loglike_eval)


        loglike_op = LogLike()

        def custom_dist_loglike(data, params):

            # create our Op
            return loglike_op(params, data)

        # use PyMC to sampler from log-likelihood
        with pm.Model() as no_grad_model:

            params = []

            if import_prior is not None:
                for i in range(self.D):
 
                    samples = import_prior[:,i]

                    smin,smax = self.parameter_range[i]

                    x = np.linspace(smin, smax, 100)
                    y = scipy.stats.gaussian_kde(samples)(x)

                    params.append(pm.Interpolated(self.parameter_names[i], x, y))
            else:

                for i in range(self.D):
                    
                    distribution = pm.Uniform(self.parameter_names[i], 
                                            lower=self.parameter_range[i][0], 
                                            upper=self.parameter_range[i][1],
                                            initval=initval[i] if initval is not None else None)
                    
                    params.append(distribution)

            # use a CustomDist with a custom logp function
            likelihood = pm.CustomDist(
                "likelihood", params, observed=data, logp=custom_dist_loglike,
            )

        ip = no_grad_model.initial_point()

        no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip)

        with no_grad_model:

            step = pm.DEMetropolisZ()

            # Use custom number of draws to replace the HMC based defaults
            idata_no_grad = pm.sample(step=step,**kwargs) #50_000, tune=50_000,cores=4,chains=4,step=step,return_inferencedata=True)
        
        return idata_no_grad