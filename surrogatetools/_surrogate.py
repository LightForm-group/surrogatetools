from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from scipy.stats import norm, uniform
import scipy

import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np

class Surrogate:
    def __init__(self,X, Y, D=None, X_names=None):
        """_summary_

        Parameters
        ----------
        X : nd.array
            Input variables (shape nxd)
        Y : nd.array
            Output variables (shape nxp)
        D : nd.array, optional
            Design variables (shape nxq), by default None
        X_names : list[str], optional
            List of parameter names of length D, by default None
        """

        self.X = X
        self.Y = Y
        self.D = D

        self.X_mean = None
        self.Y_mean = None

        self.X_std = None
        self.Y_std = None

        self.bounds = np.array([np.min(self.X,axis=0),
                                np.max(self.X,axis=0)]).T

        self.X_names = X_names

        self.n, self.d = self.X.shape                               # input dimensionality
        self.n, self.p = self.Y.shape                               # output dimensionality
        self.n, self.q = self.D.shape if self.D else (self.n, 0)    # design dimensionality
        self.z = self.d + self.q                                    # surrogate input dimensionality

        
    def scale_data(self, scale_X=True,scale_Y=True, X_mean=None, Y_mean=None, X_std=None,Y_std=None):

        if (X_mean == None) and (scale_X == True):
            self.X_mean = self.X.mean(axis=0)
            self.X_std = self.X.std(axis=0) if X_std is None else X_std

        if (Y_mean == None) and (scale_Y == True):
            self.Y_mean = self.Y.mean(axis=0)
            self.Y_std = self.Y.std(axis=0) if Y_std is None else Y_std

        if scale_X==True:
            self.X = (self.X - self.X_mean)/self.X_std

        if scale_y==True:
            self.Y = (self.Y - self.Y_mean)/self.Y_std
        
            
        return None
    
    def build_model(self,
                    kernel='matern',
                    cross_validate=False,
                    cross_validator=5,
                    scoring='neg_mean_absolute_error',
                    **kwargs):
        
        # concatenate the design and input variables
        self.Z = np.concatenate((self.X,self.D),axis=-1)

        # Firstly we define the kernel to be used
        if kernel=='matern':
            kernel = 1.0 * Matern(length_scale=self.z*[1],nu=2.5) + 1.0

        else:
            kernel = kernel
        
        # Build the kernel along with any additional parameters

        # Optimise with cross validation
        if cross_validate==True:

            model = GaussianProcessRegressor(kernel=kernel,**kwargs)
            score = cross_val_score(model, self.Z, self.Y, cv=cross_validator, scoring=scoring)
                
            print('Cross validation score: ', np.round(score,2))

        self.model = GaussianProcessRegressor(kernel=kernel,**kwargs)

        self.model.fit(self.Z, self.Y)

        return None

    def make_prediction(self,X,D=None,return_std=False):
        """
        Make a surrogate prediction. Note that we can either have a single set of
        input parameters and multiple sets of design variables or multiple sets
        of input variables and a single or no design variable. This ensures the 
        output is interpretable. 

        Parameters
        ----------
        X : nd.array, input variables
            mxd array of input variables
        D : nd.array, optional
            lxq array of design variables, by default None
        return_std : bool, optional
            Return the uncertainty in the GP prediction, by default False

        Returns
        -------
        _type_
            _description_
        """
        
        if self.X_mean is not None:
            X = (X - self.X_mean)/self.X_std

        # multiple inputs and single design
        if X.shape[0] > 1 and D.shape[0]==1:

            D = np.array(D.tolist()*X.shape[0])

            Z = np.concatenate((X,D),axis=-1)

        # single input and multiple design 
        elif X.shape[0] == 1 and D.shape[0] > 1:

            X = np.array(X.tolist()*D.shape[0])

            Z = np.concatenate((X,D),axis=-1)

        # if no design 
        elif D == None:
            Z = X

        # raise error if inputs do not comply
        else:
            raise AssertionError()

        Y_prediction, Y_error = self.model.predict(Z,return_std=True)

        if self.Y_mean is not None:
            Y_prediction = Y_prediction*self.Y_std + self.Y_mean
            Y_error = Y_error*self.Y_std

        if return_std:
            return Y_prediction, Y_error
        else:
            return Y_prediction
        
    def make_prediction_sobol(self,X):
        
        X = np.array(X).T

        Y_prediction = self.make_prediction(X,return_std=False)

        return Y_prediction.T

    def generate_sobol(self,use_fit=False,**kwargs):
        """
        Wrapper function for scipy.stats.sobol_indices

        Args:
            n (int): number of samples

            **kwargs: additonal parameters

        Returns:
            sobol: SobolResult
        """

        if use_fit==True:
            distributions = [norm(loc=self.X_mean[i], scale=self.X_std[i]) for i in range(len(self.bounds))]

        else:
            distributions = [uniform(loc=x[0], scale = x[1]-x[0]) for x in self.bounds]

        sobol = scipy.stats.sobol_indices(func=self.make_prediction_sobol,
                                          dists=distributions,
                                          **kwargs)
        
        return sobol
    
    def fit(self,Y_actual,Y_error,use_std=True,error_scale=1.0,loss_func=None,**kwargs):

        def _loss(params,design,Y_actual,Y_error):

            params = np.array(params).reshape(1, -1)

            if use_std==True:

                Y_prediction,Y_prediction_error = self.make_prediction(X=params,return_std=True)

                residual_square = (Y_prediction[0] - Y_actual)**2 + error_scale*Y_prediction_error**2

            else:
                Y_prediction = self.make_prediction(X=params)

                residual_square = (Y_prediction[0] - Y_actual)**2

            error_residual_square = Y_error**2

            loss = residual_square / error_residual_square
                                
            return loss.sum()
        
        loss = loss_func if loss_func is not None else _loss

        res = scipy.optimize.shgo(loss,
                                bounds=self.bounds,
                                args=(design, Y_actual, Y_error),
                                **kwargs)

        return res

    def perfom_inference(self,Y_actual,Y_error,initval=None,use_std=True,error_scale=1.0,loss_func=None,**kwargs):

        def my_loglike(params,design, data):
            # We fail explicitly if inputs are not numerical types for the sake of this tutorial
            # As defined, my_loglike would actually work fine with PyTensor variables!

            design = design
            params = np.array(params).reshape(1, -1)
            
            y_actual, y_actual_error = data

            if loss_func is None:
                # The surrogate must provide a prediction and uncertainty estimate
                Y_prediction, Y_prediction_error = self.make_prediction(params, return_std=True)

                if use_std == True:
                    residual_square = (Y_prediction[0] - y_actual)**2 + error_scale*Y_prediction_error[0]**2

                else:
                    residual_square = (Y_prediction[0] - y_actual)**2

                loss = residual_square / y_actual_error**2

            else: 
                loss = loss_func(params,Y_actual,Y_error)

            f = -1 * loss
            
            return f 

        class LogLike(Op):
            def make_node(self, params, design, data) -> Apply:
                # Convert inputs to tensor variables
                params = pt.as_tensor(params)
                design = pt.as_tensor(design)
                data = pt.as_tensor(data)

                inputs = [params, design, data]
                # Define output type, in our case a vector of likelihoods
                # with the same dimensions and same data type as data
                # If data must always be a vector, we could have hard-coded
                outputs = [pt.vector()]

                # Apply is an object that combines inputs, outputs and an Op (self)
                return Apply(self, inputs, outputs)

            def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
                # This is the method that compute numerical output
                # given numerical inputs. Everything here is numpy arrays
                params, design, data = inputs  # this will contain my variables

                # call our numpy log-likelihood function
                loglike_eval = my_loglike(params, design, data)

                # Save the result in the outputs list provided by PyTensor
                # There is one list per output, each containing another list
                # pre-populated with a `None` where the result should be saved.
                outputs[0][0] = np.asarray(loglike_eval)

        data = [Y_actual,Y_error]

        loglike_op = LogLike()

        def custom_dist_loglike(data, params, design):

            # create our Op
            # data, or observed is always passed as the first input of CustomDist
            return loglike_op(params, design, data)

        # use PyMC to sampler from log-likelihood
        with pm.Model() as no_grad_model:

            params = []

            for i in range(self.D):
                
                distribution = pm.Uniform(self.X_names[i], 
                                        lower=self.bounds[i][0], 
                                        upper=self.bounds[i][1],
                                        initval=initval[i] if initval is not None else None)
                
                params.append(distribution)

            # use a CustomDist with a custom logp function
            likelihood = pm.CustomDist(
                "likelihood", params, design, observed=data, logp=custom_dist_loglike,
            )

        ip = no_grad_model.initial_point()

        no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip)

        with no_grad_model:

            step = pm.DEMetropolisZ()

            # Use custom number of draws to replace the HMC based defaults
            idata_no_grad = pm.sample(step=step,**kwargs) #50_000, tune=50_000,cores=4,chains=4,step=step,return_inferencedata=True)
        
        return idata_no_grad