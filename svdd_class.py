import numpy as np

class Instance(object):
    """A class to model the Gurobi model and tupledicts of variables. b is the potential set of centers for the discrete case."""
    def __init__(self, data, ntrain, nval, ntest, y, Nu=0.1, kernel='linear', sigma=None, degree=None, k=1, max_iter=20, random_state=0, weights=None):
        """
        Initializes an instance with data and model parameters.
        
        Params:
        -------
            data (np.ndarray): Complete dataset with samples stored by rows.
            ntrain (int): Number of training samples.
            nval (int): Number of validation samples.
            ntest (int): Number of test samples.
            y (np.ndarray): Labels for all samples.
            Nu (float): Nu parameter controlling the upper bound on the anomaly fraction.
            kernel (str): Kernel type used by the model.
            sigma (float or None): Kernel width used by the RBF kernel.
            degree (int or None): Polynomial degree used by the polynomial kernel.
            k (int): Number of spheres or clusters requested.
            max_iter (int): Maximum number of iterations for iterative methods.
            random_state (int): Random seed used by clustering methods.
            weights (np.ndarray or None): Optional training weights. If None, uniform weights are used.
        
        Returns:
        --------
            None: The instance attributes are stored in the object.
        """
        self.data = data
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.y = y
        self.Nu = Nu
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree

        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

        if weights is None:
            self.weights = np.ones(ntrain)
        else:
            self.weights = np.asarray(weights).reshape(-1)
            if len(self.weights) != ntrain:
                raise ValueError("weights debe tener longitud ntrain.")

class Solution(object):
    """A class to model the Gurobi model and tupledicts of variables"""
    def __init__(self, x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, obj, runtime, nodes, fingerprint, c=None, R=None, xi=None, alpha=None, auc_roc=None, sub_opt=False):
        """
        Initializes a fitted SVDD solution object.
        
        Params:
        -------
            x (np.ndarray): Training data used to fit the SVDD model.
            data (np.ndarray): Complete dataset used in the experiment.
            ntrain (int): Number of training samples.
            nval (int): Number of validation samples.
            ntest (int): Number of test samples.
            y (np.ndarray): Labels for all samples.
            Nu (float): Nu parameter used to fit the model.
            kernel (str): Kernel type used by the model.
            sigma (float or None): Kernel width used by the RBF kernel.
            degree (int or None): Polynomial degree used by the polynomial kernel.
            obj (float or None): Objective value returned by the optimizer.
            runtime (float or None): Runtime reported by the optimizer.
            nodes (int or None): Number of branch-and-bound nodes reported by the optimizer.
            fingerprint (int or None): Gurobi model fingerprint.
            c (np.ndarray or None): Center of the primal SVDD sphere.
            R (float or None): Squared radius of the SVDD sphere.
            xi (list or None): Slack variables from the primal solution.
            alpha (np.ndarray or None): Dual variables from the dual solution.
            auc_roc (float or None): AUC-ROC value associated with the solution.
            sub_opt (bool): Whether the optimizer stopped with a suboptimal solution.
        
        Returns:
        --------
            None: The solution attributes are stored in the object.
        """
        self.x=x
        self.data=data
        self.ntrain=ntrain
        self.nval=nval
        self.ntest=ntest
        self.y=y
        self.Nu=Nu
        self.kernel=kernel
        self.sigma=sigma
        self.degree=degree
        
        self.obj=obj
        self.runtime=runtime
        self.nodes=nodes
        self.fingerprint=fingerprint

        self.c=c
        self.R=R
        self.xi=xi
        self.alpha=alpha
        
        self.auc_roc=auc_roc
        
        self.sub_opt=sub_opt
        
class SklearnSolution(object):
    def __init__(self, model, model_type,data, ntrain, nval, ntest, y,Nu, kernel, sigma, degree,runtime=None):
        """
        Initializes an object for fitted sklearn SVM models.
        
        Params:
        -------
            model (object): Fitted sklearn model.
            model_type (str): Type of model, usually 'ocsvm' or 'svm'.
            data (np.ndarray): Complete dataset used in the experiment.
            ntrain (int): Number of training samples.
            nval (int): Number of validation samples.
            ntest (int): Number of test samples.
            y (np.ndarray): Labels for all samples.
            Nu (float): Nu parameter used for OCSVM.
            kernel (str): Kernel type used by the model.
            sigma (float or None): Kernel width used by the RBF kernel.
            degree (int or None): Polynomial degree used by the polynomial kernel.
            runtime (float or None): Training runtime in seconds.
        
        Returns:
        --------
            None: The sklearn solution attributes are stored in the object.
        """
        self.model = model
        self.model_type = model_type  # "ocsvm" o "svm"

        self.data = data
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.y = y

        self.Nu = Nu
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree


        self.obj = float("nan")
        self.runtime = runtime
        self.nodes = 0
        self.fingerprint = None

        self.x = data[:ntrain, :]
        self.c = None
        self.R = float("nan")
        self.xi = None
        self.alpha = None
        self.auc_roc = None
        self.sub_opt = False

class MultiSphereSolution(object):
    def __init__(self, components, labels_train, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, obj=None, runtime=None):
        """
        Initializes a multi-sphere solution object.
        
        Params:
        -------
            components (list): List of fitted single-sphere SVDD solutions.
            labels_train (np.ndarray): Final cluster label assigned to each training sample.
            data (np.ndarray): Complete dataset used in the experiment.
            ntrain (int): Number of training samples.
            nval (int): Number of validation samples.
            ntest (int): Number of test samples.
            y (np.ndarray): Labels for all samples.
            Nu (float): Nu parameter used to fit the model.
            kernel (str): Kernel type used by the model.
            sigma (float or None): Kernel width used by the RBF kernel.
            degree (int or None): Polynomial degree used by the polynomial kernel.
            obj (float or None): Aggregated objective value of all components.
            runtime (float or None): Aggregated runtime of all components.
        
        Returns:
        --------
            None: The multi-sphere solution attributes are stored in the object.
        """
        self.components = components      # lista de soluciones SVDD individuales
        self.labels_train = labels_train  # asignación final de train
        self.data = data
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.y = y
        self.Nu = Nu
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.obj = obj
        self.runtime = runtime
        self.k = len(components)

        # Para compatibilidad en el caso primal
        self.c = np.array([s.c for s in components if s.c is not None])
        self.R = np.array([s.R for s in components])
        self.sub_opt = any(getattr(component, "sub_opt", False) for component in components)

        self.infeasible = (len(components) == 0 or len(components) == 0 or any(component.obj is None and not getattr(component, "sub_opt", False) for component in components))
        self.nodes = sum(getattr(component, "nodes", 0) for component in components)

