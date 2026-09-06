import numpy as np
from gurobipy import Model, GRB, quicksum
from sklearn.svm import OneClassSVM, SVC
import time
from svdd_class import (Solution, SklearnSolution, MultiSphereSolution)
from utilities import (Kernel, fit_svdds_by_labels, predict_scores_multisphere)


def Primal_SVDD(instance):
    """
    Solves the primal linear SVDD formulation with Gurobi.
    
    Params:
    -------
        instance (Instance): Instance containing training data, labels, parameters, and optional weights.
    
    Returns:
    --------
        Solution: Fitted SVDD solution containing the center, squared radius, slack variables, and solver information.
    """
    # Leer instancia
    data = instance.data
    y = instance.y
    
    Nu = instance.Nu
    kernel  = instance.kernel
    sigma = instance.sigma
    degree = instance.degree
    
    ntrain = int(instance.ntrain)
    nval = int(instance.nval)
    ntest = int(instance.ntest)
    
    train = np.array(range(ntrain), dtype='i')

    C = 1 / (Nu * instance.ntrain)
    weights = instance.weights
    C_i = C * weights
    
    x = data[train, :]
    
    # Crear el modelo
    model = Model('svdd_primal')
    
    n = x.shape[0]
    d = x.shape[1]

    print("Primal - Nº Datos:", n, " - Dim:", d, ' - Nu:', Nu, ' - C:', C)

    ##Index sets:
    N = range(n)
    D = range(d)

    ## Variables:
    c = model.addVars(d, lb=-GRB.INFINITY, name="c")# centers
    R  = model.addVar(name="R")# radius
    xi = model.addVars(n, name="xi")# error
    
    ## Define auxiliary vars to compute norms
    v = {}
    for i in N:
        for k in D:
            v[i,k] = x[i,k]-c[k]
    
    ## Obj
    obj = R + quicksum(C_i[i]*xi[i] for i in N)
    model.setObjective(obj, GRB.MINIMIZE)
    
    ## Ctrs        
    for i in N:
        ### SVDD Constr -- SOC
        model.addConstr(quicksum(v[i,k]*v[i,k] for k in D) <= R + xi[i], name="R7")
    
    model.update()
    
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = 3600

    # Optimizar
    model.optimize()

    if model.Status == GRB.Status.OPTIMAL:
        
        print("Obj Primal: ", model.ObjVal, "CPU Time: ", model.Runtime)

        csol=np.array([c[l].X for l in D])

        Rsol = R.X
        xisol = [xi[i].X for i in N]
        
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, model.ObjVal, model.Runtime, model.NodeCount, model.Fingerprint, csol, Rsol, xisol)
    
    elif (model.Status == GRB.INFEASIBLE):
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
    elif (model.Status == GRB.SUBOPTIMAL):
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
        sol.sub_opt = True
    else:
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, None, None, model.Fingerprint)

    return sol

def Dual_SVDD(instance) -> Solution:
    """
    Solves the dual SVDD formulation with Gurobi.
    
    Params:
    -------
        instance (Instance): Instance containing training data, labels, kernel parameters, and optional weights.
    
    Returns:
    --------
        Solution: Fitted SVDD solution containing dual variables, radius information, and solver information.
    """
    # Leer instancia
    data = instance.data
    y = instance.y

    ntrain = int(instance.ntrain)
    nval = int(instance.nval)
    ntest = int(instance.ntest)

    train = np.array(range(ntrain), dtype='i')

    x = data[train, :]
    Nu = instance.Nu
    C = 1 / (Nu * instance.ntrain)
    weights = instance.weights
    C_i = C * weights
    kernel = instance.kernel
    sigma = instance.sigma
    degree = instance.degree    
    
    model = Model("svdd_dual")

    n = ntrain
    d = x.shape[1]

    print("Dual - Nº Datos:", n, " - Dim:", d, ' - Nu:', Nu, ' - C:', C)
    
    N = range(n)

    #### Kernelized Constraints:
    K = Kernel(x, x, kernel=kernel, sigma=sigma, degree=degree)
    
    if K is None:
        raise ValueError("Kernel computation failed. Check kernel parameters.")
    
    ## Vars
    # alpha=model.addVars(n, lb=0.0, ub=C, name="alpha")
    alpha = model.addVars(n, lb=0.0, name="alpha")

    ## Obj
    obj = - quicksum(quicksum(alpha[i]*alpha[k]*K[i,k] for k in N) for i in N) + quicksum(alpha[i]*K[i,i] for i in N)
    model.setObjective(obj, GRB.MAXIMIZE)

    ## Ctrs
    for i in N:
        model.addConstr(alpha[i] <= C_i[i], name=f"alpha_ub_{i}")
    model.addConstr(quicksum(alpha[i] for i in N) <= 1+1e-6, name="R14-1")
    model.addConstr(quicksum(alpha[i] for i in N) >= 1-1e-6, name="R14-2")
    model.update()

    model.Params.OutputFlag = 1
    model.optimize()

    if model.Status == GRB.Status.OPTIMAL:
        print("Obj Dualized: ", model.ObjVal, "CPU Time: ", model.Runtime)
        
        alphasol=np.empty((n,1))
        for i in N:
            alphasol[i] = alpha[i].X

        Dsv = [i for i in N if alpha[i].X > 0.0]
        # k_aux = [i for i in Dsv if (alpha[i].X < C and alpha[i].X > 0.0)]
        tol = 1e-6
        k_aux = [i for i in Dsv if alpha[i].X > tol and alpha[i].X < C_i[i] - tol]

        # kindex = max(k_aux, key=lambda i: alpha[i].X)
        if len(k_aux) > 0:
            kindex = max(k_aux, key=lambda i: alpha[i].X)
        else:
            kindex = max(Dsv, key=lambda i: alpha[i].X)

        Rsol = K[kindex,kindex] - 2*sum(alpha[i].X * K[i,kindex] for i in Dsv) + sum(sum(alpha[i].X * alpha[j].X * K[i,j] for j in Dsv) for i in Dsv)

        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, model.ObjVal, model.Runtime, model.NodeCount, model.Fingerprint, None, Rsol, None, alphasol)
        
    elif (model.Status == GRB.INFEASIBLE):
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
    elif (model.Status == GRB.SUBOPTIMAL):
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
        sol.sub_opt = True
    else:
        sol = Solution(x, data, ntrain, nval, ntest, y, Nu, kernel, sigma, degree, None, None, None, model.Fingerprint)
        
    return sol

def Sklearn_SVM(instance, method="ocsvm", train_ocsvm_on_normals=False):
    """
    Fits an sklearn OCSVM or supervised SVM model.
    
    Params:
    -------
        instance (Instance): Instance containing data, labels, and model parameters.
        method (str): Model to fit: 'ocsvm' for OneClassSVM or 'svm' for supervised SVC.
        train_ocsvm_on_normals (bool): Whether to fit OCSVM only on samples labeled as normal.
    
    Returns:
    --------
        SklearnSolution: Object containing the fitted sklearn model and experiment data.
    """
    X_train = instance.data[:instance.ntrain, :]
    y_train = instance.y[:instance.ntrain]

    if instance.kernel == "rbf":
        kernel = "rbf"
        # gamma = 1 / instance.sigma
        gamma = 1.0 / (2.0 * instance.sigma**2)
    else:
        kernel = "linear"
        gamma = "scale"

    t0 = time.perf_counter()

    if method == "ocsvm":
        if train_ocsvm_on_normals:
            X_fit = X_train[y_train == 1]
        else:
            X_fit = X_train

        model = OneClassSVM(kernel=kernel, nu=instance.Nu, gamma=gamma)
        model.fit(X_fit)

    elif method == "svm":
        y_train_bin = np.where(y_train == -1, 1, 0)

        if len(np.unique(y_train_bin)) < 2:
            raise ValueError("SVM supervisado necesita ejemplos normales y anómalos en train.")

        # Utilizar Nu como en SVDD para construir C
        C = 1 / (instance.Nu * instance.ntrain)

        model = SVC(kernel=kernel, gamma=gamma, C=C, class_weight="balanced")

        model.fit(X_train, y_train_bin)

    else:
        raise ValueError(f"Método sklearn no reconocido: {method}")

    runtime = time.perf_counter() - t0

    return SklearnSolution(model=model, model_type=method, data=instance.data, ntrain=instance.ntrain, nval=instance.nval, ntest=instance.ntest, y=instance.y, Nu=instance.Nu, kernel=instance.kernel, sigma=instance.sigma, degree=instance.degree, runtime=runtime)


from sklearn.cluster import KMeans

def Cluster_SVDD(instance):
    """
    Fits the iterative ClusterSVDD model.
    The algorithm alternates between fitting SVDD components and reassigning training samples to the closest component.
    
    Params:
    -------
        instance (Instance): Instance containing data, parameters, and the requested number of clusters.
    
    Returns:
    --------
        MultiSphereSolution: Fitted multi-sphere solution after iterative label reassignment and SVDD refitting.
    """
    xtrain = instance.data[:instance.ntrain, :]

    km = KMeans(n_clusters=instance.k, random_state=instance.random_state, n_init=10)

    labels = km.fit_predict(xtrain)
    final_components = None
    final_labels = labels
    final_obj = None
    final_runtime = None

    for it in range(instance.max_iter):
        components, total_obj, total_runtime = fit_svdds_by_labels(instance, labels)

        # Si por algún motivo se han perdido esferas, paramos
        if len(components) < instance.k:
            print(f"ClusterSVDD detenido: solo quedan {len(components)} esferas.")
            break

        temp_sol = MultiSphereSolution(
            components=components,
            labels_train=labels,
            data=instance.data,
            ntrain=instance.ntrain,
            nval=instance.nval,
            ntest=instance.ntest,
            y=instance.y,
            Nu=instance.Nu,
            kernel=instance.kernel,
            sigma=instance.sigma,
            degree=instance.degree,
            obj=total_obj,
            runtime=total_runtime
        )
        _, new_labels, _ = predict_scores_multisphere(temp_sol, xtrain)

        # Si una reasignación deja algún clúster vacío, no la aceptamos
        if len(np.unique(new_labels)) < instance.k:
            print("ClusterSVDD stopped: reasigning collapses a cluster.")
            final_components = components
            final_labels = labels
            final_obj = total_obj
            final_runtime = total_runtime
            break

        # Si ya no cambian las asignaciones, hemos convergido
        if np.array_equal(labels, new_labels):
            final_components = components
            final_labels = labels
            final_obj = total_obj
            final_runtime = total_runtime
            break

        labels = new_labels
        final_components = components
        final_labels = labels
        final_obj = total_obj
        final_runtime = total_runtime

    # Si no se guardó nada por algún caso raro, entrenamos con las últimas labels válidas
    if final_components is None:
        final_components, final_obj, final_runtime = fit_svdds_by_labels(instance, final_labels)

    sol = MultiSphereSolution(
        components=final_components,
        labels_train=final_labels,
        data=instance.data,
        ntrain=instance.ntrain,
        nval=instance.nval,
        ntest=instance.ntest,
        y=instance.y,
        Nu=instance.Nu,
        kernel=instance.kernel,
        sigma=instance.sigma,
        degree=instance.degree,
        obj=final_obj,
        runtime=final_runtime
    )
    return sol