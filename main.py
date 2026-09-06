from models import *
from utilities import evaluate, metrics, gen_new_data, plot_cross_val

import os
import sys
import argparse
import numpy as np


if __name__ == '__main__':
    """
    Runs the complete experiment from the provided arguments.
    
    Params:
    -------
        args (argparse.Namespace or types.SimpleNamespace): Object containing data, model, evaluation, and plotting options.
    
    Returns:
    --------
        None: This function generates data, runs evaluations, computes metrics, and optionally plots cross-validation results.
    """

    # ============================================================
    # Parse command line arguments
    # ============================================================

    parser = argparse.ArgumentParser(description='SVDD / OC-SVM / ClusterSVDD Model Parameters')

    # Data generation
    parser.add_argument('--new_data', action='store_true', help='Generate new data')

    parser.add_argument('--anom_frac',nargs='+',type=float,default=[0.1],help='Anomaly fraction in data generation')
    parser.add_argument('--num_train',nargs='+',type=int,help='List of training set sizes')
    parser.add_argument('--num_val',nargs='+',type=int,help='List of validation set sizes')
    parser.add_argument('--num_test',nargs='+',type=int,help='List of test set sizes')
    parser.add_argument('--data_shape',type=str,default='spherical',help='Shape of generated data')
    parser.add_argument('--data_k',type=int,default=1,help='Number of clusters used to generate the data')
    parser.add_argument('--random_state',type=int,default=0,help='Random seed')

    parser.add_argument('--method',type=str,default='svdd',choices=['svdd', 'ocsvm', 'mssvdd', 'clustersvdd'],help='Model to evaluate')
    parser.add_argument('--k',type=int,default=1,help='Number of clusters/spheres used by the model')
    parser.add_argument('--max_iter',type=int,default=20,help='Maximum number of iterations')
    parser.add_argument('--Nus',nargs='+',type=float,help='List of Nu values')
    parser.add_argument('--sigmas',nargs='+',type=float,help='List of sigma values')
    parser.add_argument('--use_kernel',action='store_true',help='Use Gaussian RBF kernel')
    parser.add_argument('--use_density',action='store_true',default=False,help='Use density weighting')

    parser.add_argument('--reps',nargs='+',type=int,default=[1, 2, 3, 4, 5],help='List of repetitions')
    parser.add_argument('--do_evaluation',action='store_true',help='Evaluate models')
    parser.add_argument('--do_metrics',action='store_true',help='Calculate metrics')
    parser.add_argument('--do_cross_val',action='store_true',help='Perform cross validation')
    parser.add_argument('--show_plot_detail',action='store_true',help='Show detailed plots')
    parser.add_argument('--save_plot_detail',action='store_true',help='Save detailed plots')

    parser.add_argument('--dir_data',type=str,default='Data',help='Directory for input data files')
    parser.add_argument('--dir_sufix_out',type=str,default='',help='Suffix for output directories')


    args = parser.parse_args()


    # ============================================================
    # Dataset sizes
    # ============================================================

    if args.num_train is None:
        sys.exit("--num_train is required")

    num_train = args.num_train

    if args.num_val is None:
        num_val = [int(t * 2 / 3) for t in num_train]
    else:
        num_val = args.num_val

    if args.num_test is None:
        num_test = [int(t * 5 / 3) for t in num_train]
    else:
        num_test = args.num_test


    # ============================================================
    # Store arguments
    # ============================================================

    new_data = args.new_data

    use_kernel = args.use_kernel
    use_density = args.use_density

    do_evaluation = args.do_evaluation
    do_metrics = args.do_metrics
    do_cross_val = args.do_cross_val

    show_plot_detail = args.show_plot_detail
    save_plot_detail = args.save_plot_detail

    Nus = args.Nus
    sigmas = args.sigmas
    reps = args.reps
    anom_frac = args.anom_frac

    method = args.method.lower()

    k = args.k
    data_k = args.data_k

    max_iter = args.max_iter
    random_state = args.random_state

    data_shape = args.data_shape.lower()


    # ============================================================
    # Kernel / primal-dual configuration
    # ============================================================

    if use_kernel:
        DorP = 'Dual'
        if sigmas is None:
            sys.exit("--sigmas is required when --use_kernel is specified")
    else:
        if data_shape=="banana":
            ''
            # sys.exit("Does not make sense to test spherical model with non spherical data")
        sigmas = [-1.0]
        DorP = 'Prim'
        
    if use_density:
        density_tag="_dw"
    else:
        density_tag=""

    # Create directories and store paths
    dir_data = f'{args.dir_data}'
    dir_sufix_out = f'{args.dir_sufix_out}'
        
    data_DorP = f'Data_{DorP}_{dir_sufix_out}'
    output_DorP = f'Salidas_{DorP}_{dir_sufix_out}'
    figures_DorP = f'Figures_{DorP}_{dir_sufix_out}'
    
    if save_plot_detail:
        os.makedirs(figures_DorP, exist_ok=True)

    for f in range(len(anom_frac)):
        for t in range(len(num_train)):
        
            if new_data:
                os.makedirs(dir_data, exist_ok=True)
                gen_new_data(dir_data, num_train[t], num_val[t], num_test[t], anom_frac[f], reps, data_k=data_k, random_state=random_state, data_shape=data_shape)
                print('Data generated with ntrain =', num_train[t], ', nval =', num_val[t], ', ntest =', num_test[t], ', anom_frac =', anom_frac[f], ', reps =', reps)
        
            if do_evaluation:
                os.makedirs(data_DorP, exist_ok=True)
                os.makedirs(output_DorP, exist_ok=True)
                os.makedirs(dir_data, exist_ok=True)
                evaluate(Nus, sigmas, reps, num_train[t], num_test[t], num_val[t], anom_frac[f], use_kernel, dir_data, data_DorP, output_DorP,method,data_shape,data_k,k,max_iter,random_state,use_density)
    
            if do_metrics:
                os.makedirs(output_DorP, exist_ok=True)
                os.makedirs(data_DorP, exist_ok=True)
                res_filename = f'{output_DorP}/res_roc_{DorP}_{method}_{density_tag}_{data_shape}_datak{data_k}_k{k}_ntr{num_train[t]}_nval{num_val[t]}_nte{num_test[t]}_anom{anom_frac[f]}_reps{len(reps)}_nus{len(Nus)}_sigmas{len(sigmas)}.npz'
                metrics(res_filename, Nus, sigmas, reps, num_train[t], num_test[t], num_val[t], anom_frac[f], use_kernel, save_plot_detail, show_plot_detail, dir_data, data_DorP, output_DorP, figures_DorP, method=method, data_shape=data_shape,data_k=data_k,k=k,density_weight=use_density) 
            
            if do_cross_val: # Solo puede llamarse si previamente se ha llamado a do_metrics_ms con los mismos parametros.
                os.makedirs(output_DorP, exist_ok=True)
                res_filename = f'{output_DorP}/res_roc_{DorP}_{method}_{density_tag}_{data_shape}_datak{data_k}_k{k}_ntr{num_train[t]}_nval{num_val[t]}_nte{num_test[t]}_anom{anom_frac[f]}_reps{len(reps)}_nus{len(Nus)}_sigmas{len(sigmas)}.npz'
                plot_cross_val(res_filename) # AUC-ROC

    print('DONE :)')
