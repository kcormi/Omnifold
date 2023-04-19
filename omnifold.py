import argparse
import gc
import os
import sys
import time
#from mytrain import *
import energyflow as ef
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# DCTR, reweights positive distribution to negative distribution
# X: features
# Y: categorical labels
# model: model with fit/predict
# fitargs: model fit arguments
def split_training_validation_selections( df, val_frac ):
    n = int(len(df) * val_frac)
    training =  df.index.to_numpy()[:-n]
    val = df.index.to_numpy()[-n:]
    return training, val

def train_model( model, df, features, labels, weights, train_idx=None, val_idx=None, val_frac=None, perm=None, **fit_args):

    #take a random permutation so that the training order is random
    train = df.loc[ train_idx ]
    val = df.loc[ val_idx ]

    perm_train = np.random.permutation(train.index)
    train = train.reindex( perm_train)

    perm_val = np.random.permutation(val.index)
    val = val.reindex( perm_val )


    if val_frac is not None:
        train_idx, val_idx = split_training_validation_selections( df_perm, val_frac)

    val_tuple = None
    if val_idx is not None:
        val_tuple=( val[features], np.vstack(val[labels].to_numpy()), val[weights])


    train_df = df.loc[train_idx] if train_idx is not None else df
    model.fit( train[features], np.vstack(train[labels].to_numpy()), sample_weight=train[weights], **fit_args, validation_data=val_tuple )
    return model


def reweight_df(df, features, labels, weights, model, fitargs, train_idx=None, val_idx=None, val_frac=None, apply_idx=None, apply_df=None):

    fitargs_tf={}
    for argkey in fitargs.keys():
      if "weight_clip" not in argkey:
        fitargs_tf[argkey]=fitargs[argkey]
    preds_ensemble=[]
    ensemble=1
    if isinstance(model,list):
      ensemble=len(model)
    for i_ensemble in range(ensemble):
        print("ensemble",i_ensemble)
        if isinstance(model,list):
          model_ensemble=model[i_ensemble]
        else:
          model_ensemble=model
        filepath_ensemble = model_ensemble.filepath
        #model_ensemble.fit(X, Y, sample_weight=w, **fitargs_tf, **val_dict)
        model_ensemble = train_model( model_ensemble, df, features, labels, weights, train_idx=train_idx, val_idx=val_idx, val_frac=val_frac )
        model_ensemble.save_weights(filepath_ensemble)
        print(f'saving models to {filepath_ensemble}')
        if apply_df is None:
            apply_df = df.loc[apply_idx] if apply_idx is not None else df
        preds = model_ensemble.predict( apply_df[features], batch_size=fitargs.get('batch_size', 500))[:,1]
        preds_ensemble.append( preds )

    preds_mean = np.mean(np.array(preds_ensemble),axis=0)
    w = apply_df[weights]
    w *= np.clip(preds_mean/(1 - preds_mean + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w


def get_model_fpath( base_fp, ensemble_idx, iter_idx ):
    return (base_fp +f'_ensemble{ensemble_idx}').format(iter_idx) + '_Epoch-{epoch}'

def get_model_ensemble_iter( base_model, ensemble_idx, iter_idx):
    model_args = { k: v for k, v in base_model[1].items() }
    base_model_fp = model_args['filepath']
    new_model_fp = get_model_fpath( base_model_fp, ensemble_idx, iter_idx )
    model_args.update( {'filepath' : new_model_fp } )
    new_model = base_model[0](**model_args)
    return new_model


def get_model_ensembles( n_models, base_model, iter_idx, load_weights=True):
        model_list=[]

        for i_model in range(n_models):
            this_model = get_model_ensemble_iter( base_model, i_model, iter_idx )
            model_list.append( this_model )

        if load_weights:
            for i_model in range(n_models):
                last_iter_model = get_model_fpath( base_model[1]['filepath'], i_model, iter_idx-1 )
                model_list[i_model].load_weights( last_iter_model )

        return model_list

def build_alt_weight_df( df, features, labels, weights, label_MC, new_weights, invert_labels=False ):
    #df = df[features + [labels] + [weights]]

    df_orig_mc = df[ df[labels] == label_MC ].copy()
    df_reweighted_mc =  df_orig_mc.copy()
    df_reweighted_mc[weights] = new_weights
    df_reweighted_mc[labels] = f'{label_MC}_reweighted'
    df_reweighted_mc = df_reweighted_mc.set_index( df_reweighted_mc.index.to_numpy() + 100*max(df_orig_mc.index) )

    df_gen = pd.concat( [ df_orig_mc, df_reweighted_mc ], ignore_index=False )
    label_vals = (df_gen[labels] != label_MC)
    if invert_labels:
        label_vals = ~label_vals

    df_gen['numeric_labels'] = list( ef.utils.to_categorical( label_vals.astype(int) ) )
    return df_gen


def do_check_plots( df, features, weights, labels, figname, nbins=20):
    if len(features) > 1:
        g = sns.PairGrid(df[features+[labels, weights]], hue=labels)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.histplot)
        plt.savefig(figname)
        plt.close()

    for feat in features:
        sns.histplot( data=df, x=feat, weights=weights, hue=labels, bins=nbins )
        plt.savefig(figname.replace('.pdf',f'_{feat}.pdf'))
        plt.close()

# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold_df( df, features_det, features_gen, labels, label_MC, weights, det_model, mc_model, fitargs, 
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False, ensemble=1, reco_cut=None, gen_cut=None,
             b1_model=None, b2_model=None, acc_eff=False, do_plots=False):



    df_is_MC = df[labels] == label_MC 
    # initialize the truth weights to the prior
    do_step2 = features_gen is not None

    #set these up so the same training/validation are used for every iteration
    df['step1_val'] = df.apply( lambda x: np.random.random() < val, axis=1)
    df['step2_val'] = df.apply( lambda x: np.random.random() < val, axis=1)

    step1_val_idx = lambda df: df[ df['step1_val'] ].index
    step2_val_idx = lambda df: df[ df['step2_val'] ].index
    step1_train_idx = lambda df: df[ ~df['step1_val'] ].index
    step2_train_idx = lambda df: df[ ~df['step2_val'] ].index

    if do_plots:
            plot_dir = './validation_plots3'
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            if features_gen:
                do_check_plots(df[ gen_cut ], features_gen, weights, labels, figname=os.path.join(plot_dir,'initial_gen.pdf'), nbins=30)
            do_check_plots(df[ reco_cut ], features_det, weights, labels, figname=os.path.join(plot_dir, 'initial_det.pdf'), nbins=30)
            #do_check_plots(df, features_det + features_gen, weights, labels, figname=os.path.join(plot_dir, 'initial_full.pdf'),  nbins=30)

    ws = []
    ws.append(df[ df_is_MC ][weights].tolist())
    # iterate the procedure
    for i in range(it):
        df.loc[ df_is_MC, weights] = ws[-1] #update the weights from last iteration

        # step 1: reweight sim to look like data
        list_model_det = get_model_ensembles( ensemble, det_model, i, load_weights = (i > 0) )
        df['numeric_labels'] = list( ef.utils.to_categorical((df[labels] != label_MC).astype( int )) )

        df_det = df[ reco_cut ]

        step1_idx = df_det[df_is_MC].index
        step1_weights = reweight_df(df_det, features_det, 'numeric_labels', weights, list_model_det, fitargs, train_idx=step1_train_idx(df_det), val_idx=step1_val_idx(df_det), apply_idx=step1_idx )#, train_idx=det_idcs_train, val_idcs=det_idcs_val )
        new_weights = df[ df_is_MC ][weights].copy()
        new_weights.loc[ step1_idx ] = step1_weights
        print(step1_idx)
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        if acc_eff and len(df) > len(df_det):
            list_model_1b = get_model_ensembles( ensemble, b1_model, i, load_weights = ( i > 0) )

            df_mc_1b = df_det[ df_is_MC ].copy() 
            df_mc_1b[weights] = new_weights
            df_mc_1b = df_mc_1b[ reco_cut ]
            df_mc_1b = build_alt_weight_df( df_mc_1b, features_gen, labels, weights, label_MC, np.ones(len(df_mc_1b)), invert_labels=True )
            df_det_ooa = df[ (~reco_cut(df)) & df_is_MC ]

            step1b_idx = df_det_ooa.index
            step1b_weights = reweight_df( df_mc_1b, features_gen, 'numeric_labels', weights, list_model_1b, fitargs, val_idx=step1_val_idx(df_mc_1b), train_idx=step1_train_idx(df_mc_1b), apply_df=df_det_ooa )
            new_weights.loc[ step1b_idx ] = step1b_weights

        ws.append( new_weights.tolist())
        print(new_weights)
        if do_plots:
            df.loc[ df_is_MC, weights] = new_weights
            do_check_plots(df[ reco_cut ], features_det, weights, labels, figname=os.path.join(plot_dir,f'iter{i}_step1.pdf'), nbins=30)

        if do_step2:
            # step 2: reweight the prior to the learned weighting
            list_model_mc =  get_model_ensembles( ensemble, mc_model, i,  load_weights = (i > 0 ) )

            df_gen = df[ df_is_MC].copy()
            df_gen[weights] = ws[trw_ind] # load "original" MC weights
            new_weights = df_gen[weights].copy()
            df_gen = build_alt_weight_df( df_gen, features_gen, labels, weights, label_MC, ws[-1] )
            dfgen_is_orig_MC =  df_gen[labels] == label_MC
            df_gen = df_gen[ gen_cut ]

            step2_idx = df_gen[ dfgen_is_orig_MC ].index
            step2_weights = reweight_df(df_gen, features_gen, 'numeric_labels', weights, list_model_mc, fitargs, val_idx=step2_val_idx(df_gen), train_idx=step2_train_idx(df_gen), apply_idx=step2_idx)
            new_weights.loc[ step2_idx ] = step2_weights
            #ws.append(rw[len(ws[-1]):]*np.sum(ws[trw_ind])/np.sum(rw[len(ws[-1]):]))
            # save the weights if specified

            if acc_eff and len(df_gen) < len(df[ df_is_MC]):
                list_model_2b = get_model_ensembles( ensemble, b2_model, i, load_weights = (i > 0) )

                df_det_2b = df_det[ df_is_MC ].copy()
                df_det_2b[weights] = new_weights
                df_det_2b = df_det_2b[ gen_cut ]
                df_det_2b = build_alt_weight_df( df_det_2b, features_det, labels, weights, label_MC, np.ones(len(df_det_2b)), invert_labels=True )
                df_gen_ooa = df[ (~gen_cut(df)) & df_is_MC ]

                step2b_idx = df_gen_ooa.index
                step2b_weights = reweight_df( df_det_2b, features_det, 'numeric_labels', weights, list_model_2b, fitargs, val_idx=step2_val_idx(df_det_2b), train_idx=step2_val_idx(df_det_2b), apply_df=df_gen_ooa )
                new_weights.loc[ step2b_idx ] = step2b_weights
                #print(step2b_idx)

            ws.append(new_weights.tolist())
            print(new_weights)
            if do_plots:
                df_gen.loc[ dfgen_is_orig_MC, weights] = step2_weights
                do_check_plots( df_gen[gen_cut], features_gen, weights, labels, figname=os.path.join(plot_dir,f'iter{i}_step2_gen.pdf'), nbins=30 )

                df.loc[ df_is_MC, weights] = new_weights
                do_check_plots(df[reco_cut], features_det, weights, labels, figname=os.path.join(plot_dir,f'iter{i}_step2_det.pdf'), nbins=30)

                #do_check_plots(df, features_det + features_gen, weights, labels, figname=os.path.join(plot_dir,f'iter{i}_step2_all.pdf'), nbins=30)

        if weights_filename is not None:
            np.save(weights_filename, ws)
            print(f"save weight {weights_filename}.npy") 

        if features_gen is None:
            break

    return ws

if __name__ == '__main__':
    main(sys.argv[1:])






















