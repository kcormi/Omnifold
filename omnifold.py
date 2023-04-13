import argparse
import gc
import os
import sys
import time
#from mytrain import *
import energyflow as ef
import numpy as np

import pandas as pd

# DCTR, reweights positive distribution to negative distribution
# X: features
# Y: categorical labels
# model: model with fit/predict
# fitargs: model fit arguments
def train_model( model, df, features, labels, weights, train_data=None, val_data=None, **fit_args):

    #take a random permutation so that the training order is random
    permutation = np.random.permutation(df.index)
    invperm = np.argsort(permutation)
    df_perm = df.reindex( permutation )
    df_perm = df_perm.dropna()

    val_tuple = None
    if val_data is not None: 
        val_sel = df_perm.iloc[val_data]
        val_tuple=( val_sel[features], np.vstack(val_sel[labels].to_numpy()), val_sel[weights])    
        

    train_df = df[train_data] if train_data is not None else df
    model.fit( df_perm[features], np.vstack(df_perm[labels].to_numpy()), sample_weight=df_perm[weights], **fit_args, validation_data=val_tuple )
    return model
        

def reweight_df(df, features, labels, weights, model, fitargs, train_data=None, val_data=None, apply_data=None):

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
        model_ensemlbe = train_model( model_ensemble, df, features, labels, weights, train_data=train_data, val_data=val_data )
        model_ensemble.save_weights(filepath_ensemble)
        apply_df = df[apply_data] if apply_data is not None else df
        preds = model_ensemble.predict( apply_df[features], batch_size=fitargs.get('batch_size', 500))[:,1]
        preds_ensemble.append( preds )

    preds_mean = np.mean(np.array(preds_ensemble),axis=0)
    w = apply_df[weights]
    w *= np.clip(preds_mean/(1 - preds_mean + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w


def reweight(X, Y, w, model, fitargs, val_data=None, apply_data=None, train_idcs=None, val_idcs=None, apply_idcs=None):

    from_idcs = True
    if (val_data is not None) or (apply_data is not None):
        assert (train_idcs is None) and (val_idcs is None) and (apply_idcs is None)
        from_idcs = False
        
    if from_idcs:
        if val_idcs is not None:
            val_data = (X[val_idcs], Y[val_idcs], w[val_idcs] )
        if apply_idcs is not None:
            apply_data = (X[apply_idcs], w[apply_idcs])
        else:
            apply_data = (X, w )
        if train_idcs is not None:
            X, Y, w = X[train_idcs], Y[train_idcs], w[train_idcs]

    else:
        if apply_data is None:
            if val_data is not None:
                apply_features = np.concatenate([X, val_data[0]])
                apply_weights = np.concatenate([w, val_data[2]])
            else:
                apply_features = X
                apply_weights = w
            apply_data = [ apply_features, apply_weights ]

    val_dict = {'validation_data': val_data} if val_data is not None else {}
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
          #if len(filepath)==ensemble:
          #  filepath_ensemble=filepath[i_ensemble]
          #else:
          #  filepath_ensemble=None
        else:
          model_ensemble=model
          #filepath_ensemble=filepath
        filepath_ensemble = model_ensemble.filepath
        model_ensemble.fit(X, Y, sample_weight=w, **fitargs_tf, **val_dict)
        model_ensemble.save_weights(filepath_ensemble)
        preds = model_ensemble.predict( apply_data[0], batch_size=fitargs.get('batch_size', 500))[:,1]
        preds_ensemble.append( preds )

    preds_mean=np.mean(np.array(preds_ensemble),axis=0)
    w = apply_data[1]
    w *= np.clip(preds_mean/(1 - preds_mean + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w


def get_permutation( winit, wdata, val, step2=False):
    if step2:
        len_ = len(winit)
        nval = int(val*len_)
        baseperm0 = np.random.permutation(len_)
        baseperm1 = baseperm0 + len_
        
        # training examples are at beginning, val at end
        # concatenate into single train and val perms (shuffle each)
        trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
        valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
        np.random.shuffle(trainperm)
        perm = np.concatenate([trainperm, valperm])
    else:
        len_ = len(winit) + len(wdata)
        nval = int(val*len_)
        perm = np.random.permutation( len_ )

    invperm = np.argsort(perm)
    return perm, invperm, nval


def get_model_fpath( base_fp, ensemble_idx, iter_idx ):
    return (base_fp +f'_ensemble{ensemble_idx}').format(iter_idx) + '_Epoch{epoch}'

def get_model_ensemble_iter( base_model, ensemble_idx, iter_idx):
    model_args = { k: v for k, v in base_model[1].items() }
    base_model_fp = model_args['filepath']
    new_model_fp = get_model_fpath( base_model_fp, ensemble_idx, iter_idx )
    model_args.update( {'filepath' : new_model_fp } )
    new_model = base_model[0](**model_args)
    return new_model


# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold_df( df, features_det, features_gen, labels, label_MC, weights, det_model, mc_model, fitargs, 
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False,ensemble=1):

    # initialize the truth weights to the prior

    do_step2 = True
    # iterate the procedure
    for i in range(it):
        list_model_det=[]
        list_model_mc=[]
        # det filepaths properly
        for i_ensemble in range(ensemble):
            this_det_model = get_model_ensemble_iter( det_model, i_ensemble, i )
            list_model_det.append( this_det_model )

            if do_step2:
                this_mc_model = get_model_ensemble_iter( mc_model, i_ensemble, i )
                list_model_mc.append( this_mc_model )

        # load weights if not model 0
        if i > 0:
            for i_ensemble in range(ensemble):
                #list_model_det[i_ensemble].load_weights(list_model_det_fp[i_ensemble].format(i-1))
                last_iter_det_model = get_model_fpath( det_model[1]['filepath'], i_ensemble, i-1 )
                list_model_det[i_ensemble].load_weights( last_iter_det_model )
                if do_step2:
                    #list_model_mc[i_ensemble].load_weights(list_model_mc_fp[i_ensemble].format(i-1))
                    last_iter_mc_model = get_model_fpath( mc_model[1]['filepath'], i_ensemble, i-1 )
                    list_model_mc[i_ensemble].load_weights( last_iter_mc_model )

        # step 1: reweight sim to look like data
        ws = []
        ws.append(df[ df[labels] == label_MC ][weights].tolist())
        #w_train, w_val = w[det_idcs_train], w[det_idcs_val]
        df['numeric_labels'] = list( ef.utils.to_categorical((df[labels] == label_MC).astype( int )) )
        #df['numeric_labels'] = df['numeric_labels'].astype(np.ndarray)
        print(df['numeric_labels'])
        rw = reweight_df(df, features_det, 'numeric_labels', weights, list_model_det, fitargs )#, train_data=det_idcs_train, val_idcs=det_idcs_val )
        ws.append(rw[ df[labels] == label_MC ].tolist())
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        if do_step2:
            # step 2: reweight the prior to the learned weighting
            print(ws)
            w = np.concatenate((ws[-1], ws[trw_ind]))
            #w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
            rw = reweight_df(df, features_gen, 'numeric_labels', weights, list_model_mc, fitargs)
            ws.append(rw[len(ws[-1]):].tolist())
            #ws.append(rw[len(ws[-1]):]*np.sum(ws[trw_ind])/np.sum(rw[len(ws[-1]):]))
            # save the weights if specified

        if weights_filename is not None:
            np.save(weights_filename, ws)
            print("save weight ",weights_filename) 
    return ws

# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold(X_gen_i, Y_gen_i, X_det_i, Y_det_i, wdata, winit, det_model, mc_model, fitargs, 
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False,ensemble=1):

    do_step2 = not ( (X_gen_i is None) or (Y_gen_i is None) )
    # get arrays (possibly globally)
    X_det = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i[:]
    Y_det = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i[:]
    if do_step2:
        X_gen = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i[:]
        Y_gen = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i[:]

    # initialize the truth weights to the prior
    ws = [winit]

    # get permutation for det
    perm_det, invperm_det, nval_det = get_permutation( winit, wdata, val )
    det_idcs_train, det_idcs_val = perm_det[:-nval_det], perm_det[-nval_det:]
    #X_det_train, X_det_val = X_det_arr[det_idcs_train], X_det_arr[det_idcs_val]
    #Y_det_train, Y_det_val = Y_det_arr[det_idcs_train], Y_det_arr[det_idcs_val]

    # remove X_det, Y_det
    if delete_global_arrays:
        if isinstance(X_det_i, str):
            del globals()[X_det_i]
        if isinstance(Y_det_i, str):
            del globals()[Y_det_i]

    if do_step2:
        perm_gen, invperm_gen, nval_gen = get_permutation( winit, wdata, val, step2=True )
        gen_idcs_train, gen_idcs_val = perm_gen[:-nval_gen], perm_gen[-nval_gen:]
        #X_gen_train, X_gen_val = X_gen_arr[gen_idcs_train], X_gen_arr[gen_idcs_val]
        #Y_gen_train, Y_gen_val = Y_gen_arr[gen_idcs_train], Y_gen_arr[gen_idcs_val]

    # remove X_gen, Y_gen
    if delete_global_arrays:
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]

    # iterate the procedure
    for i in range(it):
        list_model_det=[]
        list_model_mc=[]
        # det filepaths properly
        for i_ensemble in range(ensemble):
            this_det_model = get_model_ensemble_iter( det_model, i_ensemble, i )
            list_model_det.append( this_det_model )

            if do_step2:
                this_mc_model = get_model_ensemble_iter( mc_model, i_ensemble, i )
                list_model_mc.append( this_mc_model )

        # load weights if not model 0
        if i > 0:
            for i_ensemble in range(ensemble):
                #list_model_det[i_ensemble].load_weights(list_model_det_fp[i_ensemble].format(i-1))
                last_iter_det_model = get_model_fpath( det_model[1]['filepath'], i_ensemble, i-1 )
                list_model_det[i_ensemble].load_weights( last_iter_det_model )
                if do_step2:
                    #list_model_mc[i_ensemble].load_weights(list_model_mc_fp[i_ensemble].format(i-1))
                    last_iter_mc_model = get_model_fpath( mc_model[1]['filepath'], i_ensemble, i-1 )
                    list_model_mc[i_ensemble].load_weights( last_iter_mc_model )

        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        #w_train, w_val = w[det_idcs_train], w[det_idcs_val]
        rw = reweight(X_det, Y_det, w, list_model_det, 
                      fitargs, train_idcs=det_idcs_train, val_idcs=det_idcs_val )[invperm_det]
        ws.append(rw[len(wdata):])
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        if do_step2:
            # step 2: reweight the prior to the learned weighting
            w = np.concatenate((ws[-1], ws[trw_ind]))
            #w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
            rw = reweight(X_gen, Y_gen, w, list_model_mc, 
                          fitargs, train_idcs=gen_idcs_train, val_idcs=gen_idcs_val)[invperm_gen]
            ws.append(rw[len(ws[-1]):])
            #ws.append(rw[len(ws[-1]):]*np.sum(ws[trw_ind])/np.sum(rw[len(ws[-1]):]))
            # save the weights if specified

        if weights_filename is not None:
            np.save(weights_filename, ws)
        print("save weight ",weights_filename) 
    return ws

# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold_acceptance_efficiency(X_gen_i, Y_gen_i, X_det_i, Y_det_i,X_det_acc_i,Y_det_acc_i, wdata, winit, gen_passgen, gen_passreco, det_passgen, det_passreco,det_passgen_acc,det_passreco_acc, det_model, mc_model,mc_model_1b,det_model_2b, fitargs,
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False,ensemble=1):

    # get arrays (possibly globally)
    X_gen_arr = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i
    Y_gen_arr = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i
    X_det_arr = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i
    Y_det_arr = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i
    X_det_acc_arr = globals()[X_det_acc_i] if isinstance(X_det_acc_i, str) else X_det_acc_i
    Y_det_acc_arr = globals()[Y_det_acc_i] if isinstance(Y_det_acc_i, str) else Y_det_acc_i

    # initialize the truth weights to the prior
    ws = [winit]

    # get permutation for det
    perm_det = np.random.permutation(len(winit) + len(wdata))
    invperm_det = np.argsort(perm_det)
    nval_det = int(val*len(perm_det))

    det_mask_reco_gen = det_passreco*det_passgen
    det_mask_reco_nogen = det_passreco*(det_passgen==False)
    det_mask_gen_noreco = det_passgen*(det_passreco==False)

    X_det_train, X_det_val = X_det_arr[perm_det[:-nval_det]], X_det_arr[perm_det[-nval_det:]]
    Y_det_train, Y_det_val = Y_det_arr[perm_det[:-nval_det]], Y_det_arr[perm_det[-nval_det:]]

    det_mask_reco_train, det_mask_reco_val = det_passreco[perm_det[:-nval_det]],det_passreco[perm_det[-nval_det:]]
    det_mask_gen_train, det_mask_gen_val = det_passgen[perm_det[:-nval_det]],det_passgen[perm_det[-nval_det:]]
    det_mask_reco_gen_train, det_mask_reco_gen_val = det_mask_reco_gen[perm_det[:-nval_det]],det_mask_reco_gen[perm_det[-nval_det:]]
    det_mask_reco_nogen_train, det_mask_reco_nogen_val = det_mask_reco_nogen[perm_det[:-nval_det]],det_mask_reco_nogen[perm_det[-nval_det:]]
    det_mask_gen_noreco_train, det_mask_gen_noreco_val = det_mask_gen_noreco[perm_det[:-nval_det]],det_mask_gen_noreco[perm_det[-nval_det:]]


    # remove X_det, Y_det
    if delete_global_arrays:
        del X_det_arr, Y_det_arr
        if isinstance(X_det_i, str):
            del globals()[X_det_i]
        if isinstance(Y_det_i, str):
            del globals()[Y_det_i]

    # get an initial permutation for gen and duplicate (offset) it
    nval = int(val*len(winit))
    baseperm0 = np.random.permutation(len(winit))
    baseperm1 = baseperm0 + len(winit)

    # training examples are at beginning, val at end
    # concatenate into single train and val perms (shuffle each)
    trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
    valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
    np.random.shuffle(trainperm)
    np.random.shuffle(valperm)

    # get final permutation for gen (ensured that the same events end up in val)
    perm_gen = np.concatenate((trainperm, valperm))
    invperm_gen = np.argsort(perm_gen)
    nval_gen = int(val*len(perm_gen))
    X_gen_train, X_gen_val = X_gen_arr[perm_gen[:-nval_gen]], X_gen_arr[perm_gen[-nval_gen:]]
    Y_gen_train, Y_gen_val = Y_gen_arr[perm_gen[:-nval_gen]], Y_gen_arr[perm_gen[-nval_gen:]]

    gen_mask_reco_gen = gen_passreco*gen_passgen
    gen_mask_reco_nogen = gen_passreco*(gen_passgen==False)
    gen_mask_gen_noreco = gen_passgen*(gen_passreco==False)

    gen_mask_reco_train, gen_mask_reco_val = gen_passreco[perm_gen[:-nval_gen]], gen_passreco[perm_gen[-nval_gen:]]
    gen_mask_gen_train, gen_mask_gen_val = gen_passgen[perm_gen[:-nval_gen]], gen_passgen[perm_gen[-nval_gen:]]
    gen_mask_reco_gen_train, gen_mask_reco_gen_val = gen_mask_reco_gen[perm_gen[:-nval_gen]],gen_mask_reco_gen[perm_gen[-nval_gen:]]
    gen_mask_reco_nogen_train, gen_mask_reco_nogen_val = gen_mask_reco_nogen[perm_gen[:-nval_gen]], gen_mask_reco_nogen[perm_gen[-nval_gen:]]
    gen_mask_gen_noreco_train, gen_mask_gen_noreco_val = gen_mask_gen_noreco[perm_gen[:-nval_gen]], gen_mask_gen_noreco[perm_gen[-nval_gen:]]

    #print("X_gen_train",X_gen_train, "len",len(X_gen_train))
    #print("gen_mask_gen_noreco_train",gen_mask_gen_noreco_train,len(gen_mask_gen_noreco_train))
    #print("X_gen_train[gen_mask_gen_noreco_train]",X_gen_train[gen_mask_gen_noreco_train])
    #print("X_gen_val",X_gen_val,len(X_gen_val))
    #print("gen_mask_gen_noreco_val",gen_mask_gen_noreco_val,len(gen_mask_gen_noreco_val))
    #print("X_gen_val[gen_mask_gen_noreco_val]",X_gen_val[gen_mask_gen_noreco_val])
    #print(np.concatenate([X_gen_train[gen_mask_gen_noreco_train],X_gen_val[gen_mask_gen_noreco_val]],axis=0))


    # remove X_gen, Y_gen
    if delete_global_arrays:
        del X_gen_arr, Y_gen_arr
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]

    # get an initial permutation for gen and duplicate (offset) it
    baseperm0 = np.random.permutation(len(winit))
    baseperm1 = baseperm0 + len(winit)

    # training examples are at beginning, val at end
    # concatenate into single train and val perms (shuffle each)
    trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
    valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
    np.random.shuffle(trainperm)
    np.random.shuffle(valperm)

    # get final permutation for gen (ensured that the same events end up in val)
    perm_det_acc = np.concatenate((trainperm, valperm))
    invperm_det_acc = np.argsort(perm_det_acc)
    nval_det_acc = int(val*len(perm_det_acc))
    X_det_acc_train, X_det_acc_val = X_det_acc_arr[perm_det_acc[:-nval_det_acc]], X_det_acc_arr[perm_det_acc[-nval_det_acc:]]
    Y_det_acc_train, Y_det_acc_val = Y_det_acc_arr[perm_det_acc[:-nval_det_acc]], Y_det_acc_arr[perm_det_acc[-nval_det_acc:]]

    det_acc_mask_reco_gen = det_passreco_acc*det_passgen_acc
    det_acc_mask_reco_nogen = det_passreco_acc*(det_passgen_acc==False)
    det_acc_mask_gen_noreco = det_passgen_acc*(det_passreco_acc==False)

    det_acc_mask_reco_train, det_acc_mask_reco_val = det_passreco_acc[perm_det_acc[:-nval_det_acc]],det_passreco_acc[perm_det_acc[-nval_det_acc:]]
    det_acc_mask_gen_train, det_acc_mask_gen_val = det_passgen_acc[perm_det_acc[:-nval_det_acc]],det_passgen_acc[perm_det_acc[-nval_det_acc:]]
    det_acc_mask_reco_gen_train, det_acc_mask_reco_gen_val = det_acc_mask_reco_gen[perm_det_acc[:-nval_det_acc]],det_acc_mask_reco_gen[perm_det_acc[-nval_det_acc:]]
    det_acc_mask_reco_nogen_train, det_acc_mask_reco_nogen_val = det_acc_mask_reco_nogen[perm_det_acc[:-nval_det_acc]],det_acc_mask_reco_nogen[perm_det_acc[-nval_det_acc:]]
    det_acc_mask_gen_noreco_train, det_acc_mask_gen_noreco_val = det_acc_mask_gen_noreco[perm_det_acc[:-nval_det_acc]],det_acc_mask_gen_noreco[perm_det_acc[-nval_det_acc:]]

    # remove X_det, Y_det
    if delete_global_arrays:
        del X_det_acc_arr, Y_det_acc_arr
        if isinstance(X_det_acc_i, str):
            del globals()[X_det_acc_i]
        if isinstance(Y_det_acc_i, str):
            del globals()[Y_det_acc_i]


    # store model filepaths
    list_model_det_fp=[]
    list_model_mc_fp=[]
    list_model_det_fp_2b=[]
    list_model_mc_fp_1b=[]
    model_det_fp, model_mc_fp,model_det_fp_2b, model_mc_fp_1b = det_model[1].get('filepath', None), mc_model[1].get('filepath', None),det_model_2b[1].get('filepath', None), mc_model_1b[1].get('filepath', None)
    for i_ensemble in range(ensemble):
      if model_det_fp is not None:
        list_model_det_fp.append(model_det_fp+"_ensemble"+str(i_ensemble))
      if model_mc_fp is not None:
        list_model_mc_fp.append(model_mc_fp+"_ensemble"+str(i_ensemble))
      if model_det_fp_2b is not None:
        list_model_det_fp_2b.append(model_det_fp_2b+"_ensemble"+str(i_ensemble))
      if model_mc_fp_1b is not None:
        list_model_mc_fp_1b.append(model_mc_fp_1b+"_ensemble"+str(i_ensemble))

    # iterate the procedure
    for i in range(it):
        list_model_det_fp_i=[]
        list_model_mc_fp_i=[]
        list_model_det_fp_2b_i=[]
        list_model_mc_fp_1b_i=[]
        list_model_det=[]
        list_model_mc=[]
        list_model_det_2b=[]
        list_model_mc_1b=[]
        for i_ensemble in range(ensemble):
            # det filepaths properly
            if model_det_fp is not None:
                list_model_det_fp_i.append(list_model_det_fp[i_ensemble].format(i))
                det_model[1]['filepath'] = list_model_det_fp_i[i_ensemble] + '_Epoch-{epoch}'
            if model_mc_fp is not None:
                list_model_mc_fp_i.append(list_model_mc_fp[i_ensemble].format(i))
                mc_model[1]['filepath'] = list_model_mc_fp_i[i_ensemble] + '_Epoch-{epoch}'
            if model_det_fp_2b is not None:
                list_model_det_fp_2b_i.append(list_model_det_fp_2b[i_ensemble].format(i))
                det_model_2b[1]['filepath'] = list_model_det_fp_2b_i[i_ensemble] + '_Epoch-{epoch}'
            if model_mc_fp_1b is not None:
                list_model_mc_fp_1b_i.append(list_model_mc_fp_1b[i_ensemble].format(i))
                mc_model_1b[1]['filepath'] = list_model_mc_fp_1b_i[i_ensemble] + '_Epoch-{epoch}'
            # define models
            list_model_det.append(det_model[0](**det_model[1]))
            list_model_mc.append(mc_model[0](**mc_model[1]))
            list_model_det_2b.append(det_model_2b[0](**det_model_2b[1]))
            list_model_mc_1b.append(mc_model_1b[0](**mc_model_1b[1]))

        # load weights if not model 0
        if i > 0:
            for i_ensemble in range(ensemble):
                list_model_det[i_ensemble].load_weights(list_model_det_fp[i_ensemble].format(i-1))
                list_model_mc[i_ensemble].load_weights(list_model_mc_fp[i_ensemble].format(i-1))
                list_model_det_2b[i_ensemble].load_weights(list_model_det_fp_2b[i_ensemble].format(i-1))
                list_model_mc_1b[i_ensemble].load_weights(list_model_mc_fp_1b[i_ensemble].format(i-1))

        # step 1: reweight sim to look like data
        print("Step 1: reweight at det-level")
        w = np.concatenate((wdata, ws[-1]))
        w_train, w_val = w[perm_det[:-nval_det]], w[perm_det[-nval_det:]]

        rw_perm_det = np.ones(len(w))
        rw_perm_det[det_passreco[perm_det]] = reweight(X_det_train[det_mask_reco_train], Y_det_train[det_mask_reco_train],
                                              w_train[det_mask_reco_train], list_model_det, 
                                              fitargs, val_data=(X_det_val[det_mask_reco_val], Y_det_val[det_mask_reco_val], w_val[det_mask_reco_val]))
        rw = rw_perm_det[invperm_det]
        rw_step1_tmp=rw[len(wdata):]
        ws.append(rw_step1_tmp)
        #print("weight step1 tmp",rw_step1_tmp)
        print("Step 1b: reweight the not-reconstructed events at gen-level")
        w = np.concatenate((rw_step1_tmp, ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw_perm_gen = np.ones(len(w))
        print("length rw_perm_gen",len(rw_perm_gen))
        #print("position of gen_mask_gen_noreco==True",np.argwhere(gen_mask_gen_noreco==True))
        #print("position of gen_mask_gen_noreco[perm_gen]==True",np.argwhere(gen_mask_gen_noreco[perm_gen]==True))
        rw_perm_gen[gen_mask_gen_noreco[perm_gen]] = reweight(X_gen_train[gen_mask_reco_gen_train], Y_gen_train[gen_mask_reco_gen_train],
                        w_train[gen_mask_reco_gen_train],list_model_mc_1b, fitargs,
                        val_data=(X_gen_val[gen_mask_reco_gen_val], Y_gen_val[gen_mask_reco_gen_val], w_val[gen_mask_reco_gen_val]),
                        apply_data=(np.concatenate([X_gen_train[gen_mask_gen_noreco_train],X_gen_val[gen_mask_gen_noreco_val]],axis=0),
                        np.concatenate([w_train[gen_mask_gen_noreco_train],w_val[gen_mask_gen_noreco_val]])))
        #print("rw_perm_gen[gen_mask_gen_noreco[perm_gen]]",rw_perm_gen[gen_mask_gen_noreco[perm_gen]])
        #print("gen_mask_gen_noreco[perm_gen]",gen_mask_gen_noreco[perm_gen])
        rw = rw_perm_gen[invperm_gen]
        #print("position of rw_perm_gen[invperm_gen] not 1.",np.argwhere(rw_perm_gen[invperm_gen]!=1.))
        #print("rw_perm_gen",rw_perm_gen)
        #print("rw_perm_gen[invperm_gen]",rw_perm_gen[invperm_gen])
        #print("weight step 1b",rw[len(ws[-1]):])
        ws.append(rw[len(ws[-1]):]*rw_step1_tmp)
        #print("weight now:",ws[-1])
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        # step 2: reweight the prior to the learned weighting
        print("Step 2: reweight at gen-level")
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw_perm_gen = np.ones(len(w))
     
        rw_perm_gen[gen_passgen[perm_gen]] = reweight(X_gen_train[gen_mask_gen_train], Y_gen_train[gen_mask_gen_train], w_train[gen_mask_gen_train],
                      list_model_mc, 
                      fitargs, val_data=(X_gen_val[gen_mask_gen_val], Y_gen_val[gen_mask_gen_val], w_val[gen_mask_gen_val]))
        rw =  rw_perm_gen[invperm_gen]
        rw_step2_tmp = rw[len(ws[-1]):]
        ws.append(rw_step2_tmp)
        print("weight step2 tmp",rw_step2_tmp)
        print("Step 2b: reweight the events not passing the gen-level selection at reco-level")
        w = np.concatenate((rw_step2_tmp,ws[trw_ind]))
        w_train, w_val = w[perm_det_acc[:-nval_det_acc]], w[perm_det_acc[-nval_det_acc:]]
        rw_perm_det_acc = np.ones(len(w))
        rw_perm_det_acc[det_acc_mask_reco_nogen[perm_det_acc]] = reweight(X_det_acc_train[det_acc_mask_reco_gen_train],Y_det_acc_train[det_acc_mask_reco_gen_train],
                         w_train[det_acc_mask_reco_gen_train],list_model_det_2b,fitargs,
                         val_data=(X_det_acc_val[det_acc_mask_reco_gen_val],Y_det_acc_val[det_acc_mask_reco_gen_val],w_val[det_acc_mask_reco_gen_val]),
                         apply_data=(np.concatenate([X_det_acc_train[det_acc_mask_reco_nogen_train],X_det_acc_val[det_acc_mask_reco_nogen_val]],axis=0),
                                     np.concatenate([w_train[det_acc_mask_reco_nogen_train],w_val[det_acc_mask_reco_nogen_val]])))
        rw = rw_perm_det_acc[invperm_det_acc]
        print("weight step 2b",rw[len(ws[-1]):])
        ws.append(rw[len(ws[-1]):]*rw_step2_tmp)
        print("weight now:",ws[-1])
        #ws.append(rw[len(ws[-1]):]*np.sum(ws[trw_ind])/np.sum(rw[len(ws[-1]):]))
        # save the weights if specified
        if weights_filename is not None:
            np.save(weights_filename, ws)
        print("save weight ",weights_filename)
    return ws


def omnifold_sys(X_i, Y_i, wdata, winit, det_mc_model, fitargs,
             val=0.2,  weights_filename=None, trw_ind=0, delete_global_arrays=False,ensemble=1):

    # get arrays (possibly globally)
    X_arr = globals()[X_i] if isinstance(X_i, str) else X_i
    Y_arr = globals()[Y_i] if isinstance(Y_i, str) else Y_i

    # initialize the truth weights to the prior
    ws = [winit]

    # get permutation
    perm = np.random.permutation(len(winit) + len(wdata))
    invperm = np.argsort(perm)
    nval = int(val*len(perm))
    X_train, X_val = X_arr[perm[:-nval]], X_arr[perm[-nval:]]
    Y_train, Y_val = Y_arr[perm[:-nval]], Y_arr[perm[-nval:]]

    # remove X, Y
    if delete_global_arrays:
        del X_arr, Y_arr
        if isinstance(X_i, str):
            del globals()[X_i]
        if isinstance(Y_i, str):
            del globals()[Y_i]

    # store model filepaths
    list_model_det_mc_fp=[]
    model_det_mc_fp = det_mc_model[1].get('filepath', None)
    for i_ensemble in range(ensemble):
        list_model_det_mc_fp.append(model_det_mc_fp+"_ensemble"+str(i_ensemble))

    # det filepaths properly
    list_model_det_mc_fp_i=[]
    list_model_det_mc=[]
    for i_ensemble in range(ensemble):
        if model_det_mc_fp is not None:
            list_model_det_mc_fp_i.append(list_model_det_mc_fp[i_ensemble].format(0))
            det_mc_model[1]['filepath'] = list_model_det_mc_fp_i[i_ensemble] + '_Epoch-{epoch}'
        list_model_det_mc.append(det_mc_model[0](**det_mc_model[1]))
    # define models
    w = np.concatenate((wdata, ws[-1]))
    w_train, w_val = w[perm[:-nval]], w[perm[-nval:]]
    rw = reweight(X_train, Y_train, w_train, list_model_det_mc,
                  fitargs, val_data=(X_val, Y_val, w_val))[invperm]
    ws.append(rw[len(wdata):])
    if weights_filename is not None:
        np.save(weights_filename, ws)
    print("save weight ",weights_filename)
    return ws





if __name__ == '__main__':
    main(sys.argv[1:])






















