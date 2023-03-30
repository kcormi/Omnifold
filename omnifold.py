import argparse
import gc
import os
import sys
import time
#from mytrain import *
import energyflow as ef
import numpy as np

from dataclasses import dataclass

# DCTR, reweights positive distribution to negative distribution
# X: features
# Y: categorical labels
# model: model with fit/predict
# fitargs: model fit arguments

@dataclass
class Sample:
    values: np.ndarray
    weights: np.ndarray
    to_keep: np.ndarray
    out_of_sample: np.ndarray

@dataclass
class DataSet:
    values:  np.ndarray
    labels:  np.ndarray
    weights: np.ndarray
    to_keep: np.ndarray
    out_of_sample: np.ndarray
    _paired_shuffle: bool = False
    

    def shuffle(self, arr ):
        pass

    def unshuffle(self, arr ):
        pass


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
            apply_data = (X[:], w[:] )
            print(apply_data[0].shape, apply_data[1].shape, 'hmmmm')
        if train_idcs is not None:
            in_sample_idcs = train_idcs
            if val_idcs is not None:
                in_sample_idcs = np.concatenate( [train_idcs, val_idcs] )
            X, Y, w = X[in_sample_idcs], Y[in_sample_idcs], w[in_sample_idcs]

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

    preds_mean = np.mean(np.array(preds_ensemble),axis=0)
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

def get_group_indices( perm, nval, basic_cut, separate_cut ):
    ''' return the indices for the train, validation and out-of-sample groups 
    based on the permutation, number of validation samples and mask of events passing cuts.'''

    train_grp = perm[:-nval]
    val_grp = perm[-nval:]

    if basic_cut is not None:
        basic_cut = np.array( basic_cut, dtype=bool)

        train_pass = basic_cut[train_grp]
        train_idcs = train_grp[train_pass]

        val_pass = basic_cut[val_grp]
        val_idcs = val_grp[val_pass]

    else:
        nasic_cut = np.ones( len(perm), dtype=bool)
        train_idcs = train_grp
        val_idcs = val_grp
        basic_cut = True #everything passes

    if separate_cut is not None:
        separate_cut = np.array(separate_cut, dtype=bool)
    else:
        separate_cut = np.ones( len(perm), dtype=bool)
    out_of_sample_idcs = perm[basic_cut & ~separate_cut]
    
    set_oos = set(out_of_sample_idcs)
    train_idcs = list( set(train_idcs).difference(set_oos) )
    val_idcs = list( set(val_idcs).difference(set_oos) )

    return train_idcs, val_idcs, out_of_sample_idcs
    

# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold(X_gen_i, Y_gen_i, X_det_i, Y_det_i, wdata, winit, det_model, mc_model, fitargs, 
             val=0.2, it=10, det_model_oos=None, gen_model_oos=None, weights_filename=None, trw_ind=0, delete_global_arrays=False,ensemble=1, det_pass_reco=None, det_pass_gen=None, gen_pass_reco=None, gen_pass_gen=None ):

    
    do_out_of_sample = det_model_oos is not None
    if det_pass_reco is None:
        det_pass_reco = np.ones(len(X_det_i), dtype=bool)
    if det_pass_gen is None:
        det_pass_gen = np.ones(len(X_det_i), dtype=bool)
    if X_gen_i is not None:
        if gen_pass_reco is None:
            gen_pass_reco = np.ones(len(X_gen_i), dtype=bool)
        if det_pass_gen is None:
            det_pass_gen = np.ones(len(X_gen_i), dtype=bool)


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
    det_i_train, det_i_val, det_i_fail_gen = get_group_indices( perm_det, nval_det, det_pass_reco, det_pass_gen)

    #perm_det[:-nval_det], perm_det[-nval_det:]
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
        gen_i_train, gen_i_val, gen_i_fail_reco = get_group_indices(perm_gen, nval_gen, gen_pass_gen, gen_pass_reco ) #perm_gen[:-nval_gen], perm_gen[-nval_gen:]
        print(gen_i_fail_reco)
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
        list_model_det_oos = []
        list_model_mc=[]
        list_model_gen_oos = []
        # det filepaths properly
        for i_ensemble in range(ensemble):
            this_det_model = get_model_ensemble_iter( det_model, i_ensemble, i )
            list_model_det.append( this_det_model )
            if do_out_of_sample:
                this_det_model_oos = get_model_ensemble_iter( det_model_oos, i_ensemble, i )
                list_model_det_oos.append( this_det_model_oos )

            if do_step2:
                this_mc_model = get_model_ensemble_iter( mc_model, i_ensemble, i )
                list_model_mc.append( this_mc_model )
                if do_out_of_sample:
                    this_gen_model_oos = get_model_ensemble_iter( gen_model_oos, i_ensemble, i )
                    list_model_gen_oos.append( this_gen_model_oos )
            

        # load weights if not model 0
        if i > 0:
            for i_ensemble in range(ensemble):
                #list_model_det[i_ensemble].load_weights(list_model_det_fp[i_ensemble].format(i-1))
                last_iter_det_model = get_model_fpath( det_model[1]['filepath'], i_ensemble, i-1 )
                list_model_det[i_ensemble].load_weights( last_iter_det_model )
                if do_out_of_sample:
                    list_iter_det_oos_model = get_model_fpath( det_model_oos[1]['filepath'], i_ensemble, i-1 )
                    list_model_det_oos[i_ensemble].load_weights( last_iter_det_oos_model )

                if do_step2:
                    #list_model_mc[i_ensemble].load_weights(list_model_mc_fp[i_ensemble].format(i-1))
                    last_iter_mc_model = get_model_fpath( mc_model[1]['filepath'], i_ensemble, i-1 )
                    list_model_mc[i_ensemble].load_weights( last_iter_mc_model )
                    if do_out_of_sample:
                        list_iter_gen_oos_model = get_model_fpath( gen_model_oos[1]['filepath'], i_ensemble, i-1 )
                        list_model_gen_oos[i_ensemble].load_weights( last_iter_gen_oos_model )

        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        #w_train, w_val = w[det_idcs_train], w[det_idcs_val]
        rw_1a = np.ones(len(w))
        in_sample = np.concatenate([det_i_train, det_i_val])
        rw_1a[in_sample[perm_det]] = reweight(X_det, Y_det, w, list_model_det, 
                      fitargs, train_idcs=det_i_train, val_idcs=det_i_val, apply_idcs=np.concatenate([det_i_train, det_i_val]) ) #[invperm_det]
        rw_1a_sim = rw_1a[len(wdata):]
        ws.append(rw_1a_sim)
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))
        if do_out_of_sample:
        
            rw_perm_gen = np.ones(len(w))
            rw_perm_gen[gen_i_fail_reco] = reweight(X_gen, Y_gen,
                            w, list_model_det_oos, fitargs,
                            train_idcs=gen_i_train, val_idcs=gen_i_val, apply_idcs=gen_i_fail_reco)
            rw_1b = rw_perm_gen[invperm_gen]
            rw_1b_sim = rw_1b[len(wdata):]
            
            ws.append(rw_1b_sim*rw_1a_sim)


        if do_step2:
            # step 2: reweight the prior to the learned weighting
            w = np.concatenate((ws[-1], ws[trw_ind]))
            rw_2a = np.ones(len(w))
            in_sample = gen_pass_gen & gen_pass_reco
            rw_2a[in_sample[perm_gen]] = reweight(X_gen, Y_gen, w, list_model_mc, 
                          fitargs, train_idcs=gen_i_train, val_idcs=gen_i_val, apply_idcs=np.concatenate([gen_i_train, gen_i_val])) #[invperm_gen]
            rw_2a_new = rw_2a[len(ws[-1]):]
            ws.append(rw_2a_new)
            #ws.append(rw[len(ws[-1]):]*np.sum(ws[trw_ind])/np.sum(rw[len(ws[-1]):]))
            # save the weights if specified

        if do_out_of_sample:

            rw_perm_det = np.ones(len(w))
            rw_perm_det[det_i_fail_gen] = reweight(X_det, Y_det,
                            w, list_model_gen_oos, fitargs,
                            train_idcs=det_i_train, val_idcs=det_i_val, apply_idcs=det_i_fail_gen)
            rw_2b = rw_perm_det[invperm_det]
            rw_2b_new = rw_2b[len(ws[-1]):]
            ws.append(rw_2b_new*rw_2a_new)

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






















