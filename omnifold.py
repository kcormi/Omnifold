import argparse
import gc
import os
import sys
import time
#from mytrain import *
import energyflow as ef
import numpy as np

# DCTR, reweights positive distribution to negative distribution
# X: features
# Y: categorical labels
# model: model with fit/predict
# fitargs: model fit arguments
def reweight(X, Y, w, model, filepath, fitargs, val_data=None):

    # permute the data, fit the model, and get preditions
    #perm = np.random.permutation(len(X))
    #model.fit(X[perm], Y[perm], sample_weight=w[perm], **fitargs)
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
          if len(filepath)==ensemble:
            filepath_ensemble=filepath[i_ensemble]
          else:
            filepath_ensemble=None
        else:
          model_ensemble=model
          filepath_ensemble=filepath
        model_ensemble.fit(X, Y, sample_weight=w, **fitargs_tf, **val_dict)
        model_ensemble.save_weights(filepath_ensemble)
        preds = model_ensemble.predict(X, batch_size=fitargs.get('batch_size', 500))[:,1]
        #preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,1]
    
        # concat_ensembleenate validation predictions into training predictions
        if val_data is not None:
            #preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,1]
            preds_val = model_ensemble.predict(val_data[0], batch_size=fitargs.get('batch_size', 500))[:,1]
            preds_ensemble.append(np.concatenate((preds, preds_val)))
        else:
            preds_ensemble.append(preds)
    preds_mean=np.mean(np.array(preds_ensemble),axis=0)
    if val_data is not None:
        w = np.concatenate((w, val_data[2]))
    w *= np.clip(preds_mean/(1 - preds_mean + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w


def reweight_acc_eff(X, Y, w, model,filepath, fitargs, val_data=None, apply_data=None):

    # permute the data, fit the model, and get preditions
    #perm = np.random.permutation(len(X))
    #model.fit(X[perm], Y[perm], sample_weight=w[perm], **fitargs)
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
        if isinstance(model,list):
          model_ensemble=model[i_ensemble]
          if len(filepath)==ensemble:
              filepath_ensemble=filepath[i_ensemble]
          else:
              filepath_ensemble=None
        else:
          model_ensemble=model
          filepath_ensemble=filepath
        model_ensemble.fit(X, Y, sample_weight=w, **fitargs_tf, **val_dict)
        model_ensemble.save_weights(filepath_ensemble)
        if apply_data is not None:
          preds = model_ensemble.predict(apply_data[0],batch_size=fitargs.get('batch_size', 500))[:,1]
          preds_ensemble.append(preds)
    preds_mean=np.mean(np.array(preds_ensemble),axis=0)
    w = apply_data[1]*np.clip(preds_mean/(1 - preds_mean + 10**-50), fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
    return w

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

    # get arrays (possibly globally)
    X_gen_arr = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i
    Y_gen_arr = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i
    X_det_arr = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i
    Y_det_arr = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i
    
    # initialize the truth weights to the prior
    ws = [winit]
    
    # get permutation for det
    perm_det = np.random.permutation(len(winit) + len(wdata))
    invperm_det = np.argsort(perm_det)
    nval_det = int(val*len(perm_det))
    X_det_train, X_det_val = X_det_arr[perm_det[:-nval_det]], X_det_arr[perm_det[-nval_det:]]
    Y_det_train, Y_det_val = Y_det_arr[perm_det[:-nval_det]], Y_det_arr[perm_det[-nval_det:]]

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

    # remove X_gen, Y_gen
    if delete_global_arrays:
        del X_gen_arr, Y_gen_arr
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]

    # store model filepaths
    list_model_det_fp=[]
    list_model_mc_fp=[]
    model_det_fp, model_mc_fp = det_model[1].get('filepath', None), mc_model[1].get('filepath', None)
    for i_ensemble in range(ensemble):
      if model_det_fp is not None:
        list_model_det_fp.append(model_det_fp+"_ensemble"+str(i_ensemble))
      if model_mc_fp is not None:
        list_model_mc_fp.append(model_mc_fp+"_ensemble"+str(i_ensemble))
    # iterate the procedure
    for i in range(it):
        list_model_det_fp_i=[]
        list_model_mc_fp_i=[]
        list_model_det=[]
        list_model_mc=[]
        # det filepaths properly
        for i_ensemble in range(ensemble):
            # det filepaths properly
            if model_det_fp is not None:
                list_model_det_fp_i.append(list_model_det_fp[i_ensemble].format(i))
                det_model[1]['filepath'] = list_model_det_fp_i[i_ensemble] + '_Epoch-{epoch}'
            if model_mc_fp is not None:
                list_model_mc_fp_i.append(list_model_mc_fp[i_ensemble].format(i))
                mc_model[1]['filepath'] = list_model_mc_fp_i[i_ensemble] + '_Epoch-{epoch}'
            list_model_det.append(det_model[0](**det_model[1]))
            list_model_mc.append(mc_model[0](**mc_model[1]))

        # load weights if not model 0
        if i > 0:
            for i_ensemble in range(ensemble):
                list_model_det[i_ensemble].load_weights(list_model_det_fp[i_ensemble].format(i-1))
                list_model_mc[i_ensemble].load_weights(list_model_mc_fp[i_ensemble].format(i-1))
        
        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        w_train, w_val = w[perm_det[:-nval_det]], w[perm_det[-nval_det:]]
        rw = reweight(X_det_train, Y_det_train, w_train, list_model_det, list_model_det_fp_i,
                      fitargs, val_data=(X_det_val, Y_det_val, w_val))[invperm_det]
        ws.append(rw[len(wdata):])
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        # step 2: reweight the prior to the learned weighting
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw = reweight(X_gen_train, Y_gen_train, w_train, list_model_mc, list_model_mc_fp_i,
                      fitargs, val_data=(X_gen_val, Y_gen_val, w_val))[invperm_gen]
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

    print("X_gen_train",X_gen_train, "len",len(X_gen_train))
    print("gen_mask_gen_noreco_train",gen_mask_gen_noreco_train,len(gen_mask_gen_noreco_train))
    print("X_gen_train[gen_mask_gen_noreco_train]",X_gen_train[gen_mask_gen_noreco_train])
    print("X_gen_val",X_gen_val,len(X_gen_val))
    print("gen_mask_gen_noreco_val",gen_mask_gen_noreco_val,len(gen_mask_gen_noreco_val))
    print("X_gen_val[gen_mask_gen_noreco_val]",X_gen_val[gen_mask_gen_noreco_val])
    print(np.concatenate([X_gen_train[gen_mask_gen_noreco_train],X_gen_val[gen_mask_gen_noreco_val]],axis=0))


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
                                              w_train[det_mask_reco_train], list_model_det, list_model_det_fp_i,
                                              fitargs, val_data=(X_det_val[det_mask_reco_val], Y_det_val[det_mask_reco_val], w_val[det_mask_reco_val]))
        rw = rw_perm_det[invperm_det]
        rw_step1_tmp=rw[len(wdata):]
        ws.append(rw_step1_tmp)
        print("weight step1 tmp",rw_step1_tmp)
        print("Step 1b: reweight the not-reconstructed events at gen-level")
        w = np.concatenate((rw_step1_tmp, ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw_perm_gen = np.ones(len(w))
        print("length rw_perm_gen",len(rw_perm_gen))
        print("position of gen_mask_gen_noreco==True",np.argwhere(gen_mask_gen_noreco==True))
        print("position of gen_mask_gen_noreco[perm_gen]==True",np.argwhere(gen_mask_gen_noreco[perm_gen]==True))
        rw_perm_gen[gen_mask_gen_noreco[perm_gen]] = reweight_acc_eff(X_gen_train[gen_mask_reco_gen_train], Y_gen_train[gen_mask_reco_gen_train],
                        w_train[gen_mask_reco_gen_train],list_model_mc_1b, list_model_mc_fp_1b_i,fitargs,
                        val_data=(X_gen_val[gen_mask_reco_gen_val], Y_gen_val[gen_mask_reco_gen_val], w_val[gen_mask_reco_gen_val]),
                        apply_data=(np.concatenate([X_gen_train[gen_mask_gen_noreco_train],X_gen_val[gen_mask_gen_noreco_val]],axis=0),
                        np.concatenate([w_train[gen_mask_gen_noreco_train],w_val[gen_mask_gen_noreco_val]])))
        print("rw_perm_gen[gen_mask_gen_noreco[perm_gen]]",rw_perm_gen[gen_mask_gen_noreco[perm_gen]])
        print("gen_mask_gen_noreco[perm_gen]",gen_mask_gen_noreco[perm_gen])
        rw = rw_perm_gen[invperm_gen]
        print("position of rw_perm_gen[invperm_gen] not 1.",np.argwhere(rw_perm_gen[invperm_gen]!=1.))
        print("rw_perm_gen",rw_perm_gen)
        print("rw_perm_gen[invperm_gen]",rw_perm_gen[invperm_gen])
        print("weight step 1b",rw[len(ws[-1]):])
        ws.append(rw[len(ws[-1]):]*rw_step1_tmp)
        print("weight now:",ws[-1])
        #what if I normalize?
        #ws.append(rw[len(wdata):]*np.sum(wdata)/np.sum(rw[len(wdata):]))

        # step 2: reweight the prior to the learned weighting
        print("Step 2: reweight at gen-level")
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw_perm_gen = np.ones(len(w))
     
        rw_perm_gen[gen_passgen[perm_gen]] = reweight(X_gen_train[gen_mask_gen_train], Y_gen_train[gen_mask_gen_train], w_train[gen_mask_gen_train],
                      list_model_mc, list_model_mc_fp_i,
                      fitargs, val_data=(X_gen_val[gen_mask_gen_val], Y_gen_val[gen_mask_gen_val], w_val[gen_mask_gen_val]))
        rw =  rw_perm_gen[invperm_gen]
        rw_step2_tmp = rw[len(ws[-1]):]
        ws.append(rw_step2_tmp)
        print("weight step2 tmp",rw_step2_tmp)
        print("Step 2b: reweight the events not passing the gen-level selection at reco-level")
        w = np.concatenate((rw_step2_tmp,ws[trw_ind]))
        w_train, w_val = w[perm_det_acc[:-nval_det_acc]], w[perm_det_acc[-nval_det_acc:]]
        rw_perm_det_acc = np.ones(len(w))
        rw_perm_det_acc[det_acc_mask_reco_nogen[perm_det_acc]] = reweight_acc_eff(X_det_acc_train[det_acc_mask_reco_gen_train],Y_det_acc_train[det_acc_mask_reco_gen_train],
                         w_train[det_acc_mask_reco_gen_train],list_model_det_2b,list_model_det_fp_2b_i,fitargs,
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
            list_model_det_mc_fp_i.append(list_model_det_mc_fp[i_ensemble].format(i))
            det_mc_model[1]['filepath'] = list_model_det_mc_fp_i[i_ensemble] + '_Epoch-{epoch}'
        list_model_det_mc.append(det_mc_model[0](**det_mc_model[1]))
    # define models
    w = np.concatenate((wdata, ws[-1]))
    w_train, w_val = w[perm[:-nval]], w[perm[-nval:]]
    rw = reweight(X_train, Y_train, w_train, list_model_det_mc, list_model_det_mc_fp_i,
                  fitargs, val_data=(X_val, Y_val, w_val))[invperm]
    ws.append(rw[len(wdata):])
    if weights_filename is not None:
        np.save(weights_filename, ws)
    print("save weight ",weights_filename)
    return ws





if __name__ == '__main__':
    main(sys.argv[1:])
