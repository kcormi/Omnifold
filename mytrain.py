import argparse
import gc
import os
import sys
import time
import omnifold
import energyflow as ef
import numpy as np

# default paths
MACHINES = {
    'multifold': {
        'data_path': '/work/jinw/omnifold/OmniFold/preselect',
        'results_path': '/work/kcormier/instantons/data'
    },
    'omnifold': {
        'data_path': '/work/jinw/omnifold/OmniFold/preselect',
        'results_path': '/work/kcormier/instans/data'
    },
    'unifold':{
        'data_path': '/work/jinw/omnifold/OmniFold/preselect',
        'results_path': '/work/kcormier/instantons/data'
    },
    'test':{
        'data_path' : '/work/kcormier/instantons/data/test',
        'results_path': '/work/kcormier/instantons/data/test',
    }
}

# default filenames
FILENAMES = {
    'Zerobias': ['flatTuple_ZeroBias_2018lowPU_new_hltv6_iso_nparticle_ext1-3.npz','flatTuple_ZeroBias_2018lowPU_new_hltv6_iso_nparticle_ext1-4.npz','flatTuple_ZeroBias_2018lowPU_new_hltv6_iso_nparticle_ext1-5.npz','flatTuple_ZeroBias_2018lowPU_new_hltv6_iso_nparticle_ext1-6.npz'],
    'Pythia8CP5': ['flatTuple_MB_trk_noPU_new.npz','flatTuple_MB_trk_noPU_new_CP5_B.npz','flatTuple_MB_trk_noPU_new_CP5_C.npz','flatTuple_MB_trk_noPU_new_CP5_D.npz'],
    #'Pythia8CP5': 'flatTuple_MB_trk_noPU_new.npz',
    #'Pythia8CP1': ['flatTuple_MB_trk_noPU_new_CP1_merge.npz','flatTuple_MB_trk_noPU_new_CP1_F.npz'],
    'Pythia8CP1': ['/pnfs/psi.ch/cms/trivcat/store/user/jinw/NPZ/flatTuple_MB_trk_noPU_new_CP1_merge_ABCDE.npz','/pnfs/psi.ch/cms/trivcat/store/user/jinw/NPZ/flatTuple_MB_trk_noPU_new_CP1_F.npz'],
    'Pythia8CP1': 'flatTuple_MB_trk_noPU_new_CP1_merge.npz',
    'Pythia8CP1_tuneES': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_add.npz',
    'Pythia8CP1_tuneNch': 'flatTuple_MB_trk_noPU_new_CP1_tuneNch.npz',
    'Pythia8CP5_trkdrop': 'flatTuple_MB_trk_noPU_new_trackdrop.npz',
    'Pythia8CP1_tuneES_trkdrop': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_trkdrop.npz',
    #'Pythia8EPOS': ['flatTuple_MB_trk_noPU_new_EPOS_merge.npz','flatTuple_MB_trk_noPU_new_EPOS_G.npz'],
    'Pythia8EPOS': ['/pnfs/psi.ch/cms/trivcat/store/user/jinw/NPZ/flatTuple_MB_trk_noPU_new_EPOS_merge_ABCDEG.npz'],
    #'Pythia8EPOS': 'flatTuple_MB_trk_noPU_new_EPOS_merge.npz',
    'Pythia8EPOS_trkdrop': ['flatTuple_MB_trk_noPU_new_EPOS_trkdrop.npz','flatTuple_MB_trk_noPU_new_EPOS_trkdrop_B.npz','flatTuple_MB_trk_noPU_new_EPOS_trkdrop_C.npz','flatTuple_MB_trk_noPU_new_EPOS_trkdrop_D.npz','flatTuple_MB_trk_noPU_new_EPOS_trkdrop_E.npz','flatTuple_MB_trk_noPU_new_EPOS_trkdrop_G.npz'],
    'Pythia8CP5_part1': 'flatTuple_MB_trk_noPU_new_1.npz',
    'Pythia8CP5_part2': 'flatTuple_MB_trk_noPU_new_2.npz',
    'Pythia8CP1_tuneES_part1': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_1.npz',
    'Pythia8CP1_tuneES_part2': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_2.npz',
    'Pythia8CP1_test': 'flatTuple_MB_trk_noPu_new_CP1_test.npz',
    'EPOS_test': 'flatTuple_MB_trk_noPu_new_EPOS_test.npz',
}


'''
SYSWEIGHTS = {
    'omnifold':{
      'MCCP1_tuneES_trkdrop':'',
      'MCCP1_tuneES_CP5':'',
      'MCCP1_tuneES_MCstat':'',
    },
    'multifold':{
      'MCCP1_tuneES_trkdrop':'',
      'MCCP1_tuneES_CP5':'',
      'MCCP1_tuneES_MCstat':'',
    },
    'unifold':{
      'MCCP1_tuneES_trkdrop':'',
      'MCCP1_tuneES_CP5':'',
      'MCCP1_tuneES_MCstat':'',
    },
}
'''

'''
SYSWEIGHTS = {
    'omnifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_omnifold_maxweight_MCCP1_tuneES_unfoldCP5',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_omnifold_maxweight_MCCP1_tuneES_unfoldtrkdrop',
      'MCCP1_tuneES_MCstat':'',
    },
    'multifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES_unfoldCP5',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES_unfoldtrkdrop',
      'MCCP1_tuneES_MCstat':'',
    },
    'unifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_unifold_maxweight_MCCP1_tuneES_unfoldCP5',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_unifold_maxweight_MCCP1_tuneES_unfoldtrkdrop',
      'MCCP1_tuneES_MCstat':'',
    },
}
'''

SYSWEIGHTS = {
    'omnifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_omnifold_maxweight_MCEPOS_unfoldCP5',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_omnifold_maxweight_MCEPOS_unfoldEPOStrkdrop',
      'MCCP1_tuneES_MCstat':'',
    },
    'multifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCEPOS_unfoldCP5_mass_ntrk',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCEPOS_unfoldtrkdrop_mass_ntrk',
      'MCCP1_tuneES_MCstat':'',
    },
    'unifold':{
      'MCCP1_tuneES_CP5':'/work/jinw/omnifold/OmniFold/results_unifold_maxweight_MCCP1_tuneES_unfoldCP5',
      'MCCP1_tuneES_trkdrop':'/work/jinw/omnifold/OmniFold/results_unifold_maxweight_MCCP1_tuneES_unfoldtrkdrop',
      'MCCP1_tuneES_MCstat':'',
    },
}

PREWEIGHTS = {
    'omnifold':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES_mass_ntrk/weights/ManyFold_DNN_Rep-0.npy',
    'multifold':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES_mass_ntrk/weights/ManyFold_DNN_Rep-0.npy',
    'unifold':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES_mass_ntrk/weights/ManyFold_DNN_Rep-0.npy'
}

DATAWEIGHT = {
    'gen_CP5_to_EPOS_multifold':'/work/jinw/omnifold/OmniFold/results_multifold_maxweight10_MCCP5_unfoldEPOS_1p3M_genweight/weights/ManyFold_DNN_Rep-0.npy'
}




def main(arg_list):

    # parse options, allow global access
    global args
    args = construct_parser(arg_list)

    # this must come before importing tensorflow to get the right GPU
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import energyflow.archs

    # handle names
    if args.unfolding == 'omnifold':
        name = args.name + 'OmniFold_{}_Rep-{}'
    elif args.unfolding == 'manyfold':
        name = args.name + 'ManyFold_DNN_Rep-{}'
    elif args.unfolding == 'unifold':
        name = args.name + 'UniFold_DNN_{}'
    if args.bootstrap:
      name+="_bs_"+str(args.bsseed)
    if args.MCbootstrap:
      name+="_MCbs_"+str(args.MCbsseed)

    # iteration loop
    for i in range(args.start_iter, args.max_iter):
        if args.unfolding == 'omnifold':
            args.name = name.format(args.omnifold_arch, i)
            if args.dosysweight:
              train_omnifold_fitsys(i)
            elif args.dogenreweight:
              train_omnifold_fitgen(i)
            else:
              train_omnifold(i)
        elif args.unfolding == 'manyfold':
            args.name = name.format(i)
            if args.dosysweight:
              train_manyfold(i, full_reweight=True)
            elif args.dogenreweight:
              train_manyfold(i, gen_only=True)
            elif args.eff_acc:
              train_manyfold_acceptance_efficiency(i)
            else:  
              train_manyfold(i)
        elif args.unfolding == 'unifold':
            args.name = name + '_Rep-{}'.format(i)
            train_unifold(i)

def construct_parser(args):

    parser = argparse.ArgumentParser(description='OmniFold unfolding.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #SYSWEIGHTS[] data selection
    parser.add_argument('--machine', '-m', choices=MACHINES.keys(), required=True)
    parser.add_argument('--dataset-mc', '-mc', choices=FILENAMES.keys(), default='Pythia8CP5')
    parser.add_argument('--dataset-data', '-data', choices=FILENAMES.keys(), default='Zerobias')

    # unfolding options
    parser.add_argument('--unfolding', '-u', choices=['omnifold', 'manyfold', 'unifold'], required=True)
    parser.add_argument('--step2-ind', type=int, choices=[0, -2], default=0)
    parser.add_argument('--unfolding-iterations', '-ui', type=int, default=8)
    parser.add_argument('--weight-clip-min', type=float, default=0.)
    parser.add_argument('--weight-clip-max', type=float, default=np.inf)

    # neural network settings
    parser.add_argument('--Phi-sizes', '-sPhi', type=int, nargs='*', default=[100, 100, 256])
    parser.add_argument('--F-sizes', '-sF', type=int, nargs='*', default=[100, 100, 100])
    parser.add_argument('--omnifold-arch', '-a', choices=['PFN'], default='PFN')
    parser.add_argument('--batch-size', '-bs', type=int, default=500)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--gpu', '-g', default='0')
    parser.add_argument('--input-dim', type=int, default=2)
    parser.add_argument('--patience', '-p', type=int, default=10)
    parser.add_argument('--save-best-only', action='store_true')
    parser.add_argument('--save-full-model', action='store_true')
    parser.add_argument('--val-frac', '-val', type=float, default=0.2)
    parser.add_argument('--verbose', '-v', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--ensemble', type=int,default=1)

    # training settings
    parser.add_argument('--max-iter', '-i', type=int, default=1)
    parser.add_argument('--name', '-n', default='')
    parser.add_argument('--start-iter', '-si', type=int, default=0)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--bsseed', type=int,default=12345)


    parser.add_argument('--MCbootstrap', action='store_true')
    parser.add_argument('--MCbsseed', type=int,default=1)
 
    parser.add_argument('--preweight', action='store_true')

    parser.add_argument('--dosysweight', action='store_true')
    parser.add_argument('--dogenreweight',action='store_true')

    parser.add_argument('--dataweight',default=None,choices=[None,'gen_CP5_to_EPOS_multifold'])

    parser.add_argument('--eff-acc',action='store_true')
    parser.add_argument('--testing', action='store_true', help="Run a shortened version for testing only.")

    p_args = parser.parse_args(args=args)
    p_args.data_path = MACHINES[p_args.machine]['data_path']
    p_args.results_path = MACHINES[p_args.machine]['results_path']

    

    return p_args

def train_omnifold(i):

    start = time.time()

    # load datasets
    if  isinstance(FILENAMES[args.dataset_mc],list):
      mc_preproc_list = [np.load(os.path.join(args.data_path, set_mc), allow_pickle=True,encoding="latin1") for set_mc in FILENAMES[args.dataset_mc]]
      gen=np.concatenate([mc[str('charged')] for mc in mc_preproc_list],axis=0)
      sim=np.concatenate([mc[str('tracks')] for mc in mc_preproc_list],axis=0)
      del mc_preproc_list
    else:
      mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True,encoding="latin1")
      gen, sim = mc_preproc[str('charged')], mc_preproc[str('tracks')]
      del mc_preproc
    if  isinstance(FILENAMES[args.dataset_data],list):
      real_preproc_list = [np.load(os.path.join(args.data_path, set_data), allow_pickle=True,encoding="latin1") for set_data in FILENAMES[args.dataset_data]]
      data=np.concatenate([data[str('tracks')] for data in real_preproc_list],axis=0)
      del real_preproc_list
    else:
      real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True,encoding="latin1")
      data=real_preproc[str('tracks')]
      del real_preproc

    # pad datasets
    start = time.time()
    sim_data_max_length = max(get_max_length(sim), get_max_length(data))
    gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
    data = pad_events(data, max_length=sim_data_max_length)
    print('Done padding in {:.3f}s'.format(time.time() - start))

    # detector/sim setup
    global X_det, Y_det
    X_det = (np.concatenate((data, sim), axis=0))
    Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))
    del data, sim

    # gen setup
    global X_gen, Y_gen
    X_gen = (np.concatenate((gen, gen)))
    Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(gen)), np.zeros(len(gen)))))
    del gen

    X_det[np.isnan(np.sum(X_det,axis=1))==False]=(X_det[np.isnan(np.sum(X_det,axis=1))==False] - np.mean(X_det[np.isnan(np.sum(X_det,axis=1))==False], axis=0))/np.std(X_det[np.isnan(np.sum(X_det,axis=1))==False], axis=0)
    X_det[np.isnan(X_det)]=-1.0
    X_gen[np.isnan(np.sum(X_gen,axis=1))==False]=(X_gen[np.isnan(np.sum(X_gen,axis=1))==False] - np.mean(X_gen[np.isnan(np.sum(X_gen,axis=1))==False], axis=0))/np.std(X_gen[np.isnan(np.sum(X_gen,axis=1))==False], axis=0)
    X_gen[np.isnan(X_gen)]=-1.0

    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models',  args.name + '_Iter-{}-Step1')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = getattr(ef.archs, args.omnifold_arch)
    print("PFN size",args.Phi_sizes,args.F_sizes)
    det_args = {'input_dim': args.input_dim, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes, 
                'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    mc_args = {'input_dim': args.input_dim, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes, 
               'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
               'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
               'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

    # apply the omnifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = np.load(DATAWEIGHT[args.dataweight],allow_pickle=True)[-1,:len(wdata)]
      wdata *= dataweight
    if args.bootstrap:
      np.random.seed(args.bsseed)
      wdata=wdata*np.random.poisson(1.0,len(wdata))
      wdata_path=os.path.join(args.results_path, 'weights', args.name+"_dataweight.npy")
      np.save(wdata_path,wdata)
    winit = np.sum(wdata)/nsim*np.ones(nsim)
    if args.MCbootstrap:
      np.random.seed(args.MCbsseed)
      for sys in SYSWEIGHTS[args.machine].keys():
        if SYSWEIGHTS[args.machine][sys] != '':
          wMC_weight = np.load(SYSWEIGHTS[args.machine][sys]+'/weights/OmniFold_PFN_Rep-0.npy',allow_pickle=True)
          winit *= np.power(wMC_weight[-1],np.random.normal())
        else:
          winit *= np.random.poisson(1.0,nsim)

    if args.preweight:
      preweightMC = np.load(PREWEIGHTS[args.machine],allow_pickle=True)
      winit = preweightMC[-1]

    ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), fitargs, 
                  val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),
                  delete_global_arrays=True,ensemble=args.ensemble)

    print('Finished OmniFold {} in {:.3f}s'.format(i, time.time() - start))


def train_omnifold_fitsys(i):

    start = time.time()

    insert_gen_tag1=np.vectorize(lambda x: np.insert(x,3,1.0,axis=-1),otypes=[np.ndarray])
    insert_reco_tag0=np.vectorize(lambda x: np.insert(x,3,0.0,axis=-1),otypes=[np.ndarray])
    merge_ndarray=np.vectorize(lambda x,y:np.concatenate((x,y),axis=1),otypes=[np.ndarray])

    if  isinstance(FILENAMES[args.dataset_mc],list):
      #mc_preproc_list = [np.load(os.path.join(args.data_path, set_mc), allow_pickle=True,encoding="latin1") for set_mc in FILENAMES[args.dataset_mc]]
      gen=[]
      sim=[]
      for set_mc in FILENAMES[args.dataset_mc]:
        with np.load(os.path.join(args.data_path, set_mc),allow_pickle=True,encoding="latin1") as mc:
          gen.append(insert_gen_tag1(mc[str('charged')]))
          sim.append(insert_reco_tag0(mc[str('tracks')]))
          print("Loaded: ",os.path.join(args.data_path, set_mc))
      gen=np.concatenate(gen)
      sim=np.concatenate(sim)
      #gen=insert_gen_tag1(np.concatenate([mc[str('charged')] for mc in mc_preproc_list],axis=0))
      #sim=insert_reco_tag0(np.concatenate([mc[str('tracks')] for mc in mc_preproc_list],axis=0))
      #del mc_preproc_list
    else:
      mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True,encoding="latin1")
      gen, sim = insert_gen_tag1(mc_preproc[str('charged')]), insert_reco_tag0(mc_preproc[str('tracks')])
      del mc_preproc

    if  isinstance(FILENAMES[args.dataset_data],list):
      #real_preproc_list = [np.load(os.path.join(args.data_path, set_data), allow_pickle=True, encoding="latin1") for set_data in FILENAMES[args.dataset_data]]
      data=[]
      truth=[]
      for set_data in FILENAMES[args.dataset_data]:
        with np.load(os.path.join(args.data_path, set_data), allow_pickle=True, encoding="latin1") as pseudodata:
          data.append(insert_reco_tag0(pseudodata[str('tracks')]))
          truth.append(insert_gen_tag1(pseudodata[str('charged')]))
          print("Loaded: ",os.path.join(args.data_path, set_data))
      data=np.concatenate(data)
      truth=np.concatenate(truth)
      #data=insert_reco_tag0(np.concatenate([data[str('tracks')] for data in real_preproc_list],axis=0))
      #truth=insert_gen_tag1(np.concatenate([data[str('charged')] for data in real_preproc_list],axis=0))
      #del real_preproc_list
    else:
      real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True,encoding="latin1")
      data, truth=insert_reco_tag0(real_preproc[str('tracks')]), insert_gen_tag1(real_preproc[str('charged')])
      del real_preproc

    # pad datasets
    sim_data_max_length = max(get_max_length(sim), get_max_length(data))
    gen_truth_max_length = max(get_max_length(gen),get_max_length(truth))
    gen, sim = pad_events(gen,max_length=gen_truth_max_length), pad_events(sim, max_length=sim_data_max_length)
    truth, data = pad_events(truth, max_length=gen_truth_max_length), pad_events(data, max_length=sim_data_max_length)
    print('Done padding in {:.3f}s'.format(time.time() - start))

    # detector/sim setup
    global X, Y
    X = (np.concatenate((np.concatenate((data,truth),axis=1), np.concatenate((sim,gen),axis=1)), axis=0))
    Y = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))
    del data, sim, truth, gen

    X[np.isnan(np.sum(X,axis=1))==False]=(X[np.isnan(np.sum(X,axis=1))==False] - np.mean(X[np.isnan(np.sum(X,axis=1))==False], axis=0))/np.std(X[np.isnan(np.sum(X,axis=1))==False], axis=0)
    X[np.isnan(X)]=-1.0

    model_fp = os.path.join(args.results_path, 'models',  args.name + '_Iter-{}-Step1')
    Model = getattr(ef.archs, args.omnifold_arch)
    print("PFN size",args.Phi_sizes,args.F_sizes)
    det_mc_args = {'input_dim': args.input_dim+1, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes,
                'patience': args.patience, 'filepath': model_fp, 'save_weights_only': args.save_full_model,
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
               'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

    ndata, nsim = np.count_nonzero(Y[:,1]), np.count_nonzero(Y[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = np.load(DATAWEIGHT[args.dataweight],allow_pickle=True)[-1,len(wdata)]
      wdata *= dataweight
    winit = np.sum(wdata)/nsim*np.ones(nsim)
    ws = omnifold.omnifold_sys(X, Y, wdata, winit, (Model, det_mc_args),
                  fitargs, val=args.val_frac, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),ensemble=args.ensemble)
    print('Finished OmniFold {} in {:.3f}s'.format(i, time.time() - start))

def train_omnifold_fitgen(i):

    start = time.time()

    if  isinstance(FILENAMES[args.dataset_mc],list):
      gen=[]
      for set_mc in FILENAMES[args.dataset_mc]:
        with np.load(os.path.join(args.data_path, set_mc),allow_pickle=True,encoding="latin1") as mc:
          gen.append(mc[str('charged')])
          print("Loaded: ",os.path.join(args.data_path, set_mc))
      gen=np.concatenate(gen)
    else:
      mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True,encoding="latin1")
      gen = mc_preproc[str('charged')]
      del mc_preproc

    if  isinstance(FILENAMES[args.dataset_data],list):
      truth=[]
      for set_data in FILENAMES[args.dataset_data]:
        with np.load(os.path.join(args.data_path, set_data), allow_pickle=True, encoding="latin1") as pseudodata:
          truth.append(pseudodata[str('charged')])
          print("Loaded: ",os.path.join(args.data_path, set_data))
      truth=np.concatenate(truth)
    else:
      real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True,encoding="latin1")
      truth = real_preproc[str('charged')]
      del real_preproc

    # pad datasets
    gen_truth_max_length = max(get_max_length(gen),get_max_length(truth))
    gen = pad_events(gen,max_length=gen_truth_max_length)
    truth = pad_events(truth, max_length=gen_truth_max_length)
    print('Done padding in {:.3f}s'.format(time.time() - start))

    # detector/sim setup
    global X, Y
    X = np.concatenate((truth, gen), axis=0)
    Y = ef.utils.to_categorical(np.concatenate((np.ones(len(truth)), np.zeros(len(gen)))))
    del truth, gen
    X[np.isnan(np.sum(X,axis=1))==False]=(X[np.isnan(np.sum(X,axis=1))==False] - np.mean(X[np.isnan(np.sum(X,axis=1))==False], axis=0))/np.std(X[np.isnan(np.sum(X,axis=1))==False], axis=0)
    X[np.isnan(X)]=-1.0
    model_fp = os.path.join(args.results_path, 'models',  args.name + '_Iter-{}-Step1')
    Model = getattr(ef.archs, args.omnifold_arch)
    print("PFN size",args.Phi_sizes,args.F_sizes)
    det_mc_args = {'input_dim': args.input_dim, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes,
                'patience': args.patience, 'filepath': model_fp, 'save_weights_only': args.save_full_model,
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
               'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

    ndata, nsim = np.count_nonzero(Y[:,1]), np.count_nonzero(Y[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = np.load(DATAWEIGHT[args.dataweight],allow_pickle=True)[-1,:len(wdata)]
      wdata *= dataweight
    winit = np.sum(wdata)/nsim*np.ones(nsim)
    ws = omnifold.omnifold_sys(X, Y, wdata, winit, (Model, det_mc_args),
                  fitargs, val=args.val_frac, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),ensemble=args.ensemble)
    print('Finished OmniFold {} in {:.3f}s'.format(i, time.time() - start))

def load_obs():

    # load datasets
    datasets = {args.dataset_mc: {}, args.dataset_data: {}}
    for dataset,v in datasets.items():
        filepath = '{}/{}_ZJet'.format(args.data_path, dataset)

        # load particles
        v.update(np.load(filepath + '.pickle', allow_pickle=True))

        # load npzs
        f = np.load(filepath + '.npz')
        v.update({k: f[k] for k in f.files})
        f.close()

        # load obs
        f = np.load(filepath + '_Obs.npz')
        v.update({k: f[k] for k in f.files})
        f.close()

    # choose what is MC and Data in this context
    mc, real = datasets[args.dataset_mc], datasets[args.dataset_data]

    # a dictionary to hold information about the observables
    obs = {
        'Mass': {'func': lambda dset, ptype: dset[ptype + '_jets'][:,3]},
        'Mult': {'func': lambda dset, ptype: dset[ptype + '_mults']},
        'Width': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,1]},
        'Tau21': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,4]/(dset[ptype + '_nsubs'][:,1] + 10**-50)},
        'zg': {'func': lambda dset, ptype: dset[ptype + '_zgs'][:,0]},
        'SDMass': {'func': lambda dset, ptype: np.log(dset[ptype + '_sdms'][:,0]**2/dset[ptype + '_jets'][:,0]**2 + 10**-100)},
        'LHA': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,0]},
        'e2': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,2]},
        'Tau32': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,7]/(dset[ptype + '_nsubs'][:,4] + 10**-50)},
        'Rapidity': {'func': lambda dset, ptype: dset[ptype + '_jets'][:,1]}
    }

    # calculate quantities to be stored in obs
    for obkey,ob in obs.items():

        # calculate observable for GEN, SIM, DATA, and TRUE
        ob['genobs'], ob['simobs'] = ob['func'](mc, 'gen'), ob['func'](mc, 'sim')
        ob['truthobs'], ob['dataobs'] = ob['func'](real, 'gen'), ob['func'](real, 'sim')
        print('Done computing', obkey)

    print()
    del mc, real, datasets
    gc.collect()

    return obs


def load_data( files, keys, max_size=None ):
    if  isinstance(files,list):
      if max_size: #assuming max size is less than whats contained in a single file
        files = files[:1]
      preproc_list = [np.load(os.path.join(args.data_path, set_mc), allow_pickle=True,encoding="latin1") for set_mc in files]
      if isinstance(preproc_list[0], np.ndarray):
        preproc = np.concatenate([preproc_list])
      preproc = { k: np.concatenate([f[k] for f in preproc_list],axis=0)
                     for k in keys}
      del preproc_list
    else:
      preproc = np.load(os.path.join(args.data_path, files), allow_pickle=True,encoding="latin1")

    if isinstance(preproc, np.ndarray):
        if len(preproc.shape) > 1:
            preproc = preproc[keys,:max_size]
        else:
            preproc = preproc[:max_size]
    else:
        preproc = { k : v[:max_size] for k, v in preproc.items() }

    return preproc


def standardize_array( arr ):
    mask = ~np.isnan(np.sum(arr,axis=1))
    arr[mask] = arr[mask] - np.mean(arr[mask], axis=0)/np.std(arr[mask], axis=0)
    arr[np.isnan(arr)] = -1
    return arr


def make_full_arrays( smp1, smp2, keys ):
    features = np.asarray([np.concatenate([smp1[k], smp2[k]]) for k in keys]).T
    labels   = ef.utils.to_categorical( np.concatenate( [np.ones(len(smp1[keys[0]])), np.zeros(len(smp2[keys[0]])) ] ) )

    features = standardize_array(features)
    return features, labels


def make_model_args( modelpath, keys, cl_args ):
    arg_dict = { 'input_dim'  : len(keys),
                 'dense_sizes': cl_args.F_sizes,
                 'patience'   : cl_args.patience,
                 'filepath'   : modelpath,
                 'save_weights_only': cl_args.save_full_model,
                 'modelcheck_opts': {'save_best_only': cl_args.save_best_only, 'verbose': 0}
               }
    return arg_dict

def make_fit_args( cl_args):
    arg_dict = {'batch_size': cl_args.batch_size,
                'epochs': cl_args.epochs,
                'verbose': cl_args.verbose,
                'weight_clip_min': cl_args.weight_clip_min,
                'weight_clip_max': cl_args.weight_clip_max
               }
    return arg_dict


def do_bootstrap( wlen, bsseed, result_path, fname ):
      np.random.seed(bsseed)
      wdata=wdata*np.random.poisson(1.0, wlen)
      wdata_path=os.path.join(results_path, 'weights', fname+"_dataweight.npy")
      np.save(wdata_path,wdata)
      return wdata, wdata_path

def do_mc_bootstrap( winit, nsim, sysweights, seed ):
      np.random.seed(seed)
      for sys in sysweights.keys():
        if sysweights[sys] != '':
          wMC_weight = np.load(sysweights[sys]+'/weights/ManyFold_DNN_Rep-0.npy',allow_pickle=True)
          winit *= np.power(wMC_weight[-1],np.random.normal())
        else:
          winit *= np.random.poisson(1.0,nsim)
      return winit

def train_manyfold(i, full_reweight=False, gen_only=False):

    # which observables to include in manyfold
    #recokeys = ['reco_mass','reco_ntrk']
    #genkeys = ['gen_mass','gen_nch']

    #recokeys = ['reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy']
    #genkeys = ['gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy']

    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']
    extra_keys = ['charged','tracks']
    unf_iters = args.unfolding_iterations
    if args.testing:
        recokeys = recokeys[:min(3,len(recokeys)+1)]
        genkeys = genkeys[:min(3,len(genkeys)+1)]
    do_step2 = True

    step1_keys = genkeys
    step2_keys = recokeys
    if full_reweight:
        step1_keys = genkeys + recokeys
        step2_keys = []
        do_step2 = False
        unf_iters = 1
    if gen_only:
        step1_keys = genkeys
        step2_keys = []
        extra_keys = ['charged']
        do_step2 = False
        unf_iters = 1

    start = time.time()
    print('ManyFolding')

    max_size = 10000 if args.testing else None
    all_keys = step1_keys + step2_keys + extra_keys
    mc_preproc  = load_data(FILENAMES[args.dataset_mc], all_keys, max_size)
    real_preproc = load_data( FILENAMES[args.dataset_data], all_keys, max_size )

    ## detector/sim setup
    X_det, Y_det = make_full_arrays( real_preproc, mc_preproc, keys=step1_keys)
    X_gen, Y_gen = None, None
    if do_step2:
        X_gen, Y_gen = make_full_arrays( mc_preproc, mc_preproc, keys=step2_keys )

    del mc_preproc, real_preproc

    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1')
    if do_step2:
        model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = ef.archs.DNN
    print("DNN size",args.F_sizes)

    det_args = make_model_args( model1_fp, step1_keys, args)
    mc_args  = make_model_args( model2_fp, step2_keys, args) if do_step2 else None
    fitargs  = make_fit_args( args )

    # apply the unifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = load_data(DATAWEIGHT[args.dataweight],-1,max_size)
      wdata *= dataweight

    if args.bootstrap:
        wdata, wdata_path = do_bootstrap( len(wdata), args.bsseed, args.result_path, args.name )

    winit = np.sum(wdata)/nsim*np.ones(nsim)
    if args.MCbootstrap:
        winit = do_mc_bootstrap(winit, nsim, SYSWEIGHTS[args.machine], args.MCbsseed )

    if args.preweight:
      preweightMC = load_data(PREWEIGHTS[args.machine],-1,max_size)
      winit = preweightMC

    ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                  fitargs, val=args.val_frac, it=unf_iters, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),ensemble=args.ensemble)

    print('Finished ManyFold {} in {:.3f}s\n'.format(i, time.time() - start))
    print("Weight in ",os.path.join(args.results_path, 'weights', args.name))


def train_manyfold_acceptance_efficiency(i):


    # which observables to include in manyfold

    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']
    if args.testing:
        recokeys = recokeys[:min(3,len(recokeys)+1)]
        genkeys = genkeys[:min(3,len(genkeys)+1)]

    start = time.time()
    print('ManyFolding')


    max_size = 10000 if args.testing else None
    all_keys = recokeys + genkeys + ['charged','tracks']
    mc_preproc  = load_data(FILENAMES[args.dataset_mc], all_keys, max_size)
    real_preproc = load_data( FILENAMES[args.dataset_data], all_keys, max_size )

    mc_pass_reco=(np.isnan(mc_preproc['reco_ntrk'])==False)
    mc_pass_gen=(mc_preproc['gen_nch']>2)
    det_pass_reco=np.concatenate((np.ones(len(real_preproc['reco_ntrk']),dtype=bool),mc_pass_reco))
    det_pass_gen=np.concatenate((np.ones(len(real_preproc['reco_ntrk']),dtype=bool),mc_pass_gen))

    det_pass_reco_acc_reweight=np.concatenate((mc_pass_reco,mc_pass_reco))
    det_pass_gen_acc_reweight=np.concatenate((mc_pass_gen,mc_pass_gen))

    X_det, Y_det = make_full_arrays(real_preproc, mc_preproc, recokeys )
    X_gen, Y_gen = make_full_arrays(mc_preproc, mc_preproc, genkeys )
    X_det_acc_reweight, Y_det_acc_reweight = make_full_arrays( mc_preproc, mc_preproc, recokeys)

    gen_pass_gen = np.concatenate((mc_pass_gen,mc_pass_gen))
    gen_pass_reco = np.concatenate((mc_pass_reco,mc_pass_reco))
    del mc_preproc, real_preproc


    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1')
    model1b_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1b')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    model2b_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2b')
    Model = ef.archs.DNN

    print("DNN size",args.F_sizes)
    det_args    = make_model_args( model1_fp, recokeys, args )
    mc_args_1b  = make_model_args( model1b_fp, genkeys, args )
    mc_args     = make_model_args( model2_fp, genkeys, args  )
    det_args_2b = make_model_args( model2b_fp, recokeys, args )
    fitargs    = make_fit_args( args )

    # apply the unifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = load_data(DATAWEIGHT[args.dataweight],-1,max_size)
      wdata *= dataweight
    if args.bootstrap:
        wdata, wdata_path = do_bootstrap( len(wdata), args.bsseed, args.result_path, args.name )

    winit = np.sum(wdata)/nsim*np.ones(nsim)
    if args.MCbootstrap:
      winit = do_mc_bootstrap(winit, nsim, SYSWEIGHTS[args.machine], args.MCbsseed )

    if args.preweight:
      preweightMC = load_data(PREWEIGHTS[args.machine],-1,max_size)
      winit = preweightMC

    ws = omnifold.omnifold_acceptance_efficiency(X_gen, Y_gen, X_det, Y_det,X_det_acc_reweight,Y_det_acc_reweight, wdata, winit, gen_pass_gen,gen_pass_reco, det_pass_gen,det_pass_reco,det_pass_gen_acc_reweight,det_pass_reco_acc_reweight,
                  (Model, det_args), (Model, mc_args),(Model, mc_args_1b),(Model, det_args_2b),
                  fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),ensemble=args.ensemble)

    print('Finished ManyFold {} in {:.3f}s\n'.format(i, time.time() - start))
    print("Weight in ",os.path.join(args.results_path, 'weights', args.name))


def train_manyfold_fitsys(i, gen_only=False):

    # which observables to include in manyfold

    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']
    extra_keys = ['charged','tracks']
    if args.testing:
        recokeys = recokeys[:min(3,len(recokeys)+1)]
        genkeys = genkeys[:min(3,len(genkeys)+1)]

    obs_keys = genkeys + recokeys
    if gen_only:
        obs_keys = genkeys
        extra_keys = ['charged']

    start = time.time()
    print('ManyFolding')

    max_size = 10000 if args.testing else None
    mc_preproc  = load_data(FILENAMES[args.dataset_mc], obs_keys + extra_keys, max_size)
    real_preproc = load_data( FILENAMES[args.dataset_data], obs_keys + extra_keys, max_size )

    X, Y = make_full_arrays( real_preproc, mc_preproc, obs_keys)
    del mc_preproc, real_preproc

    # specify the model and the training parameters
    model_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1')
    Model = ef.archs.DNN

    print("DNN size",args.F_sizes)
    det_mc_args = make_model_args( model_fp, obs_keys, args )
    fitargs     = make_fit_args( args )

    ndata, nsim = np.count_nonzero(Y[:,1]), np.count_nonzero(Y[:,0])
    wdata = np.ones(ndata)
    if args.dataweight is not None:
      dataweight = load_data(DATAWEIGHT[args.dataweight],-1,max_size)
      wdata *= dataweight
    winit = np.sum(wdata)/nsim*np.ones(nsim)
    ws = omnifold.omnifold_sys(X, Y, wdata, winit, (Model, det_mc_args),
                  fitargs, val=args.val_frac, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),ensemble=args.ensemble)

    print('Finished ManyFold {} in {:.3f}s\n'.format(i, time.time() - start))
    print("Weight in ",os.path.join(args.results_path, 'weights', args.name))


def train_unifold(i):
    keys = ['nch','spherocity','thrust','broaden','transversespherocity','transversethrust','isotropy','pt']
    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']
    if args.testing:
        keys = ['nch']
        recokeys = ['reco_ntrk']
        genkeys = ['gen_nch']
    #keys = ['spherocity','thrust','broaden','transversespherocity','transversethrust','isotropy']
    #recokeys = ['reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy']
    #genkeys = ['gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy']

    #keys=["transversespherocity","transversethrust"]
    #recokeys=["reco_transversespherocity","reco_transversethrust"]
    #genkeys=["gen_transversespherocity","gen_transversethrust"]

    max_size = 10000 if args.testing else None
    all_keys = recokeys + genkeys + ['charged','tracks']
    mc_preproc  = load_data(FILENAMES[args.dataset_mc], all_keys, max_size)
    real_preproc = load_data( FILENAMES[args.dataset_data], all_keys, max_size )

    # UniFold
    for (key,recokey,genkey) in zip(keys,recokeys,genkeys):
        start = time.time()

        print('Un[i]Folding', recokey, genkey)
        ob_filename = args.name.format(key)        

        X_det, Y_det = make_full_arrays( real_preproc, mc_preproc, [recokey] )
        X_gen, Y_gen = make_full_arrays( real_preproc, mc_preproc, [genkey] )

        # specify the model and the training parameters
        model1_fp = os.path.join(args.results_path, 'models',  ob_filename + '_Iter-{}-Step1')
        model2_fp = os.path.join(args.results_path, 'models', ob_filename + '_Iter-{}-Step2')
        Model = ef.archs.DNN

        det_args = make_model_args( model1_fp, [recokey], args )
        mc_args  = make_model_args( model2_fp, [genkey], args )
        fitargs  = make_fit_args( args )

        # apply the unifold technique to this one dimensional space
        ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
        wdata = np.ones(ndata)
        if args.dataweight is not None:
          dataweight = load_data(DATAWEIGHT[args.dataweight],-1,max_size)
          wdata *= dataweight
        if args.bootstrap:
            wdata, wdata_path = do_bootstrap( len(wdata), args.bsseed, args.result_path, args.name )

        winit = np.sum(wdata)/nsim*np.ones(nsim)
        if args.MCbootstrap:
            winit = do_mc_bootstrap(winit, nsim, SYSWEIGHTS[args.machine], args.MCbsseed )

        if args.preweight:
          preweightMC = load_data(PREWEIGHTS[args.machine],-1,max_size)
          winit = preweightMC
        ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                      fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                      weights_filename=os.path.join(args.results_path, 'weights', ob_filename),ensemble=args.ensemble)

        print('Finished UniFold {} for {} in {:.3f}s\n'.format(i, key, time.time() - start))

def pad_events(events, val=0, max_length=None):
    event_lengths = [event.shape[0] for event in events]
    if max_length is None:
        max_length = max(event_lengths)
    return np.asarray([np.vstack((event, val*np.ones((max_length - ev_len, event.shape[1]))))
                       for event,ev_len in zip(events, event_lengths)],dtype=np.float32)

def get_max_length(events):
    return max([event.shape[0] for event in events])
if __name__ == '__main__':
    main(sys.argv[1:])
