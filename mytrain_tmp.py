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
        'results_path': '/work/jinw/omnifold/OmniFold/results_multifold_maxweight_MCCP1_tuneES'
    },
    'omnifold': {
        'data_path': '/work/jinw/omnifold/OmniFold/preselect',
        'results_path': '/work/jinw/omnifold/OmniFold/results_omnifold_maxweight_MCCP1_tuneES'
    },
    'unifold':{
        'data_path': '/work/jinw/omnifold/OmniFold/preselect',
        'results_path': '/work/jinw/omnifold/OmniFold/results_unifold_maxweight_MCCP1_tuneES'
    },
}

# default filenames
FILENAMES = {
    'Zerobias': 'flatTuple_ZeroBias_2018lowPU_new_hltv6_iso_nparticle.npz',
    'Pythia8CP5': 'flatTuple_MB_trk_noPU_new.npz',
    'Pythia8CP1': 'flatTuple_MB_trk_noPU_new_CP1.npz',
    'Pythia8CP1_tuneES': 'flatTuple_MB_trk_noPU_new_CP1_tuneES.npz',
    'Pythia8CP1_tuneNch': 'flatTuple_MB_trk_noPU_new_CP1_tuneNch.npz',
    'Pythia8CP5_trkdrop': 'flatTuple_MB_trk_noPU_new_trackdrop.npz',
    'Pythia8CP1_tuneES_trkdrop': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_trackdrop.npz',
    'Pythia8CP5_part1': 'flatTuple_MB_trk_noPU_new_1.npz',
    'Pythia8CP5_part2': 'flatTuple_MB_trk_noPU_new_2.npz',
    'Pythia8CP1_tuneES_part1': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_1.npz',
    'Pythia8CP1_tuneES_part2': 'flatTuple_MB_trk_noPU_new_CP1_tuneES_2.npz',
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
            train_omnifold(i)
        elif args.unfolding == 'manyfold':
            args.name = name.format(i)
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

    # training settings
    parser.add_argument('--max-iter', '-i', type=int, default=1)
    parser.add_argument('--name', '-n', default='')
    parser.add_argument('--start-iter', '-si', type=int, default=0)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--bsseed', type=int,default=12345)


    parser.add_argument('--MCbootstrap', action='store_true')
    parser.add_argument('--MCbsseed', type=int,default=1)
    p_args = parser.parse_args(args=args)
    p_args.data_path = MACHINES[p_args.machine]['data_path']
    p_args.results_path = MACHINES[p_args.machine]['results_path']

    return p_args

def train_omnifold(i):

    start = time.time()

    # load datasets

    mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True)
    real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True)
    gen, sim, data = mc_preproc[str('charged')], mc_preproc[str('tracks')], real_preproc[str('tracks')]
    del mc_preproc, real_preproc

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
    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models',  args.name + '_Iter-{}-Step1')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = getattr(ef.archs, args.omnifold_arch)
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
    if args.bootstrap:
      np.random.seed(args.bsseed)
      wdata=wdata*np.random.poisson(1.0,len(wdata))
      wdata_path=os.path.join(args.results_path, 'weights', args.name+"_dataweight.npy")
      np.save(wdata_path,wdata)
    winit = ndata/nsim*np.ones(nsim)
    if args.MCbootstrap:
      np.random.seed(args.MCbsseed)
      for sys in SYSWEIGHTS[args.machine].keys():
        if SYSWEIGHTS[args.machine][sys] != '':
          wMC_weight = np.load(SYSWEIGHTS[args.machine][sys]+'/weights/OmniFold_PFN_Rep-0.npy',allow_pickle=True)
          winit *= np.power(wMC_weight[-1],np.random.normal())
        else:
          winit *= np.random.poisson(1.0,nsim)
    ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), fitargs, 
                  val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),
                  delete_global_arrays=True)

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

def train_manyfold(i):


    # which observables to include in manyfold

    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']

    start = time.time()
    print('ManyFolding')

    mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True)
    real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True)
    gen, sim, data = [mc_preproc['charged']], mc_preproc['tracks'], real_preproc['tracks']
    
    # detector/sim setup
    X_det = np.asarray([np.concatenate((real_preproc[obkey], mc_preproc[obkey])) for obkey in recokeys]).T
    Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(real_preproc['reco_ntrk'])), np.zeros(len(mc_preproc['reco_ntrk'])))))

    # gen setup
    X_gen = np.asarray([np.concatenate((mc_preproc[obkey], mc_preproc[obkey])) for obkey in genkeys]).T
    Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(mc_preproc['gen_nch'])), np.zeros(len(mc_preproc['gen_nch'])))))
    del mc_preproc, real_preproc

    # standardize the inputs
    X_det = (X_det - np.mean(X_det, axis=0))/np.std(X_det, axis=0)
    X_gen = (X_gen - np.mean(X_gen, axis=0))/np.std(X_gen, axis=0)

    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = ef.archs.DNN
    det_args = {'input_dim': len(recokeys), 'dense_sizes': args.F_sizes, 
                'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    mc_args = {'input_dim': len(genkeys), 'dense_sizes': args.F_sizes, 
               'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
               'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
               'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

    # apply the unifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    if args.bootstrap:
      np.random.seed(args.bsseed)
      wdata=wdata*np.random.poisson(1.0,len(wdata))
      wdata_path=os.path.join(args.results_path, 'weights', args.name+"_dataweight.npy")
      np.save(wdata_path,wdata)
    winit = ndata/nsim*np.ones(nsim)
    if args.MCbootstrap:
      np.random.seed(args.MCbsseed)
      for sys in SYSWEIGHTS[args.machine].keys():
        if SYSWEIGHTS[args.machine][sys] != '':
          wMC_weight = np.load(SYSWEIGHTS[args.machine][sys]+'/weights/ManyFold_DNN_Rep-0.npy',allow_pickle=True)
          winit *= np.power(wMC_weight[-1],np.random.normal())
        else:
          winit *= np.random.poisson(1.0,nsim)
    ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                  fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name))

    print('Finished ManyFold {} in {:.3f}s\n'.format(i, time.time() - start))

def train_unifold(i):
    keys = ['nch','spherocity','thrust','broaden','transversespherocity','transversethrust','isotropy','pt']
    recokeys = ['reco_ntrk','reco_spherocity','reco_thrust','reco_broaden','reco_transversespherocity','reco_transversethrust','reco_isotropy','reco_pt']
    genkeys = ['gen_nch','gen_spherocity','gen_thrust','gen_broaden','gen_transversespherocity','gen_transversethrust','gen_isotropy','gen_pt']

    #keys=["transversespherocity","transversethrust"]
    #recokeys=["reco_transversespherocity","reco_transversethrust"]
    #genkeys=["gen_transversespherocity","gen_transversethrust"]

    mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True)
    real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True)
    # UniFold
    for (key,recokey,genkey) in zip(keys,recokeys,genkeys):
        start = time.time()

        print('Un[i]Folding', recokey, genkey)
        ob_filename = args.name.format(key)        

        # detector/sim setup
        X_det = (np.concatenate((real_preproc[recokey], mc_preproc[recokey]), axis=0))
        Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(real_preproc[recokey])), np.zeros(len(mc_preproc[recokey])))))

        # gen setup
        X_gen = (np.concatenate((mc_preproc[genkey], mc_preproc[genkey])))
        Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(mc_preproc[genkey])), np.zeros(len(mc_preproc[genkey])))))
        
        # standardize the inputs
        X_det = (X_det - np.mean(X_det))/np.std(X_det)
        X_gen = (X_gen - np.mean(X_gen))/np.std(X_gen)

        # specify the model and the training parameters
        model1_fp = os.path.join(args.results_path, 'models',  ob_filename + '_Iter-{}-Step1')
        model2_fp = os.path.join(args.results_path, 'models', ob_filename + '_Iter-{}-Step2')
        Model = ef.archs.DNN
        det_args = {'input_dim': 1, 'dense_sizes': args.F_sizes, 
                    'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                    'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
        mc_args = {'input_dim': 1, 'dense_sizes': args.F_sizes, 
                   'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
                   'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
        fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
                   'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

        # apply the unifold technique to this one dimensional space
        ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
        wdata = np.ones(ndata)
        if args.bootstrap:
          np.random.seed(args.bsseed)
          wdata=wdata*np.random.poisson(1.0,len(wdata))
          wdata_path=os.path.join(args.results_path, 'weights', ob_filename+"_dataweight.npy")
          np.save(wdata_path,wdata)
        winit = ndata/nsim*np.ones(nsim)
        if args.MCbootstrap:
          np.random.seed(args.MCbsseed)
          for sys in SYSWEIGHTS[args.machine].keys():
            if SYSWEIGHTS[args.machine][sys] != '':
              wMC_weight = np.load(SYSWEIGHTS[args.machine][sys]+'/weights/UniFold_DNN_{}_Rep-0.npy'.format(key),allow_pickle=True)
              winit *= np.power(wMC_weight[-1],np.random.normal())
            else:
              winit *= np.random.poisson(1.0,nsim)
        ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                      fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                      weights_filename=os.path.join(args.results_path, 'weights', ob_filename))

        print('Finished UniFold {} for {} in {:.3f}s\n'.format(i, key, time.time() - start))

def pad_events(events, val=0, max_length=None):
    event_lengths = [event.shape[0] for event in events]
    if max_length is None:
        max_length = max(event_lengths)
    return np.asarray([np.vstack((event, val*np.ones((max_length - ev_len, event.shape[1]))))
                       for event,ev_len in zip(events, event_lengths)])

def get_max_length(events):
    return max([event.shape[0] for event in events])
if __name__ == '__main__':
    main(sys.argv[1:])
