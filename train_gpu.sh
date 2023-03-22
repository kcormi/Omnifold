#!/bin/bash
#
#SBATCH --job-name=test_job 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                       # request to run job on single node                                       
#SBATCH --ntasks=10                    # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
##SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=40G                     # memory (per job)
#SBATCH --time=1-12:00                   # time  in format DD-HH:MM


# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}


# Slurm reserves two GPU's (according to requirement above), those ones that are recorded in shell variable CUDA_VISIBLE_DEVICES
echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES

# QUICK TESTING
python mytrain.py -m unifold -u unifold -data Pythia8CP1_tuneES -mc Pythia8CP1_tuneES  -e 3 -ui 2 --save-best-only --weight-clip-max 100.0 --testing

#python mytrain.py -m omnifold -u omnifold --input-dim 3 -e 50 -ui 6 -data Pythia8CP1
#python mytrain.py -m omnifold -u omnifold -data Pythia8CP1 --input-dim 3 -e 50 -ui 6
#python mytrain.py -m multifold -u manyfold -e 50 -ui 8 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m omnifold -u omnifold --input-dim 3 -e 50 -ui 8 --save-best-only --weight-clip-max 100.0
# python mytrain.py -m omnifold -u omnifold --input-dim 3 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0 --bootstrap 
# python mytrain.py -m unifold -u unifold -e 50 -ui 8 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold  -e 20 -ui 6 --save-best-only --weight-clip-max 100.0 -data Pythia8CP1

##ASIMOV

#python mytrain.py -m omnifold -u omnifold -data Pythia8CP5 --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -data Pythia8CP5 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -data Pythia8CP5  -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m omnifold -u omnifold -data Pythia8CP1_tuneES -mc Pythia8CP1_tuneES --input-dim 3 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -data Pythia8CP1_tuneES -mc Pythia8CP1_tuneES -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -data Pythia8CP1_tuneES -mc Pythia8CP1_tuneES  -e 50 -ui 6 --save-best-only --weight-clip-max 100.0

##SPLIT TEST
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP5_part1 -data Pythia8CP5_part2 --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP5_part1 -data Pythia8CP5_part2 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -mc Pythia8CP5_part1 -data Pythia8CP5_part2 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0

#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneES_part1 -data Pythia8CP1_tuneES_part2 --input-dim 3 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES_part1 -data Pythia8CP1_tuneES_part2 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -mc Pythia8CP1_tuneES_part1 -data Pythia8CP1_tuneES_part2 -e 50 -ui 6 --save-best-only --weight-clip-max 100.0


#Unfold using CP1 as MC
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1 --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1 -e 50 -ui 6 --weight-clip-max 100.0


#Unfold using CP1_tuneES as MC
# python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneES --input-dim 3 -e 50 -ui 8 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES -e 50 -ui 8 --save-best-only --weight-clip-max 100.0
 #python mytrain.py -m unifold -u unifold -mc Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold using CP1_tuneNch as MC
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneNch --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneNch -e 50 -ui 6 --weight-clip-max 100.0

#Unfold using CP1_tuneES_trkdrop as MC
# python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneES_trkdrop --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES_trkdrop -e 50 -ui 8 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m unifold -u unifold -mc Pythia8CP1_tuneES_trkdrop -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold using CP5_trkdrop as MC
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP5_trkdrop --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP5_trkdrop -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m unifold -u unifold -mc Pythia8CP5_trkdrop -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold CP1_tuneES using CP5_trkdrop as MC
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP5_trkdrop -data Pythia8CP1_tuneES --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP5_trkdrop -data Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m unifold -u unifold -mc Pythia8CP5_trkdrop -data Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only


#Unfold using CP1_tuneES pseudodata
 #python mytrain.py -m multifold -u manyfold -data Pythia8CP1_tuneES -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -data Pythia8CP1_tuneES -e 20 -ui 9 --save-best-only --weight-clip-max 100.0
# python mytrain.py -m unifold -u unifold -data Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold using CP5 pseudodata with CP1_tuneES
# python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES -data Pythia8CP5 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
# python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES -data Pythia8CP5 -e 50 -ui 8 --weight-clip-max 100.0 --save-best-only
# python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneES -data Pythia8CP5 --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold CP1_tuneES_trkdrop with CP1_tuneES
#  python mytrain.py -m omnifold -u omnifold -data Pythia8CP1_tuneES_trkdrop -mc Pythia8CP1_tuneES --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m multifold -u manyfold -data Pythia8CP1_tuneES_trkdrop -mc Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m unifold -u unifold -data Pythia8CP1_tuneES_trkdrop -mc Pythia8CP1_tuneES -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold with preweights
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1_tuneES --preweight -e 50 -ui 8 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m omnifold -u omnifold -mc Pythia8CP1_tuneES --input-dim 3 --preweight -e 50 -ui 6 --save-best-only --weight-clip-max 100.0
#python mytrain.py -m unifold -u unifold -mc Pythia8CP1_tuneES --preweight -e 50 -ui 8 --save-best-only --weight-clip-max 100.0


#Unfold using Pythia8EPOS as MC
 #python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -e 50 -ui 6 --save-best-only --weight-clip-max 100.0 --MCbootstrap --MCbsseed 20
 #python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m unifold -u unifold -mc Pythia8EPOS -e 50 -ui 8 --save-best-only --weight-clip-max 100.0

# python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS_trkdrop --input-dim 3 -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only
# python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS_trkdrop  -e 50 -ui 6 --weight-clip-max 100.0 --save-best-only

#Unfold using Pythia8EPOS as MC to unfold CP5
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP5 -e 50 -ui 70 --weight-clip-max 10.0 --save-best-only --dogenreweight
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8CP5 --input-dim 3 -e 50 -ui 70 --save-best-only --weight-clip-max 10.0 --dosysweight

#Unfold using Pythia8EPOS as MC to reweight to CP5 which is first weighted to EPOS at gen-level
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP5 -e 50 -ui 70 --weight-clip-max 10.0 --save-best-only --dosysweight --dataweight gen_CP5_to_EPOS_multifold

#Unfold using CP5 as MC to unfold Pythia8EPOS 
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP5 -data Pythia8EPOS -e 50 -ui 70 --weight-clip-max 10.0 --save-best-only --dogenreweight

#Unfold using CP1 as MC to unfold Pythia8EPOS 
#python mytrain.py -m multifold -u manyfold -mc Pythia8CP1 -data Pythia8EPOS -e 50 -ui 70 --weight-clip-max 10.0 --save-best-only --dogenreweight

#Unfold using Pythia8EPOS as MC to unfold CP1_tuneES
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP1_tuneES -e 50 -ui 8 --weight-clip-max 100.0 --save-best-only
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8CP1_tuneES --input-dim 3 -e 50 -ui 8 --save-best-only --weight-clip-max 100.0

#Unfold using Pythia8EPOS as MC to unfold Pythia8EPOS_trkdrop
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8EPOS_trkdrop -e 50 -ui 20 --weight-clip-max 10.0 --save-best-only --dosysweight
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8EPOS_trkdrop --input-dim 3 -e 50 -ui 20 --save-best-only --weight-clip-max 10.0 --dosysweight

#Unfold using Pythia8EPOS as MC to unfold CP1
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP1 -e 50 -ui 20 --weight-clip-max 10.0 --save-best-only --eff-acc --ensemble 4
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP1 -e 50 -ui 20 --weight-clip-max 10.0 --save-best-only --ensemble 4 --dogenreweight
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8CP1 --input-dim 3 -e 50 -ui 50 --save-best-only --weight-clip-max 10.0 --dosysweight

#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Pythia8CP1 -e 50 -ui 70 --weight-clip-max 100.0 --save-best-only -sF 100 100
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8CP1 --input-dim 3 -e 50 -ui 70 --save-best-only --weight-clip-max 100.0 -sPhi 100 100 100 256
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Pythia8CP1 --input-dim 3 -e 50 -ui 70 --save-best-only --weight-clip-max 100.0 -sF 100 100

#Unfold using EPOS as MC to unfold data
#python mytrain.py -m multifold -u manyfold -mc Pythia8EPOS -data Zerobias -e 50 -ui 20 --weight-clip-max 10.0 --save-best-only -i 5
#python mytrain.py -m omnifold -u omnifold -mc Pythia8EPOS -data Zerobias --input-dim 3 -e 50 -ui 20 --weight-clip-max 10.0 --save-best-only -i 5

# cleaning of temporal working dir when job was completed:
rm -rf /scratch/$USER/${SLURM_JOB_ID}

