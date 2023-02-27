import re
import fileinput
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--outputfile',default='flatTuple_MB_trk_noPU_new.npz',help='name of the output .npz file')
parser.add_argument('--inputfile',default='slurm-multifold_MCEPOS_unfoldCP1.out',help='name of the input text file')
args=parser.parse_args()
pattern_setup=re.compile('Train on ([0-9]+) samples, validate on ([0-9]+) samples')
pattern_epoch=re.compile('Epoch ([0-9]+)/50')
pattern_train=re.compile('([0-9]+)/([0-9]+) - ([0-9]+)s - loss: ([0-9]+\.[0-9]+) - acc: ([0-9]+\.[0-9]+) - val_loss: ([0-9]+\.[0-9]+) - val_acc: ([0-9]+\.[0-9]+)')

fout=args.outputfile
fin=args.inputfile
it=0
step=0
train_round=-1
epoch=0
result_train=[]
result_val=[]
for line in open(fin,'r').readlines():
  match_setup = pattern_setup.match(line)
  if match_setup:
    train_round+=1
    it=train_round/2
    step=train_round%2
    epoch=0
    #print "iter ",it,", step ",step
    if step==0:
      result_train.append([[],[]])
      result_val.append([[],[]])
  match_epoch = pattern_epoch.match(line)
  if match_epoch:
    epoch+=1
    #print "epoch ",epoch
  match_train = pattern_train.match(line)
  if match_train:
    train_loss=match_train.group(4)
    val_loss=match_train.group(6)
    #print "train_loss ",train_loss,", val_loss ",val_loss
    result_train[-1][step].append(float(train_loss))
    result_val[-1][step].append(float(val_loss))
#print [np.array([np.array(result_train_iter[0]),np.array(result_train_iter[1])]) for result_train_iter in result_train]
#print len([np.array([np.array(result_train_iter[0]),np.array(result_train_iter[1])]) for result_train_iter in result_train])
#print np.asarray([np.array([np.array(result_train_iter[0]),np.array(result_train_iter[1])]) for result_train_iter in result_train])
np.savez(fout,train_loss=np.asarray(result_train),val_loss=np.asarray(result_val))
