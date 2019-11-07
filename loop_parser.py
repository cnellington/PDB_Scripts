# Nao and Caleb

from pyrosetta import *
from scipy.spatial import distance, distance_matrix
import matplotlib.pyplot as plt

import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import os
import glob
import random
import matplotlib.pyplot as plt
import sys
import re
init(extra_options="-constant_seed -mute all")

# GATHER FULL PDB DATA
amino_acids = ['-','A','V','I','L','M','F','Y','W','S','T','N','Q','C','G','P','R','H','K','D','E']
aa_mapping = {amino_acids[i]:i for i in range(len(amino_acids))}
reverse_aa_mapping = {i:amino_acids[i] for i in range(len(amino_acids))}
structure_mapping = {"H":1, "L":2, "E":3}
reverse_structure_mapping = {1:"H", 2:"L", 3:"E", 0:"-"}

# Gets distance for various atoms given a pose
def get_distmap(pose, atom1="CB", atom2="CB", default="CA"):
    psize = pose.size()
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(1, psize+1):
        r = pose.residue(i)
        if type(atom1) == str:
            if r.has(atom1):
                xyz1[i-1, :] = np.array(r.xyz(atom1))
            else:
                xyz1[i-1, :] = np.array(r.xyz(default))
        else:
            xyz1[i-1, :] = np.array(r.xyz(atom1.get(r.name(), default)))
        if type(atom2) == str:
            if r.has(atom2):
                xyz2[i-1, :] = np.array(r.xyz(atom2))
            else:
                xyz2[i-1, :] = np.array(r.xyz(default))
        else:
            xyz2[i-1, :]  = np.array(r.xyz(atom2.get(r.name(), default)))
    return distance_matrix(xyz1, xyz2)

# Gets sequence given a pose
def get_sequence(pose):
    p = pose
    seq=[aa_mapping[p.residue(i).name1()] for i in range(1,p.size()+1)]
    return np.array(seq)

# Gets secondary structure given a pose
def get_ss(pose):
    fa_scorefxn = get_fa_scorefxn()
    score = fa_scorefxn(pose)
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)
    SS_mat = np.zeros((4, pose.size()))
    for ires in range(1, pose.size()+1):
        SS = pose.secstruct(ires)
        SS_mat[structure_mapping.get(SS, 0), ires-1] = 1
    return SS_mat

# Remove single occurence of E within loop given argmaxed secondary structure arrray
def remove_sheet(x):
    out = x.copy()
    temp = np.zeros(len(x)+2)
    temp[1:-1] = x
    for i in range(len(x)):
        _3mer = temp[i:i+3]
        # Change it to loop
        if _3mer[0] in [2,0] and _3mer[1] == 3 and _3mer[2] in [2,0]:
            out[i] = 2
    return out

def process(args,
            window_size = 64,
            length_min=5,
            length_max=15,
            stride = 5,
            padding = 32,
            verbose = False):
    
    pdb_file, savepath = args
    pose = Pose()
    pose_from_file(pose, pdb_file)
    full_distmap = get_distmap(pose)
    full_seq = get_sequence(pose)
    full_ss = np.argmax(get_ss(pose), axis=0)

    padded_distmap = np.pad(full_distmap, padding, 'constant', constant_values=0)
    padded_seq = np.pad(full_seq, padding, 'constant', constant_values=0)
    padded_ss = np.pad(full_ss, padding, 'constant', constant_values=0)

    if verbose: print(full_distmap.shape, full_seq.shape, full_ss.shape)
    if verbose: print(padded_distmap.shape, padded_seq.shape, padded_ss.shape)

    modified_ss = remove_sheet(full_ss)
    if verbose: print("".join([reverse_structure_mapping[i] for i in full_ss]))

    # FIND ALL LOOP REGIONS
    loops = []
    ss_string = ''.join(modified_ss.astype(int).astype(str))

    #################################################################################
    # This does not give back overlapping regions and we might need to revisit that.
    # See 3ia1A example
    #################################################################################
    pattern = r"[2]+([1]{0,4}|[3]{0,2})[2]+"

    count = 0
    for match in re.finditer(pattern, ss_string):
        loop_size = match.end(0)-match.start(0)
        sequence = modified_ss[match.start(0):match.end(0)]
        # Make sure it passes the length threshold:
        if loop_size >= length_min:
            if verbose: print("RAW:", match.start(0), match.end(0), loop_size, "".join([reverse_structure_mapping[i] for i in sequence]))
            for i in range(0, max(loop_size-length_max, 0)+1, stride):
                loop = sequence[i:i+length_max]
                cur_lsize = len(loop)
                midpoint = match.start(0)+int(cur_lsize/2)+padding
                window_start = midpoint-int(window_size/2)
                window_end = midpoint+int(window_size/2)
                if verbose: print(cur_lsize, midpoint, window_start, window_end)
                loops.append((window_start, cur_lsize))
                count += 1
    loops = np.array(loops)

    if count != 0:
        # SAVE THE SAMPLE
        np.savez(savepath,
                 distmap=padded_distmap,
                 seq=padded_seq,
                 ss=padded_ss,
                 loops=loops
                )
        
if len(sys.argv) != 3:
    print("Usage: script.py <input folder> <output folder>")
    exit(1)

infolder = sys.argv[1]
outfolder = sys.argv[2]
if isdir(infolder) and isdir(outfolder):
    files = [join(infolder,i) for i in listdir(infolder) if ".pdb" == i[-4:]]
    savepaths = [join(outfolder,i[:-4]) for i in listdir(infolder) if ".pdb" == i[-4:]]
    arguments = [(files[i], savepaths[i]) for i in range(len(files))]
    
    import multiprocessing
    pool = multiprocessing.Pool(10)
    out = pool.map(process, arguments)
    
elif isfile(infolder) and isfile(outfolder):
    process(infolder, outfolder)
    