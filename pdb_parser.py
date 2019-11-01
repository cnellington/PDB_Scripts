# Caleb Ellington
# caleb.n.ellington@gmail.com
# 10/31/19

"""
This script gets 64 residue samples for a protein distance map, sequence,
and secondary structure centered around an 8-15 residue loop region.
Usage: python script.py <input_file.pdb> <output_file.npz>
"""

from pyrosetta import *
from scipy.spatial import distance, distance_matrix

import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Usage: script.py <input.pdb> <output.npz>")
    exit(1)

pdb_file = sys.argv[1]
savepath = sys.argv[2]


init(extra_options="-constant_seed -mute all")
pose = Pose()
pose_from_file(pose, pdb_file)
window_size = 64

# GATHER FULL PDB DATA
amino_acids = ['A','V','I','L','M','F','Y','W','S','T','N','Q','C','G','P','R','H','K','D','E']
aa_mapping = {amino_acids[i]:i for i in range(len(amino_acids))}
structure_mapping = {"H":1, "L":2, "E":3}

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
    return seq

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
    
full_distmap = get_distmap(pose)
full_seq = get_sequence(pose)
full_ss = get_ss(pose)

# WINDOW DATA AROUND LOOP REGIONS
distmap_windows = []
seq_windows = []
ss_windows = []
loopsizes = []

import re
ss = full_ss[1]*1 + full_ss[2]*2 + full_ss[3]*3
ss_string = ''.join(ss.astype(int).astype(str))
pattern = r"[2]+[1]{0,4}[2]+"  # Find any sequence of loop residues with a single interrupt of a 0-4 helix residues
valid_loops = []
prev_window = -16
for match in re.finditer(pattern, ss_string):
    loop_size = match.end(0)-match.start(0)
    if loop_size <= 15 and loop_size >=8:
        midpoint = match.start(0)+int(loop_size/2)
        window_start = midpoint-int(window_size/2)
        window_end = midpoint+int(window_size/2)
        if window_start > 32 and window_end < len(ss) and window_start > prev_window+16:
            distmap_window = full_distmap[window_start:window_end, window_start:window_end]
            seq_window = full_seq[window_start:window_end]
            ss_window = ss[window_start:window_end]
            distmap_windows.append(distmap_window)
            seq_windows.append(seq_window)
            ss_windows.append(ss_window)
            loopsizes.append(loop_size)

loopsizes = np.array(loopsizes)
distmap_windows = np.array(distmap_windows)
seq_windows = np.array(seq_windows)
ss_windows = np.array(ss_windows)


# SCALE THE DISTANCE MAP AND TILE THE SEQUENCE
# Scale the dmaps
distmap_windows = np.arcsinh(distmap_windows)
distmap_windows_scaled = np.expand_dims(distmap_windows, 3)

# Onehot encode and tile the sequence, then add residue pairs
seq_windows_onehot = np.eye(20)[seq_windows]
seq_windows_onehot_tiled = np.repeat(seq_windows_onehot[:, np.newaxis, :, :], window_size, axis=1)
seq_windows_onehot_transpose = seq_windows_onehot_tiled.transpose((0, 2, 1, 3))
seq_windows_tiled = seq_windows_onehot_tiled + seq_windows_onehot_transpose

# Concatenate dmap value, onehot sequence, and transposed onehot sequence
distmap_seq_matrix = np.concatenate((distmap_windows_scaled, seq_windows_tiled), axis=3)

# SAVE THE SAMPLES
np.savez(savepath,
         windows=len(loopsizes),
         loop_sizes=loopsizes,
         distmap_seq_windows=distmap_seq_matrix,
         ss_windows=ss_windows
        )
