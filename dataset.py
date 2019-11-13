# Caleb Ellington
# 11/12/2019

import sys
import numpy as np
from os import listdir
from os.path import isdir, join, abspath
import glob
import random

class Dataset():
    """
    An iterator class to create trainging and test sets from the .npz outputs of loop_parser.py
    INPUTS
    sample_folder: Folder to gather .npz files from
    protein_list: List of protein names to consider in the dataset
    batch_size: Number of samples to return per iteration
    window_size: Sample size
    padding1: Minimum number of residues between the loop and the edge of the window
    padding2: Maxiumum number of residues past the C or N terminus in any window
    pad_with: Value to pad with past the C or N terminus
    transform: Function applied to all distance map values
    """
    def __init__(self, sample_folder, protein_list,
                 batch_size=128, window_size=64, padding1=4, padding2=4, pad_with=0, transform=None):
        if not isdir(sample_folder):
            raise ValueError("sample_folder must be a directory")
        self.batch_size=batch_size
        self.window_size=window_size
        self.padding1=padding1
        self.padding2=padding2
        self.pad_with=pad_with
        self.transform=transform
        self.files = [abspath(join(sample_folder,file)) for file in listdir(sample_folder)
                      if ".npz" == file[-4:] and file[-12:-6] in protein_list]
        if len(self.files) < self.batch_size:
            raise ValueError("Must have more files than batch_size")
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Get a random batch of files
        random.shuffle(self.files)
        ground_truth_batch = []
        mask_batch = []
        ss_batch = []
        count = 0
        file_index = 0
        while count < self.batch_size:
            data = np.load(self.files[file_index])
            file_index += 1
            
            # Gather data
            distmap = data['distmap']
            seq = data['seq']
            ss = data['ss']
            loops = data['loops']
            
            # Apply transform to distmap
            if self.transform is not None:
                distmap = np.array(list(map(self.transform, distmap)))
            
            # Get window
            distmap_padded = np.pad(distmap, self.padding2, 'constant', constant_values=self.pad_with)
            seq_padded = np.pad(seq, self.padding2, 'constant', constant_values=0)
            ss_padded = np.pad(ss, self.padding2, 'constant', constant_values=0)
            loops_padded = loops + self.padding2
            valid_loops_padded = [loop for loop in loops_padded if loop[0]>self.padding1 and loop[1]<len(seq)-self.padding1]
            if not valid_loops_padded:
                continue
            chosen_loop = valid_loops_padded[random.randint(0,len(valid_loops_padded)-1)]
            window_start_min = max(0, chosen_loop[1] + self.padding1 - self.window_size)
            window_end_max = min(len(seq), chosen_loop[0] - self.padding1 + self.window_size)
            if window_start_min < 0 or window_end_max < self.window_size:
                continue
            window_start = random.randint(window_start_min, window_end_max-self.window_size)
            window_end = window_start + self.window_size
            
            # Window data
            distmap_window = distmap_padded[window_start:window_end, window_start:window_end]
            seq_window = seq_padded[window_start:window_end]
            ss_window = ss_padded[window_start:window_end]
            loop_start = chosen_loop[0] - window_start
            loop_end = chosen_loop[1] - window_start
            
            # Create output data
            mask = np.zeros((self.window_size, self.window_size)) + 1
            mask[loop_start:loop_end, :] = 0
            mask[:, loop_start:loop_end] = 0
            
            seq_onehot = np.eye(21)[seq_window]
            seq_onehot_tiled = np.repeat(seq_onehot[np.newaxis, :, :], self.window_size, axis=0)
            seq_transpose = seq_onehot_tiled.transpose(1, 0, 2)
            seq_final = seq_onehot_tiled + seq_transpose
            distmap_final = np.expand_dims(distmap_window, 2)
            ground_truth = np.concatenate((distmap_final, seq_final.astype(float)), axis=2)
            
            ground_truth_batch.append(ground_truth)
            mask_batch.append(mask)
            ss_batch.append(ss_window)
            count += 1
        return (ground_truth_batch, mask_batch, ss_batch)
