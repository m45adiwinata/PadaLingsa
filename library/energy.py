# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 10:25:39 2018

@author: Grenceng
"""
import numpy as np
def root_mean_square(wavedata, block_length, sample_rate):
    
    #JUMLAH BLOK YANG AKAN DIPROSES
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    #WAKTU BLOK TERSEBUT DIMULAI
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))
    
    rms = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))
        rms.append(rms_seg)
    
    return np.asarray(rms), np.asarray(timestamps)