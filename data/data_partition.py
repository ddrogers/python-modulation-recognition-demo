"""
The code will do the following:
- Read in RadioML2016.10a data set
- filter and save the data set as a csv based on modulation type
    - a csv for digital signals will be saved
    - a csv for analog signals will be saved
- randomly select samples and save as csv of less than 25 MB

Want a data set with equal distribution of modulation with snr values
BPSK 0
BPSK 2
...
BPSK 18

Parse the main file to separate the data by modulation and snr values

The following python script will: 
- Read in full GNU Radio - RadioML2016.10A file
- Partiion the file by modulation and SNR values
- Store Partitioned data into pkl files
- Read in pickle files
- Output 2 x 128 image of desired modulation with snr
- Output 16 x 16 image of desired modulation with snr
- Combine pickle files into one***


Why?
The current data set is too large to be transfered for storage to other system. 
While efforts could be attempted to place the dataset into a SQL database, the number of 
parameter for each intance in the databse would exceed 256 entry. This isn't necessarily a 
problem. It would just be pretty time consuming for a manual input into a database for tranfer. 
This process would also have to be repeated for any new radio frequency signals. 

Our current options are to partition the data for individual tranfer. When it reaches it's 
intended destination, we can have Python read in the indivdidual files and combine then into one
as needed.

***If needed.
"""
# Import libraries
import os,sys,random
from glob import glob
import pickle as cPickle
import numpy as np
import pandas as pd

# Import filepath for data
data_dir = os.path.join(r'C:\+CSCoRE','*')
rfs_path = glob(os.path.join(data_dir,'*','*.dat'))


'''
Read in RadioML2016.10a Data Set
'''
# Load the dataset
Xd = cPickle.load(open(rfs_path[0],'rb'),encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

# Sample dictionary
analog_mods = ['WBFM', 'AM-SSB','AM-DSB']
digital_mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK','PAM4','GFSK']
digital_mods_psk = ['BPSK', 'QPSK', '8PSK']
digital_mods_fsk = ['CPFSK','GFSK']
digital_mods_am = ['QAM16', 'QAM64','PAM4']

'''
Partition the data by modulation and signal-to-noise ratio values

Store partitioned data into pkl files

Modulations:

Analog: 
- WBFM
- AM-SSB
- AM-DSB

Digital:
- BPSK
- QPSK
- 8PSK
- CPFSK
- GFSK
- QAM16
- QAM64
- PAM4

Signal-to-Noise Ratio Values:
-18 to 18 by parsed by 2

Signal Collection information:
DeepSig Dataset: RadioML 2016.10A

A synthetic dataset, generated with GNU Radio, consisting of 11 modulations. 
This is a variable-SNR dataset with moderate LO drift, light fading, and numerous 
different labeled SNR increments for use in measuring performance across different 
signal and noise power scenarios.

This represents a cleaner and more normalized version of the RadioML2016.04C dataset, which this supersedes.

Files for parsing: 
analog_neg - Analog signals with negative snr values from -18 to -2
analog_zero - Analog signals with snr values of 0
analog_pos - Analog signals with snr values from 2 to 18

digital_neg_psk - Digital PSK signals with negative snr values from -18 to -2
digital_zero_psk - Digtal PSK signals with snr values of 0
digital_pos_psk - Digtal PSK signals with snr values from 2 to 18

digital_neg_fsk - Digital FSK signals with negative snr values from -18 to -2
digital_zero_fsk - Digital FSK signals with snr values of 0
digital_pos_fsk - Digital FSK signals with snr values from 2 to 18

digital_neg_am - Digital QAM and PAM signals with negative snr values from -18 to -2
digital_zero_am - Digital QAM and PAM signals with snr values of 0
digital_pos_am - Digital QAM and PAM signals with snr values from 2 to 18

'''    

def pkl_creator(mods,snrs,Xd,name):
    mod_dict = {}
    for mod in mods:
        for snr in snrs: 
            if (mod,snr) in Xd:
                mod_dict[(mod,snr)] =  Xd[(mod,snr)]
    
    filename = '{}.pkl'.format(name)
    f = open(filename,"wb")
    cPickle.dump(mod_dict,f)
    f.close()

# -20 SNR is available, but here will maintain the range from -18 to 18
pkl_creator(analog_mods,[-18,-16,-14,-12,-10,-8,-6,-4,-2],Xd,'analog_neg')
pkl_creator(analog_mods,[0],Xd,'analog_zero')
pkl_creator(analog_mods,[2,4,6,8,10,12,14,16,18],Xd,'analog_pos')

pkl_creator(digital_mods_psk,[-18,-16,-14,-12,-10,-8,-6,-4,-2],Xd,'digital_neg_psk')
pkl_creator(digital_mods_psk,[0],Xd,'digital_zero_psk')
pkl_creator(digital_mods_psk,[2,4,6,8,10,12,14,16,18],Xd,'digital_pos_psk')

pkl_creator(digital_mods_fsk,[-18,-16,-14,-12,-10,-8,-6,-4,-2],Xd,'digital_neg_fsk')
pkl_creator(digital_mods_fsk,[0],Xd,'digital_zero_fsk')
pkl_creator(digital_mods_fsk,[2,4,6,8,10,12,14,16,18],Xd,'digital_pos_fsk')

pkl_creator(digital_mods_am,[-18,-16,-14,-12,-10,-8,-6,-4,-2],Xd,'digital_neg_am')
pkl_creator(digital_mods_am,[0],Xd,'digital_zero_am')
pkl_creator(digital_mods_am,[2,4,6,8,10,12,14,16,18],Xd,'digital_pos_am')


'''
Read in and output pkl files
'''
def load_pkl(pkl_file):
    '''
    Load pkl file
    '''
    Xd = cPickle.load(open(pkl_file,'rb'),encoding="latin1")
    # Partition dictionary into signal-to-noise ratios and modulations
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    # Initialize list for storage
    X = []  
    lbl = []
    # Store radio signal informaiton (I/Q) data into list named X
    # Store modulations and signal-to-noise ratios into list named lbl
    for mod in mods: 
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]): lbl.append((mod,snr))
    # Stack values in list X
    X = np.vstack(X)
    return Xd, X, lbl, mods, snrs

Xd, X, lbl, mods, snrs = load_pkl("analog_neg.pkl")

'''
View shape of modulation and snrs of Xd
'''
def view_shape(Xd,mods,snrs):
    for mod in mods:
        for snr in snrs:
            print("{} with SNR={}".format(mod,snr))
            print(Xd[(mod,snr)].shape)

# Sample of view_shape        
# view_shape(Xd,mods, snrs)


'''
Output 2 x 128 image of desired modulation with snr
'''
'''
Store indices for a desired modulation
'''
def ind_store(mod,snr):
    # Get indices for all modulations of BPSK with snr = 0
    indices = [i for i, x in enumerate(lbl) if x == (mod,snr)]
    return indices

mod_snr_idx = ind_store('WBFM',-2)
tmp = X[mod_snr_idx]
mod_snr_idx = ind_store('WBFM',-18)
tmp2 = X[mod_snr_idx]

# Output image
import matplotlib.pyplot as plt

def img_out(num,data, data2):
    n = num# how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i])
        plt.title("%s with SNR=%d" % ('WBFM',-2))
        plt.gray() 
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(tmp2[i])
        plt.title("%s with SNR=%d" % ('WBFM',-18))        
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def img_out_color(num,data, data2):
    n = num# how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1) 
        plt.imshow(data[i],cmap = 'hot')
        plt.title("%s with SNR=%d" % ('WBFM',-2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(tmp2[i], cmap='nipy_spectral')
        plt.title("%s with SNR=%d" % ('WBFM',-18))        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Create histogram
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.hist(tmp[i].ravel(), bins=256, range=(-1.0, 1.0), fc='k', ec='k')
    plt.show()

img_out(3,tmp,tmp2)
img_out_color(3,tmp,tmp2)


# Restructure I and Q data to 16 x 16
# Change size of arrays from 2 by 128 to 16 by 16
# Xd = {k: np.asarray([np.resize(v,(16,16))]) for k, v in Xd.items()}
# Change size of arrays from 2 by 128 to 16 by 16
tmp = np.asarray([np.resize(tmp[i],(16,16)) for i in range(len(tmp))])
tmp2 = np.asarray([np.resize(tmp2[i],(16,16)) for i in range(len(tmp2))])

# Data is now ready for analysis!
# Output image
import matplotlib.pyplot as plt

img_out(3,tmp,tmp2)
img_out_color(3,tmp,tmp2)

'''
Create histogram of image data

Histogram Source:
https://matplotlib.org/users/image_tutorial.html
'''
