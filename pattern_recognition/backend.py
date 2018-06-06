"""
A program that acts as the backend for digital_dae_app.

"""
# Load Libraries
import os,sys,random
import numpy as np
import matplotlib.pyplot as plt
import pickle as cPickle
from glob import glob

# Load the dataset
def load_data():
	# Import filepath for data
	data_dir = os.path.join('C:\+CSCoRE','*')
	rfs_path = glob(os.path.join(data_dir,'*','*','*','*','digital.pkl'))
	#  You will need to seperately download or generate this file
	Xd = cPickle.load(open(rfs_path[0],'rb'),encoding="latin1")
	snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
	
	""" Digital """
	digital = []
	lbl = []
	for mod in mods: 
	    for snr in snrs:
	        digital.append(Xd[(mod,snr)])
	        for i in range(Xd[(mod,snr)].shape[0]): lbl.append((mod,snr))
	digital = np.vstack(digital)

	# Partition the data
	#  into training and test sets of the form we can train/test on 
	#  while keeping SNR and Mod labels handy for each
	np.random.seed(2018)
	n_examples = digital.shape[0]
	n_train = int(n_examples * 0.7)
	train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
	test_idx = list(set(range(0,n_examples))-set(train_idx))
	X_train = digital[train_idx]
	X_test =  digital[test_idx]
	def to_onehot(yy):
	    yy = list(yy)
	    yy1 = np.zeros([len(yy), max(yy)+1])
	    yy1[np.arange(len(yy)),yy] = 1
	    return yy1
	Y_train = list(map(lambda x: (lbl[x][0],lbl[x][1]), train_idx))
	Y_test = list(map(lambda x: (lbl[x][0], lbl[x][1]), test_idx))
	Y_train_encoding = list(map(lambda x: mods.index(lbl[x][0]), train_idx)) # change to digital_mods
	Y_test_encoding = list(map(lambda x: mods.index(lbl[x][0]), test_idx)) # change to digital_mods

	return Xd, X_train, Y_train, X_test, Y_test, Y_train_encoding, Y_test_encoding   

# Visualize I/Q Output prior to Autoencoding
def input_vis(mod):
	fig = tk.Figure()
	a = fig.add_subplot(111)
	plt.plot(X_train[mod][0][0])
	plt.plot(X_train[mod][0][1])

# Load AutoEncoder Model
def load_dae(enc_filepath, ae_filepath):
	'''
	'''
	encoder = load_model(enc_filepath)
	autoencoder = load_model(ae_filepath)
	return encoder, autoencoder

# Get Decoded X_training data
def get_pred_train_dae(enc_filepath,ae_filepath, x_train, x_test, batch_size, y_train):
	'''
	Loads in DAE model, Makes predictions using model, Outputs reconstructed data
	'''
	# Predict the Autoencoder output from corrupted test images*
	encoder, autoencoder = load_dae(enc_filepath,ae_filepath)
	x_train = np.reshape(x_train,(len(x_train),2,128,1))
	x_decode = autoencoder.predict(x_train, batch_size= batch_size)
	
	# Create dictionary for keys and decoded values; Reattach Keys to decoded data
	Xd_train_decoded = {}
	for i in range(len(y_train)):
	    Xd_train_decoded[y_train[i]] = x_decode[i]
	# Get encoded data for latent space visualization
	x_test = np.reshape(x_test,(len(x_test),2,128,1))
	x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
	return Xd_train_decoded, x_test_encoded  
	
	


# Visualize Output after Autoencoding
def recon_vis():
	plt.plot(X_train_decoded[mod][0][0])
	plt.plot(X_train_decoded[mod][0][1])
	pass

# Visualize Sparse Data 
def sparse_vis():
	pass