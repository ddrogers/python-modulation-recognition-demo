"""
Pattern Recognition Front End
A program to perform pattern recogntion for digital radio frequency signals.
Modulations include:
- 'BPSK'
- 'QPSK'
- '8PSK'
- 'QAM16'
- 'QAM64'
- 'CPFSK'
- 'PAM4'
- 'GFSK'
Analog:
- 'WBFM'
- 'AM-SSB'
- 'AM-DSB'
User can*: 
Select Modulation
View input, reconstructed, and sparse plots of autoencoding

The application will:
- Allow users to select a modulation 
	- digital
		- BPSK
		- QPSK
		- 8PSK
		- QAM16
		- QAM64
		- PAM4
		- CPFSK
		- GFSK 
	- analog
		- WBFM
		- AM-SSB
		- AM-DSB
- Allow users to select a signal-to-noise ratio value
	- -18 to 18 listed by 2's 
- Allow users to select the dimensionality of their data
	- 2 x 128
	- 16 x 16

Users can: 
- View 2D I/Q plot of selected signal before encoding
- View Recontructed 2/D I/Q plot after auto encoding

- View histogram of selected signal modulation and signal-to-noise ratio, and dimensionality before autoencoding
- View histogram of selected signal modulation and signal-to-noise ratio, and dimensionality before autoencoding
- View Image of selected modulation, signal-to-noise ratio, and dimensionality before autoenoding
- View Image of selected modulation, signal-to-noise ratio, and dimensionality after autoenoding
"""
import os,sys,random
from glob import glob
import pickle as cPickle
import numpy as np
import tkinter as tk
import matplotlib as mpl
import numpy as np
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import tkinter as ttk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

""""""
# Load the dataset
def load_data():
	# Import filepath for data
	#data_dir = os.path.join('C:\+CSCoRE','*')
	#rfs_path = glob(os.path.join(data_dir,'*','*','*','*','digital.pkl'))
	#  You will need to seperately download or generate this file
	Xd = cPickle.load(open("data/digital.pkl",'rb'),encoding="latin1")
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
	
	Y_train = list(map(lambda x: (lbl[x][0],lbl[x][1]), train_idx))
	Y_test = list(map(lambda x: (lbl[x][0], lbl[x][1]), test_idx))
	Y_train_encoding = list(map(lambda x: mods.index(lbl[x][0]), train_idx)) 
	Y_test_encoding = list(map(lambda x: mods.index(lbl[x][0]), test_idx)) 

	return Xd, X_train, Y_train, X_test, Y_test, Y_train_encoding, Y_test_encoding   

# Visualize I/Q Output prior to Autoencoding
def input_vis(mod):
	fig = tk.Figure()
	fig.add_subplot(111)
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
def recon_vis(mod,X_train_decoded):
	plt.plot(X_train_decoded[mod][0][0])
	plt.plot(X_train_decoded[mod][0][1])
	pass

# Visualize Sparse Data 
def sparse_vis():
	pass
""""""

LARGE_FONT = ("Verdana",12)

class pattern_recognition(tk.Tk):
	def __init__(self, *args, **kwargs):

		tk.Tk.__init__(self, *args, **kwargs)

		tk.Tk.iconbitmap(self, default = "C:/Users/donal/Pictures/test.ico")
		tk.Tk.wm_title(self, "Pattern Recognition")

		container = tk.Frame(self)
		container.pack(side = "top", fill="both", expand = "True")
		container.grid_rowconfigure(0,weight = 1)
		container.grid_columnconfigure(0,weight = 1)

		self.frames = {}

		for F in (welcome, welcome2, scatter,IQ):
			frame = F(container, self)
			self.frames[F] = frame
			frame.grid(row=0,column=0,sticky="nsew")
		
		self.show_frame(welcome)
	
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()

class welcome(tk.Frame): # to be renamed to visuals
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		#label = tk.Label(self, text = "Denoising AutoEncoder: Digital Modulation (Preview)", font = LARGE_FONT)
		#label.pack(pady=10,padx=10)

		# Labels
		l1 = tk.Label(self, text="Modulation: ")
		l1.grid(row=0, column = 0, sticky = tk.W)
		l2 = tk.Label(self, text="Signal-to-Noise Ratio (SNR):")
		l2.grid(row=1, column = 0, sticky = tk.W)
		l3 = tk.Label(self, text="Input (I/Q Data)")
		l3.grid(row=0, column = 2)
		l4 = tk.Label(self, text="Reconstruction (I/Q Data)")
		l4.grid(row=0, column = 3)
		l5 = tk.Label(self, text="Input (16 x 16)")
		l5.grid(row=2, column = 2)
		l6 = tk.Label(self, text="Reconstruction (16 x 16)")
		l6.grid(row=2, column = 3)


		# Menus
		digital_mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK','PAM4','GFSK']

		mods = tk.StringVar()
		mods.set(digital_mods[0]) # Inializes


		w = tk.OptionMenu(self, mods, *digital_mods)
		w.grid(row = 0, column= 1, sticky = tk.W)


		# Scale
		s = tk.Scale(self, from_=-20, to_=20, resolution = 2, orient=tk.VERTICAL)
		s.grid(row =1, column = 1, sticky = tk.W)

		# Visuals
		xd, _, _, _, _, _, _ = load_data()
    
		# Initialize Canvases
		c1 = tk.Canvas(self, width = 300, height = 200)
		c1.grid(row = 1, column = 2, padx = 5)
		c2 = tk.Canvas(self, width = 300, height = 200)
		c2.grid(row = 1, column = 3, padx = 5)
		c3 = tk.Canvas(self, width = 300, height = 200)
		c3.grid(row = 3, column = 2, padx = 5)
		c4 = tk.Canvas(self, width = 300, height = 200)
		c4.grid(row = 3, column = 3, padx = 5)
		

		def get_mod_snr():
			mod = mods.get()
			snr = s.get()
			return (mod, snr)

		#enc_filepath = 'cnn_1.0.h5'
		#ae_filepath = 'cnn_1.1.h5' 

		def input_plot ():
			mod_snr = get_mod_snr()
			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0][0])
			a.plot(xd[mod_snr][0][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=1, column=2, padx = 5)
			canvas.draw()

			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0])
			a.plot(xd[mod_snr][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=1, column=3, padx = 5)
			canvas.draw()

			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0][0])
			a.plot(xd[mod_snr][0][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=3, column=2, padx = 5)
			canvas.draw()

			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0])
			a.plot(xd[mod_snr][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=3, column=3, padx = 5)
			canvas.draw()


		# Buttons
		b1 = tk.Button(self,text="Encode", command = input_plot)
		b1.grid(row = 4,column = 3, sticky = tk.E, padx = 5)

		button = ttk.Button(self, text = "Next",
		command = lambda: controller.show_frame(welcome2))
		button.grid(row = 4, column = 4, sticky = tk.E, padx = 5)

class welcome2(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		#label = tk.Label(self, text = "Denoising AutoEncoder: Digital Modulation (Preview)", font = LARGE_FONT)
		#label.pack(pady=10,padx=10)

		# Labels
		l1 = tk.Label(self, text="Modulation: ")
		l1.grid(row=0, column = 0, sticky = tk.W)
		l2 = tk.Label(self, text="Signal-to-Noise Ratio (SNR):")
		l2.grid(row=1, column = 0, sticky = tk.W)
		l3 = tk.Label(self, text="Input")
		l3.grid(row=0, column = 2)
		l4 = tk.Label(self, text="Reconstruction")
		l4.grid(row=0, column = 3)
		l5 = tk.Label(self, text="Latent Space: AE Clustering (For Selected SNR)")
		l5.grid(row=2, column = 2, columnspan = 2)

		# Menus
		digital_mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK','PAM4','GFSK']

		mods = tk.StringVar()
		mods.set(digital_mods[0]) # Inializes


		w = tk.OptionMenu(self, mods, *digital_mods)
		w.grid(row = 0, column= 1, sticky = tk.W)


		# Scale
		s = tk.Scale(self, from_=-20, to_=20, resolution = 2, orient=tk.VERTICAL)
		s.grid(row =1, column = 1, sticky = tk.W)

		# Visuals
		xd, x_train, y_train, x_test, y_test, y_train_encoding, y_test_encoding = load_data()

		# Initialize Canvases
		c1 = tk.Canvas(self, width = 300, height = 200)
		c1.grid(row = 1, column = 2, padx = 5)
		c2 = tk.Canvas(self, width = 300, height = 200)
		c2.grid(row = 1, column = 3, padx = 5)
		c3 = tk.Canvas(self, width = 300, height = 200)
		c3.grid(row=3, column=2, columnspan = 2, stick = tk.W + tk.E + tk.N + tk.S,
				padx = 5, pady = 5)


		def get_mod_snr():
			mod = mods.get()
			snr = s.get()
			return (mod, snr)

		enc_filepath = 'cnn_1.0.h5'
		ae_filepath = 'cnn_1.1.h5' 

		def input_plot ():
			mod_snr = get_mod_snr()
			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0][0])
			a.plot(xd[mod_snr][0][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=1, column=2, padx = 5)
			canvas.draw()

			fig = Figure(figsize=(5,3))
			a = fig.add_subplot(111)
			a.plot(xd[mod_snr][0])
			a.plot(xd[mod_snr][1])
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=1, column=3, padx = 5)
			canvas.draw()

			fig = Figure(figsize=(3,3))
			num_bins = 50
			a = fig.add_subplot(111)
			
			n,bins,patches = a.hist(xd[mod_snr][0][0],num_bins,density = 1)
			a.set_xlabel('Inphase')
			a.set_ylabel('# of Occurences')
			canvas = FigureCanvasTkAgg(fig, master=self)
			canvas.get_tk_widget().grid(row=3, column=2, columnspan = 2, stick = tk.W + tk.E + tk.N + tk.S,
				padx = 5, pady = 5)
			canvas.draw()

		# Buttons
		b1 = tk.Button(self,text="Encode", command = input_plot)
		b1.grid(row = 4,column = 3, sticky = tk.E, padx = 5)

		button = ttk.Button(self, text = "Next",
		command = lambda: controller.show_frame(scatter))
		button.grid(row = 4, column = 4, sticky = tk.E, padx = 5)


class scatter(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		label = tk.Label(self, text = "3D Scatter Plot", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
	
		button = ttk.Button(self, text = "Back",
		command = lambda: controller.show_frame(welcome))
		button.pack()
		button2 = ttk.Button(self, text = "Next",
		command = lambda: controller.show_frame(IQ))
		button2.pack()

class IQ(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		label = tk.Label(self, text = "I/Q Data", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
	
		button = ttk.Button(self, text = "Back",
		command = lambda: controller.show_frame(scatter))
		button.pack()

app = pattern_recognition()
app.mainloop()