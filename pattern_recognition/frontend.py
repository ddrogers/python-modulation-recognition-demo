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

Users can: 
- View can view an introduction welcome
- Select Data
- Validate and Test Data
- View Network Architecture
- Train Network
- View Confusion Matrices
- View ROC Curves
- Evaluate Network

"""
import os,sys,random
from glob import glob
import pickle as cPickle
import numpy as np
import tkinter as tk
import matplotlib as mpl
import matplotlib as plt
import numpy as np
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import ttk
from PIL import ImageTk,Image 
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
	def to_onehot(yy):
	    yy = list(yy)
	    yy1 = np.zeros([len(yy), max(yy)+1])
	    yy1[np.arange(len(yy)),yy] = 1
	    return yy1
	Y_train = list(map(lambda x: (lbl[x][0],lbl[x][1]), train_idx))
	Y_test = list(map(lambda x: (lbl[x][0], lbl[x][1]), test_idx))
	Y_train_encoding = list(map(lambda x: mods.index(lbl[x][0]), train_idx)) 
	Y_test_encoding = list(map(lambda x: mods.index(lbl[x][0]), test_idx)) 

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

		for F in (StartWindow, SelectData, scatter,IQ):
			frame = F(container, self)
			self.frames[F] = frame
			frame.grid(row=0,column=0,sticky="nsew")
		
		self.show_frame(StartWindow)
	
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()

class StartWindow(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		
		# Create a Main Frame
		mf = tk.Frame(self, width = 500, height = 400)
		mf.grid(row = 0, column = 0, columnspan = 2, padx = 10, pady = 10, sticky = tk.W)

		logo=tk.PhotoImage(file='icons/nn.png')
		w1 = tk.Label(mf, justify = 'center', image = logo)
		w1.image = logo
		w1.grid(row = 0, column = 0, padx = 10, pady = 10, rowspan = 2)
		
		header = """Welcome To The Modulation Recognition Application"""

		explanation = """Learn how to classify radio frequency signals with neural networks."""

		w2 = tk.Label(mf, 
					justify=tk.LEFT,
					padx = 5, pady =1,
					text=header, font='Helvetica 12 bold')
		w2.grid(row = 0, column = 1, sticky = 'W')

		w3 = tk.Label(mf, 
					justify=tk.LEFT, 
					padx = 5,
					text=explanation)
		w3.grid(row = 1, column = 1, sticky = 'W')

		mf2 = tk.Frame(self, width = 500, height = 400)
		mf2.grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 10)
		intro = """
In modulation recognition problems, a neural network should be able to classify
inputs into a set of target categories.

For example, recognizing the modulation that a particular radio signal came
from based on feature analysis.
		
The Modulation Recognition app will help select data, develop and 
train a network, and evaluate the networks pefromance using categorical 
cross-entropy loss and other evaluation metrics.

"""
		nnet = """
A feed forward network, with rectified linear unit (ReLU) hidden and softmax
output neurons, will classify the inphase and quadrature inputs, given the 
right number of neurons in the hidden layers. 

The network will be trained with adaptive gradient (Adagrad) batch gradient 
backwards propagation.
		"""
		lf1 = tk.LabelFrame(mf2, width = 500, height = 400, text = "Introduction", relief = "groove", borderwidth = 2)
		lf1.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = "NSEW")

		m1 = tk.Label(lf1,
		justify = tk.LEFT,
		padx = 5, text = intro)
		m1.grid(row = 0, column = 0, pady = (0,100))

		lf2 = tk.LabelFrame(mf2, width = 500, height = 400, text = "Neural Network",relief = "groove", borderwidth = 2)
		lf2.grid(row = 0, column = 1, padx = 5, pady = 5, sticky = "NSEW")

		narch=tk.PhotoImage(file='icons/nn_arch.png')
		img = tk.Label(lf2, justify = 'center', image = narch)
		img.image = narch
		img.grid(row = 0, column = 0)
		

		m2 = tk.Label(lf2,
		justify = tk.LEFT,
		padx = 5, text = nnet)
		m2.grid(row = 1, column = 0, pady = (0,100))
		
		# Create Main Frame
		mf3 = tk.Frame(self, width = 500, height = 50)
		mf3.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = tk.W)

		# Buttons
		image=tk.PhotoImage(file='icons/right_arrow.png')
		# image label
		il = tk.Button(mf3, image = image, text="To continue, click [Next]", compound = tk.LEFT,relief = tk.FLAT,
		font='Helvetica 12 bold')
		il.image = image
		il.grid(row = 0, column = 0, columnspan = 2, sticky = tk.W)
		
#		il2 = tk.Label(mf3, 
#					justify=tk.LEFT,
#					text="To continue, click [Next]", font='Helvetica 12 bold')
#		il2.grid(row = 0, column = 1, sticky = tk.W)

		image2 = ImageTk.PhotoImage(file="icons/nn_mini.png")
		b1 = tk.Button(mf3, image = image2, text = "Neural Network Start", compound = "left")
		b1.image = image2
		b1.grid(row=1, column = 0, sticky = tk.SW, padx = 5)

		image3 = ImageTk.PhotoImage(file="icons/reverse.png")
		b2 = tk.Button(mf3, image = image3, text = "Welcome", compound = "left", state = "disabled")
		b2.image = image3
		b2.grid(row = 1, column = 1, sticky = tk.SW)

		mf4 = tk.Frame(self)
		mf4.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = tk.SE)

		image4 = ImageTk.PhotoImage(file="icons/left_arrow.png")
		b11 = tk.Button(mf4,image = image4, text="Back", compound = tk.LEFT, state = "disabled")
		b11.image = image4
		b11.grid(row = 1,column = 0, sticky = tk.SE, padx = 5)

		button = tk.Button(mf4, image = image, text = "Next", compound = tk.LEFT,
		command = lambda: controller.show_frame(SelectData))
		button.grid(row = 1, column = 1, sticky = tk.SE, padx = 10)

		image5 = ImageTk.PhotoImage(file="icons/cancel.png")
		button2 = tk.Button(mf4, image = image5, text = "Cancel", compound = tk.LEFT)
		button2.image = image5
		button2.grid(row=1, column = 3, sticky = tk.SE)

class SelectData(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		#label = tk.Label(self, text = "Denoising AutoEncoder: Digital Modulation (Preview)", font = LARGE_FONT)
		#label.pack(pady=10,padx=10)

		# Create a Main Frame
		mf = tk.Frame(self, width = 500, height = 400)
		mf.grid(row = 0, column = 0, columnspan = 2, padx = 10, pady = 10, sticky = tk.W)

		logo=tk.PhotoImage(file='icons/select_data.png')
		w1 = tk.Label(mf, justify = 'center', image = logo)
		w1.image = logo
		w1.grid(row = 0, column = 0, padx = 5, pady = 5, rowspan = 2)
		
		header = """Select Data"""

		explanation = """ Select inputs and targets to define the modulation recognition problem."""

		w2 = tk.Label(mf, 
					justify=tk.LEFT,
					padx = 5, 
					text=header, font='Helvetica 12 bold')
		w2.grid(row = 0, column = 1, sticky = 'W')

		w3 = tk.Label(mf, 
					justify=tk.LEFT, 
					padx = 5,
					text=explanation)
		w3.grid(row = 1, column = 1, sticky = 'W')	
		
		# Create Main Frame
		mf2 = tk.Frame(self, width = 500, height = 400)
		mf2.grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 10)
		#test = """ test test test test test test 
		#test test test test test test test test test"""
		lf1 = tk.LabelFrame(mf2, width = 500, height = 400, text = "Collect Data from Workspace",relief = "groove", borderwidth = 2)
		lf1.grid(row = 0, column = 0, padx = 5, pady = 5)


		m1 = tk.Label(lf1,
		justify = tk.LEFT,
		padx = 5, text = "Select data to present to the network.")
		m1.grid(row = 0, column = 0, columnspan = 3)

		images=tk.PhotoImage(file='icons/right_arrow.png')
		# image label
		m11 = tk.Button(lf1, image = images, text="Inputs: ", compound = tk.LEFT,relief = tk.FLAT)
		m11.image = images
		m11.grid(row = 1, column = 0, columnspan = 2, sticky = tk.E)

		optionList = ['(none)', 'sample 2', 'sample 3']
		
		options = tk.StringVar()
		options.set(optionList[0])
		
		m12 = tk.OptionMenu(lf1, options, *optionList)
		
		m12.grid(row = 1, column  = 2, padx = 5, sticky = tk.E)

		optionChoice = ['...', 'Train Set', 'Test Set']
		
		optionsC = tk.StringVar()
		optionsC.set(optionChoice[0])
		
		m13 = tk.OptionMenu(lf1, optionsC, *optionChoice)
		
		m13.grid(row = 1, column  = 3, sticky = tk.E)

		sep = ttk.Separator(lf1,orient = "horizontal")
		sep.grid(row = 2, column = 0, sticky = "nsew")
		
		#s1 = tk.Frame(lf1, height = 1, width = 250, relief = "groove", bg = "grey")
		#s1.grid(row = 1, column = 0, columnspan = 3)
		
		#m11.grid(row = 2, column = 0)
		
		lf2 = tk.LabelFrame(mf2, width = 500, height = 400, text = "Test Title",relief = "groove", borderwidth = 2)
		lf2.grid(row = 0, column = 1, padx = 5, pady = 5)

		m2 = tk.Message(lf2,
		justify = tk.LEFT,
		padx = 5, text = "test")
		#m2.grid(row = 0, column = 0)
		
		# Create Main Frame
		mf3 = tk.Frame(self, width = 500, height = 50)
		mf3.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = tk.W)

		# Buttons
		image=tk.PhotoImage(file='icons/right_arrow.png')
		# image label
		il = tk.Button(mf3, image = image, text="To continue, click [Next]", compound = tk.LEFT,relief = tk.FLAT,
		font='Helvetica 12 bold')
		il.image = image
		il.grid(row = 0, column = 0, columnspan = 2, sticky = tk.W)
		
#		il2 = tk.Label(mf3, 
#					justify=tk.LEFT,
#					text="To continue, click [Next]", font='Helvetica 12 bold')
#		il2.grid(row = 0, column = 1, sticky = tk.W)

		image2 = ImageTk.PhotoImage(file="icons/nn_mini.png")
		b1 = tk.Button(mf3, image = image2, text = "Neural Network Start", compound = "left")
		b1.image = image2
		b1.grid(row=1, column = 0, sticky = tk.SW, padx = 5)

		image3 = ImageTk.PhotoImage(file="icons/reverse.png")
		b2 = tk.Button(mf3, image = image3, text = "Welcome", compound = "left", state = "disabled")
		b2.image = image3
		b2.grid(row = 1, column = 1, sticky = tk.SW)

		mf4 = tk.Frame(self)
		mf4.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = tk.SE)

		image4 = ImageTk.PhotoImage(file="icons/left_arrow.png")
		b11 = tk.Button(mf4,image = image4, text="Back", compound = tk.LEFT, command = lambda: controller.show_frame(StartWindow))
		b11.image = image4
		b11.grid(row = 1,column = 0, sticky = tk.SE, padx = 5)

		button = tk.Button(mf4, image = image, text = "Next", compound = tk.LEFT,
		command = lambda: controller.show_frame(scatter))
		button.grid(row = 1, column = 1, sticky = tk.SE, padx = 10)

		image5 = ImageTk.PhotoImage(file="icons/cancel.png")
		button2 = tk.Button(mf4, image = image5, text = "Cancel", compound = tk.LEFT)
		button2.image = image5
		button2.grid(row=1, column = 3, sticky = tk.SE)


class scatter(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		label = tk.Label(self, text = "3D Scatter Plot", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
	
		button = tk.Button(self, text = "Back",
		command = lambda: controller.show_frame(SelectData))
		button.pack()
		button2 = tk.Button(self, text = "Next",
		command = lambda: controller.show_frame(IQ))
		button2.pack()

class IQ(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		label = tk.Label(self, text = "I/Q Data", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
	
		button = tk.Button(self, text = "Back",
		command = lambda: controller.show_frame(scatter))
		button.pack()

app = pattern_recognition()
app.mainloop()