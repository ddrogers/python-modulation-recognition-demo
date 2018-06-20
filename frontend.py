"""
Neural Network Model Analytics - Frontend

This applicaition is designed to showcase various neural network model designs relating to the task of modulation recognition.

Possible Add-ons: 
Add Student Demo App (Path to student Submissions C:\+CSCoRE\Objective\Submissions)
Add Loss Surface Curve App

This application will: 
- Welcome users to the application.
- Allow users to select an application for use in the "Getting Started" section.
- View additional information about the application in the "More Information" section.

Applications: 
- Modulation Recognition**
	- Users can:

- Auto Encoder**
	- Users can:
		- View 2D I/Q plot of selected signal before encoding
		- View Recontructed 2/D I/Q plot after auto encoding

		- View histogram of selected signal modulation and signal-to-noise ratio, and dimensionality before autoencoding
		- View histogram of selected signal modulation and signal-to-noise ratio, and dimensionality before autoencoding
		- View Image of selected modulation, signal-to-noise ratio, and dimensionality before autoenoding
		- View Image of selected modulation, signal-to-noise ratio, and dimensionality after autoenoding

- Student Demos
	- using student models and the full set of data, users can:
		- train models and sample data
		- view confusion matrices of results
		- view loss curves of results

TO DO (frontend.py - Pattern Recognition):
- Clean Up code 
- Build Select Data Front End
	- Build data selection Import Front End 
- Build Validation and Test Data Front End
- Build Network Architecture Front End
- Build Train Network Front End
	- Build Network Training Screen
- Build Performance Front End (Plotly Interaction)
- Build ROC Curve Front End (Plotly Interaction)
- Attach and Test Back Ends
- Clean up Data
- Upload to BitBucket

TO DO: (frontend.py - AutoEncoder)
- Add Welcome Window
- Add Data Select Window
- Add Auto Encoder Training Section
- Add Auto Encoder Analysis Windows
	- Before Image
	- After Image
	- Before IQ
	- After IQ
	- Before Histogram 
	- After Histogram
	- Users Can: 
		- Select Modulation
		- Select SNR
		- Select amount of information to display

TO DO: (frontend.py - Student Demonstration)
- Add Welcome Window

"""


import os
import tkinter as tk
import numpy as np
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import ttk
from PIL import ImageTk,Image 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import backend

#from pattern_recognition import frontend
LARGE_FONT = ("Times New Roman", 12)

def pattern_recognition_app():
    os.system('python ./pattern_recognition/frontend.py')

def autoencoder_app():
    os.system('python ./autoencoder/frontend.py')

class NNA(tk.Tk):
	def __init__(self, *args, **kwargs):

		tk.Tk.__init__(self, *args, **kwargs)

		tk.Tk.iconbitmap(self, default = "icons/placeholder.ico")
		tk.Tk.wm_title(self, "Neural Network Model Analysis")

		container = tk.Frame(self)
		container.pack(side = "top", fill = "both", expand = "True")
		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}

		for F in (StartWindow, page2, page3):

			frame = F(container, self)

			self.frames[F] = frame

			frame.grid(row=0,column = 0, sticky = "nsew")
		
		self.show_frame(StartWindow)

	def show_frame(self, cont):

		frame = self.frames[cont]
		frame.tkraise()

class StartWindow(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		
		logo=tk.PhotoImage(file='icons/nn.png')
		w1 = tk.Label(self, justify = 'center', image = logo)
		w1.image = logo
		w1.grid(row = 0, column = 0, padx = 10, pady = 10, rowspan = 2)
		
		header = """Welcome To The Neural Network Application"""

		w2 = tk.Label(self, 
					justify=tk.LEFT,
					padx = 5, pady =1,
					text=header, font='Helvetica 12 bold')
		w2.grid(row = 0, column = 1, sticky = 'W')


		explanation = """Learn how to classify radio frequency signals with neural networks."""

		w3 = tk.Label(self, 
					justify=tk.LEFT, 
					padx = 5,
					text=explanation)
		w3.grid(row = 1, column = 1, sticky = 'W')

		tab_control = ttk.Notebook(self)
		tab_control.grid(row = 2,column = 0, columnspan = 2, padx = 10, pady = 10,sticky = 'NESW')
		t1 = ttk.Frame(tab_control)
		t2 = ttk.Frame(tab_control)
		tab_control.add(t1, text = "Getting Started")
		tab_control.add(t2, text = "More Information")
		start_exp = """
Each of these applications will help to solve a different kind of problem. The last of panel of
each wizard generates a Python script for solving the same or similar problems. 
Example datasets are provided if no user provided data is available.

Currently, only the Data Analysis app button is operational. 
Clicking it will open a draft of a data analysis section of the code.

"""
		l1 = tk.Label(t1, 
			justify = tk.LEFT,
			text = start_exp)
		l1.grid(row=0, column=0, padx = 10, pady = 10, columnspan = 2, sticky = "NSEW")

		l12 = tk.Label(t1,
			justify = tk.LEFT,
			text = "Modulation recognition and classification:")
		l12.grid(row=1, column = 0, padx = 10, sticky = 'W')
		l13 = tk.Label(t1,
			justify = tk.LEFT,
			text = "Auto Encoder:")
		l13.grid(row=2, column = 0, padx = 10,sticky = 'W')
		l14 = tk.Label(t1,
			justify = tk.LEFT,
			text = "Student Demos:")
		l14.grid(row=3, column = 0, padx = 10,sticky = 'W')

		l21 = tk.Label(t2, text = "More information to come.")
		l21.grid(row=0,column=0)

		image = ImageTk.PhotoImage(file="icons/nn_mini.png")
		b12 = ttk.Button(t1,
			image = image, text = "Modulation Recognition App", compound = "left", command = lambda : pattern_recognition_app())
		b12.image = image
		b12.grid(row = 1, column = 1, sticky = 'W')
		b13 = ttk.Button(t1,
			image = image, text = "Auto Encoder App", compound = "left", command = lambda : autoencoder_app())
		b13.grid(row = 2, column = 1, sticky = 'W')
		b14 = ttk.Button(t1,
			image = image, text = "Student Demo App", compound = "left")
		b14.grid(row = 3, column = 1, sticky = 'W')

class page2(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text = "Page 1", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
        
		button = ttk.Button(self, text="Page 2",
		                command = lambda: controller.show_frame(page2))
		button.pack()

class page3(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text = "Page 1", font = LARGE_FONT)
		label.pack(pady=10,padx=10)
        
		button = ttk.Button(self, text="Page 2",
		                command = lambda: controller.show_frame(page2))
		button.pack()

app = NNA()
app.mainloop()