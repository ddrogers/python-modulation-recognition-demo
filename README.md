Copyright (c) 2018, [Donald D. Rogers II](https://www.linkedin.com/in/donaldrogersii/)

# Modulation Recognition with Convolutional Neural Networks

The following is a demsonstraion of modulation recogntion on digital and analog radio frequency signals using the software-defined data RadioML dataset from [DeepSig Inc.](https://www.deepsig.io/datasets). Model design is deep learning focused using a deep artificial neural network and parameter optimization with k-fold cross-validation. 

The two benchmark scripts established the basis on which the research for digital and analog modulation recognition was built. The modulation_recogntion.py script offers a single class ojbect capable of training either digital or analog radio signals over a wide range of signal-to-noise (SNR) values. Resulting output from trials display the best loss and accuracy values. To visualize this information the following visuals are produced: accuracty curves, confusion matrices, and ROC curves. Running the modulation_recognition.py script also yields storage of the best model(s) from all trials. 

### Signal Types
Digital Modulations: 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK', 'PAM4', 'GFSK'

Analog Modulations: 'WBFM', 'AM-SSB', 'AM-DSB'

### Signal-to-Noise Ratio values
The range of SNR values lies between -20 and 18. 
