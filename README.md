# LSTM-KTH
A recurrent deep neural network for human activity recognition, using the KTH dataset as an example.

## prerequisites
This code contains OS commands for linux. It also uses keras with tensorflow, and ffmpeg. I have not tested the code on Python 3.

## Instructions
After downloading the KTH dataset:

1. Edit the path to the dataset in both KTH_prepare.py and KTH_LSTM.py
2. Run KTH_prepare.py to extract frames for the relevant segments of the dataset.
3. Try running KTH_LSTM.py to train the network. You might want to change the parameters like the amount of data used for training and testing.
4. Modify the network architecture and hyper parameters to improve accuracy.
