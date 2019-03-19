# NeuralNetwork

This is to implement a neural network in python from scratch with no external dependancies.

Modules used:
  1. math (comes with python)
    for math operations (sqrt, exp)
  2. pickle(comes with python)
    to load and save the model to a .dat file
  3. tqdm(external library)
    for the loading bar. Can modify script to run without it.
  4. tkinter(comes with python)
    for gui to select the file to load
  5. future for print_function
  
Currently supports only one hidden layer. Working on support for multiple layers.

* UPDATE *
Added support for multiple hidden layers! Yay!

The MNIST dataset was used to train this network.
The dataset is in .csv form, taken from https://pjreddie.com/projects/mnist-in-csv/
