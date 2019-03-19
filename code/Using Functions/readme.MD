 This folder contains the following scripts:

    1. loadData.py
        Functions:
        ##########################################################
        def folder_finder_gui()
            Input : 
            ----------------------------------------------------
            this function takes no inputs
            ----------------------------------------------------
            Output: 
            ----------------------------------------------------
            returns folder name
            ----------------------------------------------------
            Purpose:
            ----------------------------------------------------
            Selecting dataset through Tkinter GUI
            ----------------------------------------------------
        ##########################################################
        def convert_csv(fileName, val)
            Input : 
            ----------------------------------------------------
            fileName  : Name of the file being loaded as input 
            val       : Value to divide input by
            ----------------------------------------------------
            Output: 
            ----------------------------------------------------
            returns list containing input and output data
            ----------------------------------------------------
            Purpose:
            ----------------------------------------------------
            Reading the csv to the input and output layers
            ----------------------------------------------------
        ##########################################################
        def beautify_input(arr, c)
            Input : 
            ----------------------------------------------------
            arr  : The input array from convert_csv
            c    : Factor to diminish each value by
            ----------------------------------------------------
            Output: 
            ----------------------------------------------------
            Returns the input array diminished by a factor c
            ----------------------------------------------------
            Purpose:
            ----------------------------------------------------
            Divide each element by factor c
            ----------------------------------------------------
        ##########################################################
        def beautify_output(arr)
            Input : 
            ----------------------------------------------------
            arr  : The output expected for the image
            ----------------------------------------------------
            Output: 
            ----------------------------------------------------
            Returns the output vector
            ----------------------------------------------------
            Purpose:
            ----------------------------------------------------
            convert from 'int' to 'list'
            ----------------------------------------------------
        ##########################################################

    2. neuralnetwork.py
        Functions:
        ------------------------------------------------
        init_network(inputs, hidden, outputs)
            Initializes the network   
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # inputs  : int : number of inputs
            # hidden  : int : number of neurons in hidden layer
            # outputs : int : length of output layer
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # network : list : layers in the network, sans input
            # ---------------------------------------------------- 
        ------------------------------------------------
        dot(weights, bias, inputs)
            Calcuates the dot product and adds bias 
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # weights : list  : contains weights from network
            # bias    : float : contains bias value from network
            # inputs  : list  : contains the input matrix
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # sum : float : value of dot product + bias
            # ----------------------------------------------------
        ------------------------------------------------
        sigmoid(val)
            Calcuates the sigmoid of input
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # val : float : value to calculate sigmoid for
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # x : float : sigmoid of input
            # ----------------------------------------------------  
        ------------------------------------------------
        transfer_derivative(val)
            Calcuates transfer derivative of value
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # val : float : value to calculate derivative for
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # x : float : derivative of input
            # ----------------------------------------------------
        ------------------------------------------------
        forward_propagation(network, inputs)
            Performs forward propagation
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # network : list : the matrix which contains the network
            # inputs  : list : the inputs for the network 
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # output : list : returns the output layer
            # ----------------------------------------------------
        ------------------------------------------------
        backpropagate_error(network, output)
            Calculates backpropagtation error for neuron
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # network : list : the matrix which contains the network
            # output  : list : the expceted output for the network 
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # output : no output, just modifies the network
            # ----------------------------------------------------
        ------------------------------------------------
        update_weights(network, row, l_rate)
            Updates the weights and biases
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # network : list  : the matrix which contains the network
            # row     : list  : the input for the network
            # l_rate  : float : the learning rate for the network
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # output : no output, just modifies the network
            # ----------------------------------------------------
        ------------------------------------------------
        train(network, l_rate, n_epoch, expected)
            Trains the network
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # network  : list  : the matrix which contains the network
            # train    : list  : the input for the network
            # l_rate   : float : the learning rate for the network
            # n_epoch  : int   : number of epochs to train for
            # expected : list  : expected output list
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # output : no output, just modifies the network
            # ----------------------------------------------------    
        ------------------------------------------------
        predict(network, row)
            Inputs the row into the network and predicts
            # ----------------------------------------------------
            # INPUT:
            # ----------------------------------------------------
            # network : list  : the matrix which contains the network
            # row     : list  : the input for the network
            # ----------------------------------------------------
            # OUTPUT: 
            # ----------------------------------------------------
            # output : returns the prediction
            # ----------------------------------------------------
        ------------------------------------------------
    
    3. start.py
        Sample code to implement the network