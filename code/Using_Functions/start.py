from __future__ import print_function

from os import sys
# Adding previous dircetory since it contains loadData.py
sys.path.insert(0, './..')

import loadData as load
import neuralnetwork as nn

# Main
def main():
    # Loading the training dataset and normalizing it
    print("> Loading the dataset...")

    # Loading the folder
    training_file = load.folder_finder()
    training_dataset = load.convert_csv(training_file, 255)
    print("> Dataset loaded...")

    # Getting the number of neurons in each layer
    
    i_neurons = len(training_dataset[0][0])
    h_neurons = []
    
    ch = int(input("> Enter the number of hidden layers: "))
    for i in range(ch):
        h_neurons.append(int(input("> Enter the number of neurons in the hidden layer {}: ".format(i+1))))

    # For loop to find distinct elements in the output matrix
    distinct_o = []
    for i in training_dataset[1]:
        if i not in distinct_o:
            distinct_o.append(i)
    o_neurons = len(distinct_o)

    # Creating the network
    print("> Initializing the neural network...")
    network = nn.init_network(i_neurons, h_neurons, o_neurons)
    
    # Getting learining rate as input
    l_rate = float(input("> Enter the learning rate of the network: "))

    # Getting number of epoches as input
    epoches = int(input("> Enter the number of epoches to train for: "))

    # Testing network after each epoch?
    testing_dataset = None
    ch = input("> Do you want to test the network after each epoch? (Y/N): ")
    if 'y' in ch.lower():
        # Getting the testing file
        print("> Select the testing dataset...")
        testing_file = load.folder_finder()
        testing_dataset = load.convert_csv(testing_file)
        print("> Testing dataset loaded...")

    # Logging model after each epoch?
    log = True
    ch = input("> Do you want to save the model after each epoch? (Y/N): ")
    if 'n' in ch.lower():
        log = False

    # Training the network
    print("> Training the network...")

    nn.train(network, training_dataset, l_rate, epoches, testing_dataset = testing_dataset)


    print("Exit")

if __name__ =='__main__':
    main()
