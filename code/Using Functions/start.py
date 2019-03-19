from __future__ import print_function
import loadData as load
import neuralnetwork as nn

# Main
def main():
    # Loading the dataset and normalizing it
    print("> Loading the dataset...")
    
    
    # Loading with gui
    file = load.folder_finder_gui()
    dataset = load.convert_csv(file, 255)

    print("> Dataset loaded...")
    
    # Splitting dataset into input and output layers
    input_layer, output_layer = dataset[0], dataset[1]
    
    # Getting the number of neurons in each layer
    
    i_neurons = int(input("> Enter the number of neruons in the input layer: "))
    h_neurons = int(input("> Enter the number of neurons in the hidden layer: "))
    o_neurons = int(input("> Enter the number of neurons in the output layer: "))
    
    # Creating the network
    print("> Initializing the neural network...")
    network = nn.init_network(i_neurons, h_neurons, o_neurons)
    
    # Getting learining rate as input
    l_rate = float(input("> Enter the learning rate of the network: "))

    # Getting number of epoches as input
    epoches = int(input("> Enter the number of epoches to train for: "))
    
    # Training the network
    print("> Training the network...")
    nn.train(network, input_layer, l_rate, epoches, output_layer)

    print("Exit")

if __name__ =='__main__':
    main()