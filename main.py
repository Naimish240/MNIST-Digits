# Simple neural network program in python
# to train on MNIST digit dataset
# written without any external dependancies

# Function to get data from csv and return a dictionary
def trainingData():
    input_data = []
    output_data = []
    
    # Opening the csv file
    with open('mnist_train.csv', 'r') as fh:
        lines = fh.readlines()
        
        for i in lines:
            i = i.strip('\n')                    # Removing new line character
            i = i.split(',')                     # Splitting with comma
            
            output_data.append(int(i[0]))        # Adding to output matrix
            input_data.append(i[1:])             # Adding to input matrix, all values are strings
            
    input_data = beautify(input_data)
    # Returns the input and output matricies
    return [input_data, output_data]

def beautify(arr):
    a = []
    
    for i in arr:
        b = []
        for j in i:
            b.append(int(j)/255)
        a.append(b)
        
    return a

if __name__ == '__main__':
    # Currently takes 13 seconds
    a = trainingData()
    
    t = 0

    for i in a[0]:
        if len(i) == 784:
            t+=1
        else:
            print(len(i))
    
    print(t)
    
            
    
            
