# Nick Dresens
# CS 131 Spring 2024
# A6 ANN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

RATE = 0.13       # Learning rate for neural net
ITERATIONS = 500  # Iterations to train the neural net

# Sigmoid activation function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Read and organize training data
def organize_data(filename):
    data = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    data_rows = []
    outcomes = []
    
    setosa = [1.0, 0.0, 0.0]
    versicolor = [0.0, 1.0, 0.0]
    virginica = [0.0, 0.0, 1.0]

    # Arrange data in proper format
    for i in range(len(data)):
        # Input data
        row = []
        row.append(data['sepal_length'].values[i])
        row.append(data['sepal_width'].values[i])
        row.append(data['petal_length'].values[i])
        row.append(data['petal_width'].values[i])
        data_rows.append(row)

        # Output data
        if data['class'].values[i] == 'Iris-setosa':
            outcomes.append(setosa)
        elif data['class'].values[i] == 'Iris-versicolor':
            outcomes.append(versicolor)
        elif data['class'].values[i] == 'Iris-virginica':
            outcomes.append(virginica)

    return data_rows, outcomes

# Initialize base weights for net
def initialize_weights(vertices):
    weights = []

    # Iterate through layers of net
    for i in range(1, len(vertices)):
        layer_weights = []

        # Iterate through neurons in current layer
        for _ in range(vertices[i]):
            neuron_weights = []

            # Iterate through previous layer
            # +1 for the bias neuron
            for _ in range(vertices[i - 1] + 1):
                # Random weight
                neuron_weights.append(np.random.uniform(-1, 1))
            
            layer_weights.append(neuron_weights)

        weights.append(np.matrix(layer_weights))

    return weights

# Forward propegation through the net
def forward_propegation(x, weights, layers):
    acts = [x]  # list to store activation vectors
    vect = x    # temporary vector to store current activation vector
    
    # Iterate through layers of net
    for i in range(layers):
        # Activation vector for current layer
        activation = sigmoid(np.dot(vect, weights[i].T))
        acts.append(activation)

        # Bias neuron for the next layer
        vect = np.append(1, activation)
    return acts

# Backwards propegation through the net
def backward_propegation(x, acts, weights, layers):
    error = np.matrix(x - acts[-1])  # error between actual and predicted output
    
    # Iterate through net in reverse order
    for i in range(layers, 0, -1):
        curr_act = acts[i]

        # Compute activation vector for previous layer
        # Include bias neuron if needed
        if i > 1:
            previous = np.append(1, acts[i - 1])
        else:
            previous = acts[0]
        
        # Compute delta for current layer
        delta = np.multiply(error, (np.multiply(curr_act, 1 - curr_act)))

        # Update weights between current and previous layers
        weights[i - 1] += (RATE * np.multiply(delta.T, previous))

        # Update error through back propegation
        error = np.dot(delta, (np.delete(weights[i - 1], [0], axis=1)))

    return weights

# Train net through adjusting weights
def train(input_set, output_set, weights):
    layers = len(weights)
    num_sets = len(input_set)

    # Iterate through input-output pairs
    for i in range(num_sets):
        input_val = input_set[i]
        output_val = output_set[i]

        # Add bias neuron to input and transform into a matrix
        input_val = np.matrix(np.append(1, input_val))

        # Forward propegation to find activations for each layer
        acts = forward_propegation(input_val, weights, layers)

        # Update weights based on updated activation values
        weights = backward_propegation(output_val, acts, weights, layers)

    return weights

# Train net by adjusting weights many times
def train_the_net(input_train, output_train, vertices):
    weights = initialize_weights(vertices)
    
    # Train net by updating weights for a set number of iterations
    for _ in range(ITERATIONS):
        weights = train(input_train, output_train, weights)
    
    return weights

# Predict plant class based on sepal/petal length/width
def prediction(data, weights):
    data = np.append(1, data)  # bias neuron to input layer
    layers = len(weights)

    # Find activations for each layer
    acts = forward_propegation(data, weights, layers)
    result = acts[-1].A1  # get result vector from last activation vector
    max_act = result[0]   # initialize value
    index = 0             # initialize value

    # Find max activation value
    for i in range(1, len(result)):
        if result[i] > max_act:
            max_act = result[i]
            index = i
   
    x = []
    
    for i in range(len(result)):
        x.append(0)
    for i in range(len(x)):
        if i == index:
            x[i] = 1

    # Calculate certainty
    certainty = round((max_act * 100), 2)
    
    return x, certainty

# Find accuracy of net given a set of data
def set_accuracy(input_set, output_set, weights):
    # Iterate through each set of data
    # Store if the prediction is correct
    right = 0
    for i in range(len(input_set)):
        input_val, output_val = input_set[i], list(output_set[i])
        output, _ = prediction(input_val, weights)
        if (output_val == output):
            right += 1
    
    # Return percentage of correct predictions
    set_accuracy = right / len(input_set)
    return round((set_accuracy * 100), 2)

def get_user_input():
    sepal_length = input("Sepal Length(cm) : ")
    sepal_width  = input("Sepal Width (cm) : ")
    petal_length = input("Petal Length(cm) : ")
    petal_width  = input("Petal Width (cm) : ")

    # Ensure correct user input
    try:
        sepal_length = float(sepal_length)
        sepal_width  = float(sepal_width)
        petal_length = float(petal_length)
        petal_width  = float(petal_width)
    except ValueError:
        print("Invalid input recieved. Quitting Program.")
        quit()

    return [sepal_length, sepal_width, petal_length, petal_width]


if __name__ == '__main__':
    print("\nTraining Model...\n")

    # Read in and organize data into training and test sets
    input_set, output_set = organize_data('data.txt')
    input_train, input_test, output_train, output_test = tts(input_set, output_set, test_size=0.15)
    input_train, input_correct, output_train, output_correct = tts(input_train, output_train, test_size=0.1)

    # Initialize neural net architecture
    num_datapoints = len(input_set[0])
    num_classes = len(output_set[0])
    layers = [num_datapoints, 5, 10, num_classes]

    # Train the net
    weights = train_the_net(input_train, output_train, layers)

    # Get accuracy of the net
    set_acc = set_accuracy(input_set, output_set, weights)
    print(f'SET ACCURACY : {set_acc}%\n')

    # Get and predict class based on user input
    user_data = get_user_input()
    flower, accuracy = prediction(user_data, weights)

    # Determine flower type based on output
    if flower[0] == 1.0:
        flower = 'Iris-setosa'
    elif flower[1] == 1.0:
        flower = 'Iris-versicolor'
    elif flower[2] == 1.0:
        flower = 'Iris-virginica'
    else:
        # Should not happen, but just in case
        print('ERROR')
        quit()

    print(f'\nFLOWER   : {flower}\nACCURACY : {accuracy}%\n')
