#import serial
import tensorflow as tf
import numpy as np
import multiprocessing as mp

#ser = serial.Serial("/dev/serial0", 9600)

#Define the Model
def multilayer_perceptron(x, weights, biases):
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer
    
def predict_command(input_data, command, id1, output, data_length):
    #print('Predict Command Activated')
    command_pending_message = '+,COMMAND_PENDING,ML TEST,-'
    #ser.write(command_pending_message)
    #print ('Input Data Good, confirmation sent to E-Map: ' + str(command_pending_message))
    #Assigning To data to variables
    tf.reset_default_graph()
    model_name = input_data[2]
    data = input_data[3:63]
    #print(int(len(data)))
    X = np.array(data)
    #print("Command: ", command)
    #print("Model Name: ", model_name)
    n_dim = data_length # used to be X.shape[1] MUST CHANGE BACK?FIND OUT WHAT WENT WRONG
    n_class = 4
    model_path = '/home/pi/node_tf/Models/model' + str(id1) + '/' + str(id1)
    #Neuron Quantity
    n_hidden_1 = data_length
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60
    #Tf PLaceholder defenition (shape)
    x = tf.placeholder(tf.float32, [None, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [None, n_class])
    #Deine the weights and biases for each layer
    weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
            }

    biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_class]))
            }

    saver = tf.train.Saver()#cretae saver object
    y = multilayer_perceptron(x, weights, biases)#call the defined model
    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    sess.run(init)
    saver.restore(sess, model_path)#restore model from specific diectory
    prediction = tf.argmax(y, 1)
    #Run The prediction

    predict_run = sess.run(prediction, feed_dict={x: X.reshape(1, data_length)})
    predicted_label = str(predict_run[0])
    print("\n")
    print('Predicted Label: ' + predicted_label)
    result = [id1, predicted_label]
    output.put(result)
