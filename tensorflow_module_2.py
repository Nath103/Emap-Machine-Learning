import tensorflow as tf

#tensorboard --logdir="C:/Anaconda3/envs/Tensorflow_csv/Tutorials/graphs/graph2"
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd #data obtaining

from sklearn.preprocessing import LabelEncoder #to split data into the labelsa nd to apply onehot encoding
from sklearn.utils import shuffle#to ensure all data entering is in random order
from sklearn.model_selection import train_test_split#split to train/test dataset

#Readoing the Dataset
def read_dataset():
    df = pd.read_csv('/home/pi/node_tf/Data/sonar4.csv')
    #print(len(df.columns))
    #print(str(df.columns))
    
    #features
    x = df[df.columns[0:60]].values#gets the values of these readings from sonar datset
    
    #labels
    y = df[df.columns[60]]#class/label
    
    #Encode the dependant variable/the class/label
    encoder = LabelEncoder()
    encoder.fit(y)#fits the label to the end of each relevant featureset (x)
    y = encoder.transform(y)#gives numerical value to each name, so that the label is less than total number of classess so that the label can be easily referanced, for example a Rock will now be transformed to have a 0, label and a bob to have a 1 label value after transformation
    y = one_hot_encode(y)#call other function
    #print(x.shape)#tensorshape
    return(x, y)

    
#Define the encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))#get how many different labels there were, as they are transormed into nmerical fomat
    one_hot_encode = np.zeros((n_labels, n_unique_labels))#create matrix
    one_hot_encode[np.arange(n_labels), labels] = 1
    return (one_hot_encode)


#Define the Model
def multilayer_perceptron(x, weights, biases):
    
    #Hidden layer with RELU activations
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    #Hidden Layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    #Hiddenlayer with Sigmoid function
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    #Layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    #OUtput LAyer
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

def train_command(id1, output, data_length, training_epochs, old_accuracy):
    print("Train Command Activated")
    #print('id1: ' + str(id1))
    X, Y = read_dataset()
    X, Y = shuffle(X, Y, random_state=1)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)
    #print(train_x.shape)
    #print(train_y.shape)
    #print(test_x.shape)
    learning_rate = 0.001
    cost_history = np.empty(shape=[1], dtype=float)
    n_dim = X.shape[1]
    #print("n_dim", n_dim)
    n_class = 4
    model_path = '/home/pi/node_tf/Models/model' + str(id1) + '/' + str(id1+1)
    model_path_1 = '/home/pi/node_tf/Models/model' + str(id1) + '/' + str(id1)
    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60
    x = tf.placeholder(tf.float32, [None, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [None, n_class])

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

    #Initialize all Variables
    saver = tf.train.Saver()#cretae saver object

    y = multilayer_perceptron(x, weights, biases)#call the defined model

    #define function and optimizer
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar("cost", cost_function)
    #training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)


    init = tf.global_variables_initializer()

    '''init      Needs to be after Adam Intializer becuase otherwise variables within th initializer are not initilized and error message created.'''

    sess = tf.Session()
    sess.run(init)

    #calculate the cost and th accuracy for each epoch

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#tf.argmax changes the [array] into an intiger[0,0,1] becomes 2 and on and on
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)


    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/home/pi/node_tf/Graphs/graph4', sess.graph)


    mse_history = []
    accuracy_history = []

    for epoch in range(training_epochs):    
        sess.run(training_step, feed_dict={x: train_x, y_: train_y})
        cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
        cost_history = np.append(cost_history, cost)
        pred_y = sess.run(y, feed_dict={x: test_x})#make a prediction using the model we are trining using data without labels see what the model outputs back/predicts
        mse = tf.reduce_mean(tf.square(pred_y - test_y))#get the mean square error
        mse_ = sess.run(mse)
        mse_history.append(mse_)#keep a log
        accuracy_1 = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
        accuracy_history.append(accuracy_1)
        #User display
        #print('Epoch : ', epoch+1, ' - ', 'Cost: ', cost, ' - Mean Square Error (MSE): ', mse_, ' - Train Accuracy: ', accuracy_1)

        if epoch % 10 == 0:
            train_accuracy, a, b, s = sess.run([accuracy, cost_function, mse, merged_summary], feed_dict={x: train_x, y_: train_y})
            writer.add_summary(s, epoch)
            print('\n')
            print('Epoch : ', epoch+1, ' - ', 'Cost: ', cost, ' - Mean Square Error (MSE): ', mse_, ' - Train Accuracy: ', accuracy_1)

        #if epoch % 50 == 0:
            #print('\n')
            #print('Epoch : ', epoch+1, ' - ', 'Cost: ', cost, ' - Mean Square Error (MSE): ', mse_, ' - Train Accuracy: ', accuracy_1)
            
        if epoch % 100 == 0:
                save_path = saver.save(sess, model_path)
                

    save_path = saver.save(sess, model_path_1)

    

    #Now Runs a test of a large chunck of data to test the true accuracy
    saver.restore(sess, model_path)#restore model from specific diectory

    prediction = tf.argmax(y, 1)#run the NN
    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))#ultimate question how close is it to each one
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_history = []#create list/array
    print("\n")
    print("Running Accuracy Test")
    df1 = pd.read_csv('/home/pi/node_tf/Data/sonar4.csv')#!!!WILL NEED TO CHANGE IN NEAR FUTURE 

    sample_size = 0.5
    print("Number of Rows:" , str(df1.shape[0]))
    print("Sample Size:", str(sample_size))
    print("Number of Test Samples:", str(int(sample_size*df1.shape[0])))
    
    for i in range(0, int(df1.shape[0] * sample_size)):    #what row to test from csv file
        j = random.randint(0, int(df1.shape[0]-1))
        predict_run = sess.run(prediction, feed_dict={x: X[j].reshape(1, 60)})#feed the X values from the csv file row [i]
        accuracy_run = sess.run(accuracy, feed_dict={x: X[j].reshape(1, 60), y_: Y[j].reshape(1, 4)})#once preciction is made decide how close it si to the specified values, run the accuracy function to ditermine if it is correct
        class_array = (sess.run(y_[j][0:], feed_dict={y_: Y}))
        print('\n')
        print("Row Number:", i+1, "of", str(int(df1.shape[0] * sample_size)))
        #print('Label Array: ', class_array)
        tf_class_num = tf.argmax(y_[j], axis=0)

        yi = df1[df1.columns[60]]#class/label
        
        #Encode the dependant variable/the class/label
        encoder = LabelEncoder()
        encoder.fit(yi)#fits the label to the end of each relevant featureset (x)
        class_array1 = encoder.classes_
        #print("Label Names Indexed:", class_array1)

        
        class_label_num = np.argwhere(class_array>0)#get index position of any value which is above 0
        
        class_element = class_array1[class_label_num]
        
        #print("Class Element: ", class_element)
        #print('tf_class_num: ', (sess.run(tf_class_num, feed_dict={y_: Y})))
        
        #print(i, "Original Class: ", (sess.run(y_[i][0:], feed_dict={y_: Y})), "Predicted Values: ", predict_run[0])#writ the results indexed to current itwration
        
        print('Row Number: ', j,'          ', "Original Class: ", (sess.run(tf_class_num, feed_dict={y_: Y})), "          ", "Original Class Label: ", class_element[0][0], "          ", "Predicted Values: ", predict_run[0], "          ", "Predicted Label: ", class_array1[int(predict_run[0])])
        print("Accuracy: ", str(accuracy_run*100) + "%")#print accuracy in "%" Percentage Format
        #print(np.argwhere(class_array>0))
        #print(yi[int(class_label_num)])
        accuracy_history.append(int(accuracy_run))

    mean_accuracy = int((sum(accuracy_history)*100)/len(accuracy_history))
    print("\n")
    print("Old Model Accuracy: " + str(old_accuracy))
    print("New Model Overall Accuracy: ", str(mean_accuracy) + "%")
    

    if mean_accuracy > old_accuracy:
        #print("New model accuarcy of:", str(mean accuarcy) + "%", "is greater than old Model Accuracy of:", str(old_accuarcy) + "%")
        print("New Model Saved Accuracy Improved")
        print('\n')
        print('Model saved in file: {}'.format(model_path_1))
    else:
        #print("New model accuarcy of: " + str(mean accuarcy) + "%" + "is less than old Model Accuracy of: " + str(old_accuarcy) + "%")
        print("New Model Not Saved, accuracy too low")
        mean_accuracy = old_accuracy


    result = [id1, 'TRAINING_COMPLETE', mean_accuracy]
    output.put(result)
