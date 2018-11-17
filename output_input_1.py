import serial
import time
import numpy as np
import commands
import tensorflow_module_1
import tensorflow_module_2
import multiprocessing as mp


def restart_message():
    #Initiates the conversation with the Emap
    print('E-Map Connected')

    linux_version = commands.getoutput('uname -a')
    sys_up = '+,SYS_UP,' + linux_version[18:28] + ',-'
    ser.write(sys_up)
    print ('This messsage has successfully been sent: ' + sys_up)
    
    time.sleep(0.1)
    
    network_cond = '+,NETWORK_STARTED,-'
    ser.write(network_cond)
    print ('This messsage has successfully been sent: ' + network_cond)
    
    time.sleep(0.1)

    ip_address = commands.getoutput('sudo ifconfig')#get str output from linux shell
    network_up = '+,NETWORK_UP,eth0,' + str(ip_address[72:84]) + ',' + str(ip_address[94:107]) + ',,-'
    ser.write(network_up)
    print ('This messsage has successfully been sent: ' + network_up)

 
def get_input_data():
    message = []
    #print("Waiting for data")
    character = ser.read(1)
    if str(character) > 0:
        message.append(character)
        while str(character) != '-':
            character = ser.read(1)
            #print(character)
            message.append(character)
            #print(str(message))
        return(''.join(message))


def send_answer_with_id(model_name, answer):
    answer_message = '+,COMMAND_RESULT,EVALUATED,' + str(model_name) + ',' + str(answer) + ',-'
    ser.write(answer_message)
    print('\nSent to E-Map with id: ' + str(answer_message))
    
def send_answer(answer):
    answer_message = '+,COMMAND_RESULT,EVALUATED,' + str(answer) + ',-'
    ser.write(answer_message)
    print('\nSent to E-Map: ' + str(answer_message))

    
def command_rejected(num_col, input_length):
    error_message = '+,COMMAND_REJECTED,INPUT_DATA_COLUMNS_EXPECTED: ' + str(num_col) + ' ,INPUT_DATA_COLUMNS_RECIEVED :' + str(input_length) + ',-'
    ser.write(error_message)
    print('Error Message Sent to Emap: "' + str(error_message) + '"')


def serial_open_message():
    if ser.is_open == True:
        print("Serial '" + ser.name + "' is Open")
        restart_message()
    else:
        print("Serial Port is Closed/Not Connected")
        

def busy_message(model_name):
    message = '+,COMMAND_REJECTED,' + str(model_name) + ',MODEL_BUSY' + ',-'
    ser.write(message)
    print('\nSent to E-Map: ' + str(message) + '\n')

def process_state(model_number, model_name):
    if process_state[(int(model_number)-1)] == 0:
        print ("\nProcess started: " + str(process_state))
        print("Model Name: " + str(model_name))
        #print("Model Number: " + str(model_number))
        return 0

    else:
        busy_message(model_name)
        return 1
        

if __name__ == '__main__':
    ser = serial.Serial("/dev/serial0", 9600)  #timeout set to 0, Might cause issuesa s the time for data to be sent may be too long
    ser.flushInput()    #flushing buffers to remove any left over data from previous conversations
    ser.flush()
    ser.flushOutput()
    
    serial_open_message() 
   
    output = mp.Queue()     #multiprocessing buffer
    processes = []
    process_state = []      #Origianlly all staes of the models are set to Null

    
    model_name_dict = {'ML TEST':0, 'ML TEST 2':1, 'ML TEST 3':2, 'ML TEST 4':3}
    model_config = [[60,1,4,20],[60,1,2,20],[60,1,3,20],[60,1,1,20]]# 0=length of inout data, 1=number of expected returned answers, 2, Ml Model Number, 3, model_accuracy
    model_config = np.array(model_config)
    for x in range(len(model_name_dict)):
        process_state.append(0)
    print("Processes: " + str(process_state))
    
    
    while True:
        #print('model_config: ' + model_config)
        iteration = 0
        list_expected_size = 64             #number of expected columns in input data
        if output.empty() == False:         #chechks if the output multiprocess buffer is free, if so then take the first of the buffer and process, get th  answer and the id and then send to the Emap
            result = output.get()
            answer = result[1]
            id1 = result[0]
            #print(id1)
            model_name1 = list(np.where(model_config == id1))
            #print("Model_name[][]", str(model_name1[0][0]))
            model_name1 = model_name_dict.keys()[(int(model_name1[0][0])+1)]
            #print("model_name_dict_index", model_name1)
            send_answer_with_id(model_name1, answer)
            if answer != 'TRAINING_COMPLETE':    #!!!Will NEED TO GET RID OF THE IF STATEMENT WHEN RUSS HAS TAIN INPUT!!!
                process_state[(id1-1)] = 0                
            else: 
                process_state[(id1-2)] = 0  #!!!!TEMPORARY MUJST BE CAHNGED TO -1 !!!!
                new_accuracy = result[2]
                model_config[model_name_dict[model_name]][3] = int(new_accuracy)
            print(model_name, answer)
            print("Processes:" + str(process_state))
        #print("Length of Data in input Buffer: " + str(ser.in_waiting))
        if  not ser.in_waiting == 0:
            input_data = get_input_data()
        else:
            input_data = []
            #print('\nWaiting For Data from E-Map')
        if len(input_data) > 0:
            #print("Input Data: ", str(input_data))
            input_list = input_data.split(',')
            #print("Data Seperated: ", input_list)
            command = input_list[1]
            model_name = input_list[2]
            model_number = model_config[model_name_dict[model_name]][2]
            #print("model_number: " + str(model_number))
            data_length = model_config[model_name_dict[model_name]][0]
            if str(command) == 'EMAP_UP':
                restart_message()
            elif str(command) == 'COMMAND_PREDICT':
                list_expected_size = model_config[model_name_dict[str(model_name)]][0]
                #print('Number of Collumns expected: ' + str(list_expected_size))
                if (int(len(input_list)-4)) == list_expected_size:
                    #print("Correct List Size")
                    #print(process_state[(int(model_number)-1)])
                    if process_state[(int(model_number)-1)] == 0:        #!!!!!TEMPORARY FIX
                        t = mp.Process(target=tensorflow_module_1.predict_command, args=(input_list, command, int(model_number), output, list_expected_size))
                        processes.append(t)
                        t.start()
                        process_state[(int(model_number)-1)] = 1
                        print ("\nProcess started: " + str(process_state))
                        print("Model Name: " + str(model_name))
                        #print("Model Number: " + str(model_number))
                    else:
                        busy_message(model_name)
                else:
                    command_rejected(list_expected_size, (len(input_list)-4))
            if str(command) == 'COMMAND_TRAIN':
                if process_state[(int(model_number)-2)] == 0: #!!!!!MUST CHANGE THIS TO MODEL_NAME_! NOT " THIS IS JUST TO ALLOW IT TO RUN!!!
                    training_epochs = 400   #Temporarily
                    #print('id1: ' + str(model_number))
                    model_accuracy = model_config[model_name_dict[model_name]][3]
                    t = mp.Process(target=tensorflow_module_2.train_command, args=(int(model_number), output, list_expected_size, training_epochs, model_accuracy))
                    processes.append(t)
                    t.start()
                    process_state[(int(model_number)-2)] = 1 #!!!MUST CHANGE TO model_number-1!!!
                    print ("\nProcess started: " + str(process_state))
                    print("Model Name: " + str(model_name))
                else:
                    busy_message(model_name)

            ser.flushInput()
            ser.flush()
            ser.flushOutput()                        
        
    print("E-Map Not Awake/Connected")
    
                    
