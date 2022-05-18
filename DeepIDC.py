#DATA PROCESSING
def class_text():
    class_files = open(r"drug-drug-set1.txt","r") #READ IDC DATA
    class_file = class_files.readlines()
    rest_file = open(r"rest1.txt","r") #READ UNKNOWN IDC DATA
    rest_f = rest_file.readlines()
    print("class_text is OK")
    return class_file, rest_f

# def all_feature_file(sim_path):
# #     sim_path = r"..\drug\sim_feature" #相似性特征数据文件夹
#     f_files = os.listdir(sim_path) #所有特征数据文件
#     print("all_feature_file is OK")
#     print(f_files)
#     return f_files, sim_path

def data_dimension(sim_path):
    class_file, rest_f = class_text()
    f_files = os.listdir(sim_path)
    
    for each_file in f_files:
        print(each_file)
        data = pd.read_csv(os.path.join(sim_path, each_file),index_col=0) #READ FEATURES
#         print(data)

        drug_data_X = np.zeros(shape=[len(class_file), data.shape[1]*2]) #Matrix to be updated
        drug_data_Y = np.zeros(len(class_file)) #shape=[len(class_file), 5]
#         print(drug_data_X.shape)
        
        drug_data_X_rest = np.zeros(shape=[len(rest_f), data.shape[1]*2]) 
    
        num = 0
        for c in class_file:
            c = c.strip().split("\t") 
            drug_data_Y[num] = int(c[0])-1 #labels change into one-hot
#             print(data.loc[c[1]])
            feature = np.array(pd.concat([data.loc[c[1]], data.loc[c[2]]], ignore_index=True))
            drug_data_X[num, :] = feature #update
            num += 1
        print("class matrix is OK")
        
        num1 = 0
        for c7 in rest_f:
            c7 = c7.strip().split("\t") 
            feature = np.array(pd.concat([data.loc[c7[0]], data.loc[c7[1]]], ignore_index=True))
            drug_data_X_rest[num1, :] = feature #update
            num1 += 1
        print("rest matrix is OK")
#         print(drug_data_X_rest)
        drug_train_X, drug_test_X, drug_train_Y, drug_test_Y = train_test_split(drug_data_X, drug_data_Y, test_size=0.3, random_state=11)
        print(drug_data_X_rest.shape)
        train_and_test(drug_train_X, drug_test_X, drug_train_Y, drug_test_Y, each_file, drug_data_X_rest)
        print(f"I have finished this one -----> {each_file}")

#CHANGE DIMENSIONS INTO LSTM-TYPE
def change_dimension(X, X_test, Y, Y_test,drug_data_X_rest):

    train_matrix_X = np.zeros(shape=[X.shape[0], 2, int(X.shape[1]/2)])
    test_matrix_X = np.zeros(shape=[X_test.shape[0], 2, int(X_test.shape[1]/2)])
    rest_data = np.zeros(shape=[drug_data_X_rest.shape[0], 2, int(drug_data_X_rest.shape[1]/2)])
    
    for i in range(X.shape[0]):
        train_matrix_X[i, :, :] = X[i].reshape(2,int(X.shape[1]/2))   
        
    for s in range(X_test.shape[0]):
        test_matrix_X[s, :, :] = X_test[s].reshape(2,int(X_test.shape[1]/2))
        
    for o in range(drug_data_X_rest.shape[0]):
        rest_data[o, :, :] = drug_data_X_rest[o].reshape(2,int(drug_data_X_rest.shape[1]/2))
#     print(rest_data)
    train_matrix_Y = to_categorical(Y, num_classes=3)
    test_matrix_Y = to_categorical(Y_test, num_classes=3)

    print("change over, shapes are as follow---> ")
    print(train_matrix_X.shape, train_matrix_Y.shape, test_matrix_X.shape, test_matrix_Y.shape, rest_data.shape)
    return train_matrix_X, train_matrix_Y, test_matrix_X, test_matrix_Y, rest_data

def flatten_y(y,p1,yv,p2):
    y_train = np.argmax(y, axis=1).reshape(-1, 1)
    y_train_p = np.argmax(p1, axis=1).reshape(-1, 1)
    y_test1 = np.argmax(yv, axis=1).reshape(-1, 1)
    y_test_p = np.argmax(p2, axis=1).reshape(-1, 1)
    return y_train, y_train_p, y_test1, y_test_p
    
def train_and_test(X, X_test, Y, Y_test, each_file, drug_data_X_rest):

    global batch_size, lr, epoch
    seed = 11
    
    print("I am ready to train! Now I am going to change dimension!")
    x,y,x_test,y_test,rest_data = change_dimension(X, X_test, Y, Y_test,drug_data_X_rest)
    
    print("I WILL SEQUENTIAL!!!GO,GO,GO!!")
    #sequential
    model = keras.Sequential([
        keras.Input(shape=(x.shape[1], x.shape[2])),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True),name="BiLSTM"),#kernel_regularizer=regularizers.l2(0.01), activation='relu'
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu",name='Dense1'),

        layers.Dense(3, activation="softmax")
        ])
    print("sequential is over, the next one is compile")

    #compile
    model.compile(loss= tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()]) #metrics.AUC(name="AUC")
    print("I'm training!! Please wait a minute!! Take it easy!! Next is fit!!")


    #fit
    history = model.fit(x, y, 
                        epochs=epoch, batch_size=batch_size, 
                        verbose=0, shuffle = True)
    
    #weight_Dense_1,bias_Dense_1 = model.get_layer('Dense1').get_weights()
    #print(weight_Dense_1)

    model.summary()
    print(history.history)
    model.save("my_model.h5")
    reconstructed_model=load_model("my_model.h5")
    p_vals11 = reconstructed_model.predict(x_test)
    print(p_vals11[-10:])
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')


    plt.plot(history.history['categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    
    #AUC
    #model.save(f'keras模型\\{each_file[:-4]}1.h5')
    p1 = model.predict(x)
    p2 = model.predict(x_test)
    print(p2[-10:])
    out_test = open(r"test_predict.txt","w")
    np.savetxt(out_test,p2)
    out_test.close()
    p3 = model.predict(rest_data)
    print(type(p3))
    out_rest = open(r"rest_predict.txt","w")
    np.savetxt(out_rest,p3)
    predictclass = np.argmax(p3,axis=1)
    out_class = open(r"class_predict.txt","w")
    np.savetxt(out_class,predictclass)

    out_class.close()
    print(predictclass)
    out_rest.close()
    
    train_auc = metrics.roc_auc_score(y, p1, multi_class='ovr', average=None)
    train_auc_ave = metrics.roc_auc_score(y, p1, multi_class='ovr', average="weighted")
    test_auc = metrics.roc_auc_score(y_test, p2, multi_class='ovr', average=None)
    test_auc_ave = metrics.roc_auc_score(y_test, p2, multi_class='ovr', average="weighted")
    test_aupr_ave = metrics.average_precision_score(y_test, p2,  average="weighted")
    print(f'aupr:{test_aupr_ave}')

    

    print(f"batch_size={batch_size},lr={lr},\t{train_auc}\t{train_auc_ave}\t{test_auc}\t{test_auc_ave}")

    #recall,precision   y_train，y_train_p,
    y_train, y_train_p, y_test1, y_test_p = flatten_y(y,p1,y_test,p2)
    target_names = ['class_1_can', 'class_2_notsure', 'class_3_cannot']
    all_train = metrics.classification_report(y_train, y_train_p, target_names=target_names)
    mcc_test = metrics.matthews_corrcoef(y_test1, y_test_p)
    all_test = metrics.classification_report(y_test1, y_test_p, target_names=target_names)        

    print(all_test)
    print(mcc_test)
    
    out_y = open(r"test_y.txt","w")
    np.savetxt(out_y, y_test1)
    out_y.close()

    presco_train = metrics.precision_score(y_train, y_train_p, average="weighted")
    
    presco_test = metrics.precision_score(y_test1, y_test_p, average="weighted")
    print(f'pr:{presco_test}')
    
    recall_train = metrics.recall_score(y_train, y_train_p, average="weighted")
    recall_test = metrics.recall_score(y_test1, y_test_p, average="weighted")
    print(f'recall:{recall_test}')

    test_f1_ave =metrics.f1_score(y_test1, y_test_p, average='weighted')
    print(f'f1:{test_f1_ave}')

    fpr_out, tpr_out,th = metrics.roc_curve(y_test.ravel(), p2.ravel())
    out_roc_data = open(f"roc//out_roc_data{each_file[:-4]}.txt","w")
    np.savetxt(out_roc_data,(fpr_out, tpr_out,th))

    afpr_out, atpr_out,ath = metrics.precision_recall_curve(y_test.ravel(), p2.ravel())
    out_roc_data1 = open(f"roc//out_aupr_data{each_file[:-4]}.txt","w")
    print(afpr_out)
    np.savetxt(out_roc_data1,(afpr_out, atpr_out),fmt = '%s')


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
import keras.backend as K
from tensorflow.keras.models import load_model
from keras.utils.np_utils import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers.normalization import layer_normalization
import json
import random as rn


#Program repetition
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(11)
rn.seed(11)
tf.random.set_seed(42)
session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
K.set_session(sess)

if __name__ == '__main__':
    lr = 0.0001
    batch_size = 16
    epoch = 100

    sim_path = r"COMPAIR"
    data_dimension(sim_path)