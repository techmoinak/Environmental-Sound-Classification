

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:14:01 2020

@author: Sayma
"""
import glob
import os
import librosa
import numpy as np
import librosa.display
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import pickle

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(30,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()
    
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(30,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(30,1,i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f))**2)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()
    
sound_file_paths=[ "3-130998-A-28.wav", "3-132340-A-37.wav","5-237499-A-4.wav"
                 ,"5-220955-A-40.wav", 
                 "5-215172-A-13.wav"]
sound_names=[ "snoring","clock_alarm","frog","helicopter",  
             "cricket"]
#raw_sounds = load_sound_files(sound_file_paths)

#plot_waves(sound_names,raw_sounds)
#plot_specgram(sound_names, raw_sounds)

#plot_log_power_specgram(sound_names,raw_sounds)

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('\\')[2].split('.')[0].split('-')[3])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    #print (one_hot_encode)
    return one_hot_encode

parent_dir = 'Sound-Data-50'




"""tr_features_1, tr_labels_1 = parse_audio_files(parent_dir,['fold1'])
tr_features_2, tr_labels_2 = parse_audio_files(parent_dir,['fold2'])
tr_features_3, tr_labels_3 = parse_audio_files(parent_dir,['fold3'])
tr_features_4, tr_labels_4 = parse_audio_files(parent_dir,['fold4'])
tr_features_5, tr_labels_5 = parse_audio_files(parent_dir,['fold5'])
#ts_features_1, ts_labels_1 = parse_audio_files(parent_dir,'fold2')

tr_labels_1 = one_hot_encode(tr_labels_1)
tr_labels_2 = one_hot_encode(tr_labels_2)
tr_labels_3 = one_hot_encode(tr_labels_3)
tr_labels_4 = one_hot_encode(tr_labels_4)
tr_labels_5 = one_hot_encode(tr_labels_5)"""
#ts_labels = one_hot_encode(ts_labels)"""




tr_features_1_file = "tr_features_1.txt"
tr_labels_1_file = "tr_labels_1.txt"
tr_features_2_file = "tr_features_2.txt"
tr_labels_2_file = "tr_labels_2.txt"
tr_features_3_file = "tr_features_3.txt"
tr_labels_3_file = "tr_labels_3.txt"
tr_features_4_file = "tr_features_4.txt"
tr_labels_4_file = "tr_labels_4.txt"
tr_features_5_file = "tr_features_5.txt"
tr_labels_5_file = "tr_labels_5.txt"



"""with open(tr_features_1_file, "wb") as file:
        pickle.dump(tr_features_1, file)
with open(tr_labels_1_file, "wb") as file:
        pickle.dump(tr_labels_1, file)
with open(tr_features_2_file, "wb") as file:
        pickle.dump(tr_features_2, file)
with open(tr_labels_2_file, "wb") as file:
        pickle.dump(tr_labels_2, file)
with open(tr_features_3_file, "wb") as file:
        pickle.dump(tr_features_3, file)
with open(tr_labels_3_file, "wb") as file:
        pickle.dump(tr_labels_3, file)
with open(tr_features_4_file, "wb") as file:
        pickle.dump(tr_features_4, file)
with open(tr_labels_4_file, "wb") as file:
        pickle.dump(tr_labels_4, file)
with open(tr_features_5_file, "wb") as file:
        pickle.dump(tr_features_5, file)
with open(tr_labels_5_file, "wb") as file:
        pickle.dump(tr_labels_5, file)"""






with open(tr_features_1_file, "rb") as file:
        tr_features_1 = pickle.load(file)
        #tr_features = pickle.load(file)
with open(tr_labels_1_file, "rb") as file:
        tr_labels_1 = pickle.load(file)
        #tr_labels = pickle.load(file)
with open(tr_features_2_file, "rb") as file:
        tr_features_2 = pickle.load(file)
        #ts_features = pickle.load(file)
with open(tr_labels_2_file, "rb") as file:
        tr_labels_2 = pickle.load(file)
        #ts_labels = pickle.load(file)
with open(tr_features_3_file, "rb") as file:
        tr_features_3 = pickle.load(file)
with open(tr_labels_3_file, "rb") as file:
        tr_labels_3 = pickle.load(file)
with open(tr_features_4_file, "rb") as file:
        tr_features_4 = pickle.load(file)
with open(tr_labels_4_file, "rb") as file:
        tr_labels_4 = pickle.load(file)
with open(tr_features_5_file, "rb") as file:
        tr_features_5 = pickle.load(file)
with open(tr_labels_5_file, "rb") as file:
        tr_labels_5 = pickle.load(file)
        







acc_array=[]
training_epochs = 6000
n_dim = tr_features_1.shape[1]
n_classes = 50
n_hidden_units_one = 400
n_hidden_units_two = 450
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.00001

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two],mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two],mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)


y_pred_=[]
y_true_=[]

confusion_matrix=[ [0 for i in range(50) ] for j in range(50)]

def run_NN(i, tr_features, tr_labels,  ts_features, ts_labels):
    


    init = tf.initialize_all_variables()
    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\n #####fold#########", i , "\n")
    cost_history = np.empty(shape=[1],dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
            cost_history = np.append(cost_history,cost)

        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_pred_ = np.array(y_pred)
        y_true = sess.run(tf.argmax(ts_labels,1))
        y_true_ = np.array(y_true)
        
        print("just for testing: ", y_pred_.item(0), y_true_.item(0))
        print("y_pred : ", y_pred, " y_true: " , y_true)
        acc = round(sess.run(accuracy,feed_dict={X: ts_features,Y: ts_labels}),3)
        print("Test accuracy: ", acc)
        acc_array.append(acc)
        for i in range(0, len(y_pred_)):
            confusion_matrix[y_pred_.item(i)][y_true_.item(i)] = 1 + confusion_matrix[y_pred_.item(i)][y_true_.item(i)]
        
        #for i in range(0, len(confusion_matrix)):
         #   print(confusion_matrix[i], "\n")
         
        print(confusion_matrix) 
         
    fig = plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()

    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    print ("F-Score:", f, " p: ", p , " r: ", r, " s: ", s)

####################1#################################

run_NN(1,np.append(np.append(np.append(tr_features_1, tr_features_2, axis = 0),
                              tr_features_3, axis = 0), tr_features_4, axis = 0), 
       np.append(np.append(np.append(tr_labels_1, tr_labels_2, axis = 0),
                              tr_labels_3, axis = 0), tr_labels_4, axis = 0), tr_features_5, tr_labels_5)





####################2#################################
run_NN(2, np.append(np.append(np.append(tr_features_1, tr_features_2, axis = 0),
                              tr_features_3, axis = 0), tr_features_5, axis = 0), 
       np.append(np.append(np.append(tr_labels_1, tr_labels_2, axis = 0),
                              tr_labels_3, axis = 0), tr_labels_5, axis = 0), tr_features_4, tr_labels_4)


####################3#################################
run_NN(3, np.append(np.append(np.append(tr_features_1, tr_features_2, axis = 0),
                              tr_features_5, axis = 0), tr_features_4, axis = 0), 
       np.append(np.append(np.append(tr_labels_1, tr_labels_2, axis = 0),
                              tr_labels_5, axis = 0), tr_labels_4, axis = 0), tr_features_3, tr_labels_3)


####################4#################################
run_NN(4, np.append(np.append(np.append(tr_features_1, tr_features_5, axis = 0),
                              tr_features_3, axis = 0), tr_features_4, axis = 0), 
       np.append(np.append(np.append(tr_labels_1, tr_labels_5, axis = 0),
                              tr_labels_3, axis = 0), tr_labels_4, axis = 0), tr_features_2, tr_labels_2)




####################5#################################
run_NN(5, np.append(np.append(np.append(tr_features_5, tr_features_2, axis = 0),
                              tr_features_3, axis = 0), tr_features_4, axis = 0), 
       np.append(np.append(np.append(tr_labels_5, tr_labels_2, axis = 0),
                              tr_labels_3, axis = 0), tr_labels_4, axis = 0), tr_features_1, tr_labels_1)



    
print("mean: ", np.mean(acc_array))
#labels = one_hot_encode(labels)

#labels = one_hot_encode(labels)'''

