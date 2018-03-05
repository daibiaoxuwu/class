#encoding:utf-8
#n51.py 
#learning rate decay
#patchlength 0 readfrom resp
#add:saving session
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import word2vec
import os
import json
import re
import requests
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"
embedding_size=100
patchlength=0

maxlength=700
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.02
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=100,decay_rate=0.9)
training_iters = 1000000
training_steps=150
display_step = 20

# number of units in RNN cell
n_hidden = 512


print('init:0')
start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = 'log/rnn_words'
writer = tf.summary.FileWriter(logs_path)

model=word2vec.load('train/combine100.bin')
# Text file containing words for training
training_path = r'train/resp'

saver=tf.train.Saver(max_to_keep=1)
max_acc=0



with open(training_path) as f:
    resp=f.readlines()
print(len(resp))

#len:2071700
print('init:1')
'''
def lemma(verb):
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
    resp = requests.post(url, verb, params=params).text
    content=json.loads(resp)
    return content['sentences'][0]['tokens'][0]['lemma']
'''
with open('train/lemma', 'rb') as f:
    ldict = pickle.load(f)

def lemma(verb):
    if verb in ldict:
        return ldict[verb]
    else:
        print('errverb:',verb)
        return verb

def list_tags(st,step):
    realength=0
    tagdict={')':0}
    inputs=[]
    pads=[]
    answer=[]
    count=st
    #fft=0
    for sentence in resp[st:]:#一个sentence是一句话
        if len(answer)==step:
            break
        outword=[]
        count+=1
        total=0
        
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        if total!=1:
            continue
        #else:
            #fft+=1
        
        for oldsentence in resp[count-patchlength:count]:
            
        
            for tag in sentence.split():
                if tag[0]=='(':
                    if tag not in tagdict:
                        tagdict[tag]=len(tagdict)
                    tagword=[0]*embedding_size
                    tagword[tagdict[tag]]=1
                    outword.append(tagword)
                else:                
                    node=re.match('([^\)]+)(\)*)',tag.strip())
                    if node:
                        if node.group(1) in model:
                            outword.append(model[node.group(1)].tolist())
                        else:
                            outword.append([0]*embedding_size)
                        tagword=[0]*embedding_size
                        tagword[0]=1
                        for _ in range(len(node.group(2))-1):
                            outword.append(tagword)

        
        for tag in sentence.split():
            if tag[0]=='(':
                if tag=='(MD':
                    mdflag=1
                else:
                    mdflag=0
                    if tag[1:] in verbtags:
                        answer.append(verbtags.index(tag[1:]))
                        tag='(VB'
                        vbflag=1
                    else:
                        vbflag=0
                    if tag not in tagdict:
                        tagdict[tag]=len(tagdict)
                    tagword=[0]*embedding_size
                    tagword[tagdict[tag]]=1
                    outword.append(tagword)
            else:
                if mdflag==0:
                    node=re.match('([^\)]+)(\)*)',tag.strip())
                    if node:
                        if node.group(1) in model:
                            if vbflag==1:
                                node2=lemma(node.group(1))
                                if node2 in model:
                                    outword.append(model[node2].tolist())
                                else:
                                    outword.append([0]*embedding_size)
                            else:
                                outword.append(model[node.group(1)].tolist())
                        else:
                            outword.append([0]*embedding_size)
                        tagword=[0]*embedding_size
                        tagword[0]=1
                        for _ in range(len(node.group(2))-1):
                            outword.append(tagword)
        outword=np.array(outword)
        if outword.shape[0]>maxlength:
            print('pass')
            answer=answer[:-1]
            continue
        pads.append(outword.shape[0])
        outword=np.pad(outword,((0,maxlength-outword.shape[0]),(0,0)),'constant')
        inputs.append(outword)
    inputs=np.array(inputs)
    answers=np.zeros((len(answer),len(verbtags)))
    for num in range(len(answer)):
        answers[num][answer[num]]=1
    #print(fft)
    return count,inputs,pads,answers
print('init:2')

#dictionary, reverse_dictionary = build_dataset(training_data)
#vocab_size = len(dictionary)

# Parameters
vocab_size=len(verbtags)
# tf Graph input
x = tf.placeholder("float", [training_steps, maxlength, embedding_size])
y = tf.placeholder("float", [training_steps, len(verbtags)])
p = tf.placeholder("float", [training_steps])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([256, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, p, weights, biases):
    #x = tf.reshape(x, [-1, maxlength])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,maxlength,1)
    #x=tf.unstack(x,maxlength,axis=1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(512),rnn.BasicLSTMCell(256)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, p, dtype=tf.float32)
#input: [10 1000 100]
#output: [10 1000 512]
    outputs=tf.transpose(outputs,[1,0,2])

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, p, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
print('ready')

# Launch the graph
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
with tf.Session(config=config) as session:
#with tf.Session() as session:
    session.run(init)
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    count=patchlength
    while step < training_iters:
        count,inputs,pads,answers=list_tags(count,training_steps)
        #count,inputs,pads,answers=list_tags(0,training_steps)
        if count>=len(resp):
            count=patchlength
            continue
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: inputs, y: answers, p:pads})
        loss_total += loss
        acc_total += acc
        '''
        if (step+1) % 100==0:
            learning_rate*=0.9
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        '''
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", used: "+str(count)+ ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
            start_time=time.time()
            acc_total = 0
            loss_total = 0
        step += 1
        print(global_step)
        if acc>max_acc:
            max_acc=acc
            saver.save(session,'ckpt/n51.ckpt',global_step=step)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
