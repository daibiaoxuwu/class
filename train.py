#-*- coding:utf-8 -*-
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

# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')#too high?
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 15, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 300, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
tf.flags.DEFINE_string('train_file', 'rt_train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'rt_test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_string('pre_trained_vec', None, 'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')
tf.flags.DEFINE_integer('vocab_size', 1000, 'num of train steps for saving model')
tf.flags.DEFINE_integer('n_classes', 6, 'num of train steps for saving model')
tf.flags.DEFINE_integer('num_batches', 1000, 'num of train steps for saving model')

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
embedding_size=100
patchlength=0

maxlength=700
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step=global_step, decay_steps=100,decay_rate=0.9)
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

def main(_):
    FLAGS.n_classes = data_loader.n_classes
    FLAGS.num_batches = data_loader.num_batches

    test_data_loader = InputHelper()
    test_data_loader.load_dictionary(FLAGS.data_dir+'/dictionary')
    test_data_loader.create_batches(FLAGS.data_dir+'/'+FLAGS.test_file, 100, FLAGS.sequence_length)

    if FLAGS.pre_trained_vec:
        embeddings = np.load(FLAGS.pre_trained_vec)
        print embeddings.shape
        FLAGS.vocab_size = embeddings.shape[0]
        FLAGS.embedding_size = embeddings.shape[1]

    if FLAGS.init_from is not None:
        assert os.path.isdir(FLAGS.init_from), '{} must be a directory'.format(FLAGS.init_from)
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)
        assert ckpt,'No checkpoint found'
        assert ckpt.model_checkpoint_path,'No model path found in checkpoint'

    # Define specified Model
    model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,    
        vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
        n_classes=FLAGS.n_classes, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)

    # define value for tensorboard
    tf.summary.scalar('train_loss', model.cost)
    tf.summary.scalar('accuracy', model.accuracy)
    merged = tf.summary.merge_all()

    # 调整GPU内存分配方案
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # using pre trained embeddings
        if FLAGS.pre_trained_vec:
            sess.run(model.embedding.assign(embeddings))
            del embeddings

        # restore model
        if FLAGS.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in xrange(FLAGS.num_batches):
            data_loader.reset_batch()#shuffle
            total_loss=0
            for b in xrange(FLAGS.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data:x, model.targets:y, model.output_keep_prob:FLAGS.dropout_keep_prob}
                train_loss, summary,  _ = sess.run([model.cost, merged, model.train_op], feed_dict=feed)
                end = time.time()


                print('{}/{} , train_loss = {:.3f}, time/batch = {:.3f}'.format(global_step, FLAGS.num_batches,  train_loss, end - start))
                total_loss+=train_loss


                if global_step % 20 == 0:
                    train_writer.add_summary(summary, e * FLAGS.num_batches + b)

                if global_step % FLAGS.save_steps == 0:
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')        
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print 'model saved to {}'.format(checkpoint_path)

            test_data_loader.reset_batch()
            print ' loss:',total_loss/FLAGS.num_batches
            '''
            test_accuracy = []
            for i in xrange(test_data_loader.num_batches):
                test_x, test_y = test_data_loader.next_batch()
                feed = {model.input_data:test_x, model.targets:test_y, model.output_keep_prob:1.0}
                accuracy = sess.run(model.accuracy, feed_dict=feed)
                test_accuracy.append(accuracy)
            print 'test accuracy:{0}'.format(np.average(test_accuracy)),' loss:',total_loss/FLAGS.num_batches
            '''

if __name__ == '__main__':
    tf.app.run()
