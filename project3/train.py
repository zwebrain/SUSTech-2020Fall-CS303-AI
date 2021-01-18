import os
import json
import joblib
import numpy as np
import argparse
from sklearn.datasets import base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

def read_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_data_file', type=str, default='train.json')
    parser.add_argument('-m', '--model_file', type=str, default='model')
    args = parser.parse_args()
    return args.train_data_file, args.model_file


if __name__ == '__main__':
    train_data_file, model_file = read_arg()
    bunch=base.Bunch(label=[], contents=[])
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    with open(train_data_file,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
        for tmp in json_data:
            bunch.contents.append(" ".join(tmp['data'].split(' ')))
            bunch.label.append(tmp['label'])
        X_train, X_test, y_train, y_test=train_test_split(bunch.contents, bunch.label, train_size=0.9)
    tfidfspace = base.Bunch(label=bunch.label,tdm=[], vocabulary={})
    vectorizer=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    clf = SVC(C=10,gamma=1)
    pip=make_pipeline(vectorizer,clf)
    param_grid = {'svc__C': [10], 'svc__gamma': [1]}
    grid_search = GridSearchCV(pip, param_grid, n_jobs=-1)
    grid_search.fit(X_train,y_train)
    # print('score', grid_search.score(X_test, y_test))
    joblib.dump(grid_search, model_file + "/my_model_svm.m")
    clf = LogisticRegression(penalty='l2')
    pip = make_pipeline(vectorizer,clf)
    pip.fit(X_train,y_train)
    # print('score', pip.score(X_test, y_test))
    joblib.dump(pip, model_file + "/my_model_lbfgs.m")
    clf = RandomForestClassifier()
    pip = make_pipeline(vectorizer,clf)
    pip.fit(X_train,y_train)
    # print('score', pip.score(X_test, y_test))
    joblib.dump(pip, model_file + "/my_model4.m")
    
    max_length = 1024
    word_dict = np.load('submodule/word_dict.npy')
    dict_array2 = np.load('submodule/dict_array2.npy')
    text_model = dict()
    for i in range(len(word_dict)):
        text_model[word_dict[i]] = dict_array2[i]
    
    X_train2, train_length, y_train2 = [], [], []
    for line in range(len(X_train)):
        content= X_train[line].split(' ')
        X, y = [], y_train[line]
        for w in content[:max_length]:
            if w in text_model:
                X.append(np.expand_dims(text_model[w], 0))
        if X:
            length = len(X)
            X = X + [np.zeros_like(X[0])] * (max_length - length)
            X = np.concatenate(X)
            X = np.expand_dims(X, 0)
            X_train2.append(X)
            y_train2.append(y)
            train_length.append(length)
    
    batch_size = 16
    lr = 1e-3
    hidden_size = 100
    X = tf.placeholder(shape=(batch_size, max_length, 100), dtype=tf.float32, name="X")
    L = tf.placeholder(shape=(batch_size), dtype=np.int32, name="L")
    y = tf.placeholder(shape=(batch_size, 1), dtype=np.float32, name="y")
    dropout = tf.placeholder(shape=(), dtype=np.float32, name="dropout")

    with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
        def lstm_cell(hidden_size, cell_id=0):
            # LSTM细胞生成器
            cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='cell%d' % cell_id)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
            return cell
    
        context = tf.get_variable("context", shape=(1, hidden_size))
        context = tf.tile(context, [batch_size, 1])
    
        # BiLSTM部分
        fw_cell = lstm_cell(hidden_size, 0)
        bw_cell = lstm_cell(hidden_size, 1)
        fw_zero = fw_cell.zero_state(batch_size, tf.float32)
        bw_zero = fw_cell.zero_state(batch_size, tf.float32)
    
        # Seq2Seq版的dynamic_rnn
        encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                             cell_bw=bw_cell,
                                                             inputs=X,
                                                             sequence_length=L,
                                                             initial_state_fw=fw_zero,
                                                             initial_state_bw=bw_zero,
                                                             dtype=tf.float32)
    
        # Attention模块
        attention_context = tf.concat(encoder_output, axis=2)
        attention_mech = seq2seq.BahdanauAttention(hidden_size * 2,
                                                     memory=attention_context,
                                                     memory_sequence_length=L,
                                                     name="AttentionMechanism")
        attention_cell = seq2seq.AttentionWrapper(cell=lstm_cell(hidden_size, 2),
                                                    attention_mechanism=attention_mech,
                                                    attention_layer_size=hidden_size,
                                                    alignment_history=True,
                                                    output_attention=True,
                                                    name="AttentionCell")
    
        # Attention加权得到的context向量
        attention_zero = attention_cell.zero_state(batch_size, tf.float32)
        attention_output, attention_state = attention_cell.call(context, attention_zero)
        aligments = attention_state[3]
    
        # 用context向量直接用MLP做二分类
        W1 = tf.get_variable("W1", shape=(hidden_size, 50))
        b1 = tf.get_variable("b1", shape=(50,))
        W2 = tf.get_variable("W2", shape=(50, 1))
        b2 = tf.get_variable("b2", shape=(1,))
        fcn1 = tf.nn.xw_plus_b(attention_output, W1, b1)
        logists = tf.nn.xw_plus_b(fcn1, W2, b2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=y))     # 交叉熵
        op = tf.train.AdamOptimizer(lr).minimize(loss)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        total_step = 10000
        step = 0
        cursor = 0
        
        while step < total_step:
            _X, _L, _y = X_train2[cursor: cursor + batch_size], train_length[cursor: cursor + batch_size], y_train2[cursor: cursor + batch_size]
            cursor += batch_size
            if len(_X) < batch_size:
                cursor = batch_size - len(_X)
                _X += X_train2[: cursor]
                _L += train_length[: cursor]
                _y += y_train2[: cursor]
            _X = np.concatenate(_X)
            _L = np.reshape(np.array(_L, dtype=np.int32), (-1))
            _y = np.reshape(np.array(_y, dtype=np.float32), (batch_size, 1))
            _, l = sess.run([op, loss], feed_dict={X: _X, L:_L, y: _y, dropout:.5})
            if step % 10 == 0:
                print("step:", step, " loss:", l)
                saver.save(sess, model_file + "/model", global_step=step)
            step += 1