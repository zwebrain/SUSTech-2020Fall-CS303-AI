import json
import joblib
import argparse
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import rnn, seq2seq

def read_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data_file', type=str, default='testdataexample')
    parser.add_argument('-m', '--model_file', type=str, default='model')
    args = parser.parse_args()
    return args.test_data_file, args.model_file


if __name__ == '__main__':
    test_data_file, model_file = read_arg()
    X_test = []
    data = open(test_data_file, 'r').read()
    for i in data[1:-1].split(','):
        X_test.append(i[1:-1])
    X_test = np.array(X_test)
    max_length = 1024
    word_dict = np.load('submodule/word_dict.npy')
    dict_array2 = np.load('submodule/dict_array2.npy')
    text_model = dict()
    for i in range(len(word_dict)):
        text_model[word_dict[i]] = dict_array2[i]

    test_X, test_length = [], []
    for content in X_test:
        X = []
        for w in content.split(' ')[:max_length]:
            if w in text_model:
                X.append(np.expand_dims(text_model[w], 0))
        if X:
            length = len(X)
            X = X + [np.zeros_like(X[0])] * (max_length - length)
            X = np.concatenate(X)
            X = np.expand_dims(X, 0)
            test_X.append(X)
            test_length.append(length)

    test_sum = len(test_X)

    tf.reset_default_graph()
    batch_size = 512
    lr = 1e-3
    hidden_size = 100

    X = tf.placeholder(shape=(batch_size, max_length, 100), dtype=tf.float32, name="X")
    L = tf.placeholder(shape=(batch_size), dtype=np.int32, name="L")
    y = tf.placeholder(shape=(batch_size, 1), dtype=np.float32, name="y")
    dropout = tf.placeholder(shape=(), dtype=np.float32, name="dropout")

    with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
        def lstm_cell(hidden_size, cell_id=0):
            cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='cell%d' % cell_id)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
            return cell


        context = tf.get_variable("context", shape=(1, hidden_size))
        context = tf.tile(context, [batch_size, 1])

        fw_cell = lstm_cell(hidden_size, 0)
        bw_cell = lstm_cell(hidden_size, 1)
        fw_zero = fw_cell.zero_state(batch_size, tf.float32)
        bw_zero = fw_cell.zero_state(batch_size, tf.float32)

        encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                         cell_bw=bw_cell,
                                                                         inputs=X,
                                                                         sequence_length=L,
                                                                         initial_state_fw=fw_zero,
                                                                         initial_state_bw=bw_zero,
                                                                         dtype=tf.float32)

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

        attention_zero = attention_cell.zero_state(batch_size, tf.float32)
        attention_output, attention_state = attention_cell.call(context, attention_zero)
        aligments = attention_state[3]

        W1 = tf.get_variable("W1", shape=(hidden_size, 50))
        b1 = tf.get_variable("b1", shape=(50,))
        W2 = tf.get_variable("W2", shape=(50, 1))
        b2 = tf.get_variable("b2", shape=(1,))
        fcn1 = tf.nn.xw_plus_b(attention_output, W1, b1)
        logists = tf.nn.xw_plus_b(fcn1, W2, b2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=y))
        op = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_file))

    step = 0
    pred1 = []
    while step < test_sum:
        if step + batch_size <= test_sum:
            result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: np.concatenate(test_X[step:step + batch_size]),
                                                                 L: np.array(test_length[step:step + batch_size]),
                                                                 dropout: 1.})
            for i in result:
                if i > 0.5:
                    pred1.append(1)
                else:
                    pred1.append(0)
        else:
            _X = np.concatenate(test_X[step:] + [np.zeros_like(test_X[0])] * (batch_size - len(test_X[step:])))
            _L = np.array(test_length[step:] + [1] * (batch_size - len(test_length[step:])))
            result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: _X, L: _L, dropout: 1.})
            for i in result[:len(test_length[step:])]:
                if i > 0.5:
                    pred1.append(1)
                else:
                    pred1.append(0)
        step += batch_size

    model_lbfgs = joblib.load(model_file + "/my_model_lbfgs.m")
    model_svm = joblib.load(model_file + "/my_model_svm.m")
    model_4 = joblib.load(model_file + "/my_model4.m")
    pred2 = model_lbfgs.predict(X_test)
    pred3 = model_svm.predict(X_test)
    pred4 = model_4.predict(X_test)
    
    prdicts = []
    for i in range(test_sum):
        if pred1[i] == pred3[i]:
            prdict = pred1[i]
        elif pred2[i] == pred4[i]:
            prdict = pred2[i]
        else:
            prdict = pred1[i]
        prdicts.append(prdict)

    with open('output.txt', 'w') as f:
        for res in prdicts[:-1]:
            f.write(str(res))
            f.write('\n')
        f.write(str(prdicts[-1]))