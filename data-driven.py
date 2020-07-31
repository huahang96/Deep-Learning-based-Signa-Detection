import tensorflow.compat.v1 as tf
import numpy as np
import math
M = 4      #接受天线数
N = 4     #发送天线数
tf.disable_v2_behavior()

snr = 15  # SNR per receive antenna [dB]
sigma2 = (N/math.pow(10,snr/10.0))
sigma_std = math.sqrt(sigma2) # SD for w
display_step = 5
k = np.asmatrix([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
def generation_data(data_num = 1000):
    HHH = []
    yyy = []
    lls = []
    for i in range(data_num):
        H_re = np.random.normal(0, 1 / 2, ((4, 4)))
        H_im = np.random.normal(0, 1 / 2, ((4, 4)))
        Hzhenshi = H_re + 1j * H_im
        yp = Hzhenshi * k + np.random.normal(0, sigma_std, ((4, 4)))
        ycon = np.asarray(np.concatenate((np.real(yp), np.imag(yp))))
        yp_vec = np.reshape(ycon, (32,))

        HLS = yp * np.linalg.inv(k)
        HLS_vec = np.reshape(np.asarray(np.concatenate((np.real(HLS), np.imag(HLS)))), (32,))


        H = np.concatenate((H_re, H_im))
        H_vec = np.reshape(H, (32,))
        HHH.append(H_vec)
        yyy.append(yp_vec)
        lls.append(HLS_vec)

    return yyy, HHH, lls
def training():
    n_hidden_1 = 200
    n_hidden_2 = 100
    n_hidden_3 = 100
    n_hidden_4 = 50
    n_hidden_5 = 50
    n_input = 32
    n_output = 32
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_input])
    def CE_NN(X):

        Yp_noise = X
        weights = {
            'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
            'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
            'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
            'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], stddev=0.1)),
            'encoder_h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5], stddev=0.1)),
            'encoder_h6': tf.Variable(tf.truncated_normal([n_hidden_5, n_output], stddev=0.1)),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
            'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
            'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
            'encoder_b4': tf.Variable(tf.truncated_normal([n_hidden_4], stddev=0.1)),
            'encoder_b5': tf.Variable(tf.truncated_normal([n_hidden_5], stddev=0.1)),
            'encoder_b6': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),

        }

        # Encoder Hidden layer with sigmoid activation #1
        # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_1 = tf.nn.relu(tf.add(tf.matmul(Yp_noise, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
        layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['encoder_h5']), biases['encoder_b5']))
        layer_6 = tf.add(tf.matmul(layer_5, weights['encoder_h6']), biases['encoder_b6'])
        return layer_6
    H_hat = CE_NN(X)
    H_true = Y
    cost = tf.reduce_mean(tf.pow(H_true - H_hat, 2))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # Start Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)

        traing_epochs = 20
        learning_rate_current = 0.001  # 0.001
        for epoch in range(traing_epochs):
            print(epoch)
            if epoch > 0 and epoch % 20 == 0:
                learning_rate_current = learning_rate_current / 5
            avg_cost = 0.
            total_batch = 100
            for index_m in range(total_batch):
                H_data,label,_ = generation_data()
                batch_H = np.asarray(H_data)
                batch_L = np.asarray(label)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_H, Y: batch_L,
                                                              learning_rate: learning_rate_current})
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
                H_test,label,ls = generation_data(10000)
                batch_x = np.asarray(H_test)
                labels = np.asarray(label)
                lss = np.asarray(ls)
                test = sess.run(H_hat, feed_dict={X: batch_x})
                mean_error = tf.reduce_mean(tf.pow(test - labels,2))
                mean_error1 = tf.reduce_mean(tf.pow(lss - labels,2))
                print("prediction and the mean error on test set are:", mean_error.eval({X: batch_x}))
                print("the mean error on ls are:", mean_error1.eval({X: batch_x}))

        H_test, label, ls = generation_data(1)
        batch_x = np.asarray(H_test)
        labels = np.asarray(label)
        lss = np.asarray(ls)
        test = sess.run(H_hat, feed_dict={X: batch_x})
        print("prediction and the mean error on test set are:", H_hat.eval({X: batch_x}))
        print("the mean error on ls are:", lss)
        print("H are:", labels)


training()
