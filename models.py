# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *
from pdb import set_trace
from random import randint
import datetime
from tqdm import tqdm
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
import math
# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def train_fancy(train_exs, dev_exs, test_exs, word_vectors, 
    train_iter=1000, batch_size=50, lstmUnits=64, learn_rate=0.001, bidir=False):
    max_seq_length = 60 #Maximum length of sentence
    num_dimensions = 300 #Dimensions for each word vector
    # 59 is the max sentence length in the corpus, so let's set this to 60
    # seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), max_seq_length) for ex in train_exs])
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), max_seq_length) for ex in dev_exs])
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), max_seq_length) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    # batch_size=50
    # lstmUnits = 64
    num_classes = 2
    num_train = len(train_mat)
    num_dev = len(dev_mat)
    num_test = len(test_mat)
    feat_vec_size = 300


    def get_batch(batch_size=50, which_dataset="train"):

         # costomize the function to the appropriate dataset
        if which_dataset=="train":
            dataset = train_mat
            dataset_labels = train_labels_arr
            seq_lens = train_seq_lens
        elif which_dataset=="dev":
            dataset = dev_mat
            dataset_labels = dev_labels_arr
            seq_lens = dev_seq_lens
        elif which_dataset=="test":
            dataset = test_mat
            dataset_labels = test_labels_arr
            seq_lens = test_seq_lens

        labels = []
        sequence_lengths = []
        arr = np.zeros([batch_size, max_seq_length], dtype=int)
        for i in range(batch_size):
            ds_size = len(dataset)
            num = randint(0,ds_size-1)
            arr[i,:] = dataset[num,:]
            labels.append(dataset_labels[num]) 
            sequence_lengths.append(seq_lens[num])

        # one-hot encoding of the labels
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        next_batch_labels = onehot_encoder.fit_transform(integer_encoded)
        return arr, next_batch_labels, sequence_lengths

    def get_ordered_data(batch_idx, which_dataset="train"):
        if which_dataset=="train":
            dataset = train_mat
            dataset_labels = train_labels_arr
            seq_lens = train_seq_lens
        elif which_dataset=="dev":
            dataset = dev_mat
            dataset_labels = dev_labels_arr
            seq_lens = dev_seq_lens
        elif which_dataset=="test":
            dataset = test_mat
            dataset_labels = test_labels_arr
            seq_lens = test_seq_lens

        labels = []
        sequence_lengths = []
        arr = np.zeros([batch_size, max_seq_length], dtype=int)
        idx = 0
        for exm in range(batch_idx * batch_size, (batch_idx+1) * batch_size):
            arr[idx,:] = dataset[exm,:]
            labels.append(dataset_labels[exm]) 
            sequence_lengths.append(seq_lens[exm])
            idx += 1
        # one-hot encoding of the labels
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        next_batch_labels = onehot_encoder.fit_transform(integer_encoded)
        return arr, next_batch_labels, sequence_lengths




    # *************************************************
    # *************************************************
    # *************************************************
    # DEFINING THE COMPUTATION GRAPH
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    input_data_idx = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    # data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]),dtype=tf.float32)
    input_data = tf.placeholder(tf.float32, [batch_size, max_seq_length, num_dimensions])
    input_data = tf.nn.embedding_lookup(word_vectors.vectors,input_data_idx)
    seq_len = tf.placeholder(tf.int32, [batch_size])
    # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.GRUCell(lstmUnits)
    lstmCell_bw = tf.contrib.rnn.GRUCell(lstmUnits)
    # lstmCell = tf.nn.BidirectionalGridLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    lstmCell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

    z_state = lstmCell.zero_state(batch_size, tf.float32)
    z_state_bw = lstmCell_bw.zero_state(batch_size, tf.float32)    
    if bidir == True:
        (rnn_outputs_fw, rnn_outputs_bw) , final_state = tf.nn.bidirectional_dynamic_rnn(lstmCell, lstmCell_bw, input_data, sequence_length=seq_len,
                                         initial_state_fw=z_state, initial_state_bw=z_state_bw)
        value = tf.concat([rnn_outputs_fw,rnn_outputs_bw],2)
    else:
        value, state = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32, time_major=False)
    # value, state = tf.nn.static_rnn(lstmCell, input_data, dtype=tf.float32)

    # weight = tf.Variable(tf.truncated_normal([lstmUnits, num_classes]))
    weight = tf.get_variable("weight", [lstmUnits, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=3))
    bias = tf.Variable(tf.constant(0.01, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    average_val = tf.reduce_mean(tf.gather(value, tf.range(0, tf.reduce_max(seq_len))), axis=0)
    # average_val = tf.reduce_mean(tf.gather(value, tf.range(0, tf.reduce_max(seq_len))), axis=1)
    last = tf.gather(value, int(value.get_shape()[1]) - 1)
    prediction = (tf.matmul(average_val, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    dev_accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    dev_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
    # global_step = tf.contrib.framework.get_or_create_global_step()
    # opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
    # # Loss is the thing that we're optimizing
    # grads = opt.compute_gradients(loss)
    # # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # with tf.control_dependencies([apply_gradient_op]):
    #     train_op = tf.no_op(name='train')

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learn_rate).minimize(loss)

    # TensorFlow
    tf.summary.scalar('Dev loss', dev_loss)
    tf.summary.scalar('Dev accuracy', dev_accuracy)
    merged1 = tf.summary.merge_all()
    tf.summary.scalar('Train loss', loss)
    tf.summary.scalar('Train Accuracy', accuracy)
    tf.summary.scalar('weight', tf.reduce_sum(weight))
    tf.summary.scalar('bias', tf.reduce_sum(bias))
    # tf.summary.tensor_summary("value", value)
    # tf.summary.scalar('average_val', tf.reduce_sum(average_val))
    merged = tf.summary.merge_all()
    logdir = "tensorboard/part2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    # logdir = "tensorboard/part2/" + "run1" + "/"
    # *************************************************
    # *************************************************
    # *************************************************
    # Training
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())



    

    writer = tf.summary.FileWriter(logdir, sess.graph)
    min_dev_loss = np.exp(10)
    for i in tqdm(range(train_iter)):
        #Next Batch of reviews
        next_batch, next_batch_labels, sequence_lengths = get_batch(batch_size, "train");
        sess.run(optimizer, {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
        # sess.run(train_op, {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
        
        # Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
            writer.add_summary(summary, i)


            next_batch, next_batch_labels, sequence_lengths = get_batch(batch_size, "dev");
            summary1 = sess.run(merged1, {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
            writer.add_summary(summary1, i)

            # print "value shape", np.shape(value)
            # print "average_val shape", np.shape(average_val)
            # print "seq_len1 shape", np.shape(seq_len1)
  test_exs_predicted          # print "data1 shape", np.shape(data1)
            # print "last1 shape", np.shape(last1)

        # if (i % 100 == 0):
        #     correctPred1 = sess.run(correctPred, {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
        #     print "correctPred:", correctPred[1:5]




        # Save the network every 10,000 training iterations
        if (i % (train_iter/10) == 0 and i != 0):
            ave_dev_loss = 0
            ave_dev_acc = 0
            num_dev_iter = 20
            for j in range(num_dev_iter):
                next_batch, next_batch_labels, sequence_lengths = get_batch(batch_size, "dev")
                dev_loss, dev_accuracy = sess.run([loss, accuracy], {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
                ave_dev_loss += dev_loss
                ave_dev_acc += dev_accuracy
            ave_dev_loss /= num_dev_iter
            ave_dev_acc /= num_dev_iter
            writer.add_summary(summary, i)

            print "dev loss: ", ave_dev_loss
            print "dev accuracy: ", ave_dev_acc
            if ave_dev_loss < min_dev_loss:
                min_dev_loss = ave_dev_loss
                save_path = saver.save(sess, "models/training_lstm.ckpt", global_step=i)
                print("saved to %s" % save_path)

    writer.close()
















    # DEFINING THE COMPUTATION GRAPH
    # tf.reset_default_graph()
    # labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    # input_data_idx = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    # # data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]),dtype=tf.float32)
    # input_data = tf.placeholder(tf.float32, [batch_size, max_seq_length, num_dimensions])
    # input_data = tf.nn.embedding_lookup(word_vectors.vectors,input_data_idx)
    # seq_len = tf.placeholder(tf.int32, [batch_size])
    # lstmCell = tf.contrib.rnn.GRUCell(lstmUnits)
    # # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    # value, state = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32, time_major=False)
    # # value, state = tf.nn.static_rnn(lstmCell, input_data, dtype=tf.float32)

    # # weight = tf.Variable(tf.truncated_normal([lstmUnits, num_classes]))
    # weight = tf.get_variable("weight", [lstmUnits, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=3))
    # bias = tf.Variable(tf.constant(0.01, shape=[num_classes]))
    # value = tf.transpose(value, [1, 0, 2])
    # average_val = tf.reduce_mean(tf.gather(value, tf.range(0, tf.reduce_max(seq_len))), axis=0)
    # # average_val = tf.reduce_mean(tf.gather(value, tf.range(0, tf.reduce_max(seq_len))), axis=1)
    # last = tf.gather(value, int(value.get_shape()[1]) - 1)
    # prediction = (tf.matmul(average_val, weight) + bias)

    # correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    # accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    # dev_accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    # dev_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    # optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
    # tf.reset_default_graph()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    # evaluating the model
    dataset_types = ["train", "dev"]
    for dataset_type in dataset_types:
        print "evaluating trained model's performance on    " + dataset_type + "    dataset: "
        # evaluate on training set
        if dataset_type == "train":
            num_batches = int(math.floor(num_train/batch_size))
        elif dataset_type == "dev":
            num_batches = int(math.floor(num_dev/batch_size))
        
        ave_loss = 0
        ave_acc = 0
        for batch_idx in range(num_batches-1):
            next_batch, next_batch_labels, sequence_lengths = get_ordered_data(batch_idx, dataset_type)
            # set_trace()
            loss1, accuracy1 = sess.run([loss, accuracy], {input_data_idx: next_batch, labels: next_batch_labels, seq_len:sequence_lengths})
            ave_loss += loss1
            ave_acc += accuracy1
        ave_loss /= num_batches
        ave_acc /= num_batches
        print "loss:   ", ave_loss
        print "accuracy:   ", ave_acc







# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors, epochs, hidden_size, init_lr, d_s, lrdf):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # Inputs are of size 2
    feat_vec_size = len(word_vectors.vectors[0])
    # Let's use 10 hidden units
    embedding_size = hidden_size
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # *************************************************
    # *************************************************
    # *************************************************
    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, feat_vec_size)
    # Other initializers like tf.random_normal_initializer are possible too
    V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.   
    z = tf.nn.relu(tf.tensordot(V, fx, 1))

    # V2 = tf.get_variable("V2", [embedding_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # # z2 is activation of second hidden layer
    # z2 = tf.nn.relu(tf.tensordot(V2, z, 1))

    # not explicitly initialized version of W
    # W = tf.get_variable("W", [num_classes, embedding_size]) 
    # explicitly initialized version of W
    W = tf.get_variable("W", [num_classes, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    unscaled_log_prob= tf.log(tf.tensordot(W, z, 1))
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(probs)         

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=unscaled_log_prob)


    # *************************************************
    # *************************************************
    # *************************************************
    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = d_s
    learning_rate_decay_factor = lrdf
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = init_lr
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                   staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')


    # *************************************************
    # *************************************************
    # *************************************************
    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = epochs
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('tensorboard/part1', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in tqdm(range(0, num_epochs)):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx, example in tqdm(enumerate(train_exs)):

                # In the next four lines we calculate the mean of the word vectors
                # for the current sentence
                sum_vec = np.zeros(len(word_vectors.vectors[0]))
                for idx in example.indexed_words:
                    sum_vec += word_vectors.vectors[idx]
                mean_vec = sum_vec/len(example.indexed_words)

                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], 
                                feed_dict = {fx: mean_vec,        label: np.array([example.label])})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                # set_trace()
                loss_this_iter += loss_this_instance
            loss_this_iter /= len(train_exs)
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
        

        # Evaluate on the train set
        train_correct = 0
        for ex_idx, example in enumerate(train_exs):
            sum_vec = np.zeros(len(word_vectors.vectors[0]))
            for idx in example.indexed_words:
                sum_vec += word_vectors.vectors[idx]
            mean_vec = sum_vec/len(example.indexed_words)

            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                      feed_dict={fx: mean_vec})
            if (example.label == pred_this_instance):
                train_correct += 1
            # print "Example " + "; gold = " + repr(example.label) + "; pred = " +\
            #        repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
        print "results for training set:"
        print repr(train_correct) + "/" + repr(len(train_exs)) + " correct after training"

        # Evaluate on the dev set
        train_correct = 0
        for ex_idx, example in enumerate(dev_exs):
            sum_vec = np.zeros(len(word_vectors.vectors[0]))
            for idx in example.indexed_words:
                sum_vec += word_vectors.vectors[idx]
            mean_vec = sum_vec/len(example.indexed_words)

            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                      feed_dict={fx: mean_vec})
            if (example.label == pred_this_instance):
                train_correct += 1
            # print "Example " + "; gold = " + repr(example.label) + "; pred = " +\
            #        repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
        print "results for dev set:"
        print repr(train_correct) + "/" + repr(len(dev_exs)) + " correct after training"

        # Evaluate on the test set
        train_correct = 0
        predictions_test = []
        for ex_idx, example in enumerate(test_exs):
            sum_vec = np.zeros(len(word_vectors.vectors[0]))
            for idx in example.indexed_words:
                sum_vec += word_vectors.vectors[idx]
            mean_vec = sum_vec/len(example.indexed_words)
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                      feed_dict={fx: mean_vec})
            if (example.label == pred_this_instance):
                train_correct += 1
            # print "Example " + "; gold = " + repr(example.label) + "; pred = " +\
            #        repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
            # print "  Hidden layer activations for this example: " + repr(z_this_instance)
            prediction = SentimentExample(example.indexed_words, pred_this_instance)
            predictions_test.append(prediction)
        print "results for test set:"
        print repr(train_correct) + "/" + repr(len(test_exs)) + " correct after training"



        return predictions_test
# Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
