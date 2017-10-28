# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *
from pdb import set_trace
from random import randint


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    maxSeqLength = 10 #Maximum length of sentence
    numDimensions = 300 #Dimensions for each word vector
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 100000
    set_trace()

    # *************************************************
    # *************************************************
    # *************************************************
    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


    # load in the network
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))


    # 
    numFiles = len(train_exs)

    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for pf in positiveFiles:
       with open(pf, "r") as f:
           indexCounter = 0
           line=f.readline()
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           fileCounter = fileCounter + 1 

    for nf in negativeFiles:
       with open(nf, "r") as f:
           indexCounter = 0
           line=f.readline()
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           fileCounter = fileCounter + 1 
    #Pass into embedding function and see if it evaluates. 

    np.save('idsMatrix', ids)





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
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx, example in enumerate(train_exs):

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
