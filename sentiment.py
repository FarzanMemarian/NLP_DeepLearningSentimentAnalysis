# parser.py

import sys
from models import *
from sentiment_data import *
from pdb import set_trace
import argparse


if __name__ == '__main__':

    a = argparse.ArgumentParser()
    a.add_argument("--system_to_run", default="FF")
    a.add_argument("--nb_exm", default=1000, type=int)
    a.add_argument("--hid_step_sz", default=1000, type=int) # size of hidden layer
    a.add_argument("--init_lr", default=0.01, type=float)
    a.add_argument("--dec_step", default=100, type=int) # this is the decay step
    a.add_argument("--lrdf", default=0.99, type=float) # learning rate decay factor
    a.add_argument("--epochs", default=5 ,type=int)
    args = a.parse_args()
    system_to_run = args.system_to_run
    nb_exm = args.nb_exm
    epochs = args.epochs
    hid_step_sz = args.hid_step_sz
    init_lr = args.init_lr
    dec_step = args.dec_step
    lrdf = args.lrdf


    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    # Load train, dev, and test exs
    train_exs_whole = read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    train_exs = train_exs_whole[:nb_exm]
    dev_exs = read_and_index_sentiment_examples("data/dev.txt", word_vectors.word_indexer)
    test_exs = read_and_index_sentiment_examples("data/test-blind.txt", word_vectors.word_indexer)

    print repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"

    if system_to_run == "FF":
        test_exs_predicted = train_ffnn(train_exs, dev_exs, test_exs, word_vectors, epochs, hid_step_sz, init_lr, dec_step, lrdf)
        write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "FANCY":
        test_exs_predicted = train_fancy(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either FF or FANCY to run the appropriate system")
    # Write the test set output
    write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)