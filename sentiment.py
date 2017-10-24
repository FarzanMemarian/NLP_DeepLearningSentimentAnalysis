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
    a.add_argument("--beam_size", default=5, type=int)
    a.add_argument("--early_stopping", default=True)
    a.add_argument("--extra_features", default=False)
    a.add_argument("--run_on_test", default=True)
    a.add_argument("--epochs", default=5 ,type=int)
    args = a.parse_args()
    system_to_run = args.system_to_run
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train, dev, and test exs
    train_exs = read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    dev_exs = read_and_index_sentiment_examples("data/dev.txt", word_vectors.word_indexer)
    test_exs = read_and_index_sentiment_examples("data/test-blind.txt", word_vectors.word_indexer)
    print repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"


    if system_to_run == "FF":
        test_exs_predicted = train_ffnn(train_exs, dev_exs, test_exs, word_vectors)
        write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "FANCY":
        test_exs_predicted = train_fancy(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either FF or FANCY to run the appropriate system")
    # Write the test set output
    write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)