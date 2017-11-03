This code includes two types of models for sentiment analysis. The first is a simple feedforwad neural network. To run the code you need to run a command like the following:

python2 sentiment.py --system_to_run="FF" --nb_exm=8530 --epochs=5 --hid_step_sz=100 --init_lr=0.01  --dec_step=100 --lrdf=0.99


The other model that is implemented is different versions of RNNs. To run the code you need a command similar to the following:

python sentiment.py --system_to_run="FANCY" --nb_exm=8530 --train_iter=1000 --batchSize=30 --lstmUnits=60 --learn_rate=0.01
