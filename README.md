# AutoML-Benchmark
Benchmark for some usual automated machine learning, such as: auto-sklearn, auto-keras, h20, tpot and autogluon

To install all dependencies run **make install**

And then there are three type of function operations:

1- **python3 start.py --ndfopenml n** where n is the number of datasets each for classification task and for regression task. This command will start a benchmark using openml dadtasets

2- **python3 start.py --dfkaggle list** where list is a sequence of dataset name separated by a space 

3- **python3 start.py --id df_id --algo algorithm** where df_id is the id of the dataset fatched in openml and algo is the algorithm that we whant to test

4- **python3 start.py --help** will display the commands
