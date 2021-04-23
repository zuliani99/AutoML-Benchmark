# AutoML-Benchmark
Benchmark for some usual automated machine learning, such as: [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [auto-keras](https://autokeras.com/), [h20](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [tpot](http://epistasislab.github.io/tpot/) and [autogluon](http://epistasislab.github.io/tpot/)


## Installation
First of all download the full package or clone it where ever you want. Then all you have to do is to run these line of code in your bash window: 
```bash
sudo apt install install python3-pip
sudo apt install install defaut-jre
sudo apt install install python3-tk
```

And finally to install all dependencies run 
```bash
make install
```


## Usage
And then there are three type of function operations:

```bash
python3 start.py --ndfopenml n
``` 
Where **n** is the number of datasets each for classification task and for regression task. This command will start a benchmark using openml dadtasets.

```bash
python3 start.py --dfkaggle list
```
Where **list** is a sequence of dataset name separated by a space. This command will start a benchmark using the dataset that you specify (if you put more than one the dataset, all of them have to be separated by a space).

```bash
python3 start.py --id df_id --algo algorithm
```
Where **df_id** is the id of the dataset fatched in openml and **algo** is the algorithm that we whant to test. This command will run a benchmark on the specifie dataset using the algorithm specified.

```bash
python3 start.py --help
```
This command will display the functions of the app
