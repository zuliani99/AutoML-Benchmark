# AutoML-Benchmark
Benchmark for some usual automated machine learning, such as: [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [auto-keras](https://autokeras.com/), [h20](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [tpot](http://epistasislab.github.io/tpot/) and [autogluon](http://epistasislab.github.io/tpot/)


## Installation
First of all download the full package or clone it where ever you want. Then all you have to do is to run these lines of code in your bash window: 
```bash
sudo apt install install python3-pip
sudo apt install install defaut-jre
sudo apt install install python3-tk
```

To install all dependencies run 
```bash
make install
```

Then you have to sing in [kaggle](https://www.kaggle.com/) or create a new. After this step you have to download you api token by moving into these page *Your Profile* -> *Account* -> *Create New API Token* and download the JSON file.
And as last command you have to move the API Token into the *.kaggle* folder, you can do this by you own or run this line of command in your bash
```bash
mv /home/YOUR_USERNAME/Downloads/kaggle.json /home/YOUR_USERNAME/.kaggle
```

## Usage
To run the app execute the following line of code:
```bash
python3 start.py
```
Open your favourite browser and go to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/). Here you will be albe to interact with the application.

There are five types of operations:

1. **OpenML Benchmark:** Here you can choose the number of datasets each for classification task and for regression task and the number of instances that a dataset at least has. This command will start a benchmark using openml datasets.

2. **Kaggle Benchmark:** Here you can choose multiple kaggle's datasets for running a benchmark on them.

3. **Test Benchmark:** Here you can run a benchmark on a specific dataset by insering the *dataset id* and using a single *algorithm* ot all of them by selecting a options

4. **Risulatati Precedenti OpenML:** Here you can navigate between past *OpenML* benchmark by selecting a specific date

5. **Risulatati Precedenti Kaggle:** Here you can navigate between past *Kaggle* benchmark by selecting a specific date
