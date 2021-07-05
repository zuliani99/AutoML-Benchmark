# AutoML-Benchmark
Benchmark for some usual automated machine learning, such as: [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [auto-keras](https://autokeras.com/), [h20](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [tpot](http://epistasislab.github.io/tpot/) and [autogluon](http://epistasislab.github.io/tpot/). All visualized via a responsive [Dash Ploty](https://dash.plotly.com/) Web Application.


## Installation
First of all download the full package or clone it where ever you want. Then all you have to do is to run thid line of code in your bash window: 
```bash
sudo apt install install python3-pip
```

To install all dependencies run 
```bash
make install
```

## Usage
To run the app execute the following line of code:
```bash
python3 start.py
```
Open your favourite browser and go to: [http://0.0.0.0:8050/](http://0.0.0.0:8050/). Here you will be albe to interact with the application

### Type of tests
There are five types of operations:

1. **OpenML Benchmark:** Here you can choose the number of datasets each for classification task and for regression task and the number of instances that a dataset at least has. This command will start a benchmark using openml datasets

2. **Kaggle Benchmark:** Here you can choose multiple kaggle's datasets for running a benchmark on them

3. **Test Benchmark:** Here you can run a benchmark on a specific dataset by insering the *dataset id* and using a single *algorithm* ot all of them by selecting a options

4. **Past Results OpenML:** Here you can navigate between past *OpenML* benchmark by selecting a specific date

5. **Past Results Kaggle:** Here you can navigate between past *Kaggle* benchmark by selecting a specific date

### Actions available 
In all operation these action are available:
* Analize the results of _Calssification Tasks_ and _Regression Tasks_ by a **Table** visualization, **Bar Charts** visualization and **Scatter Plot** visualization
* See the **Lifetime** of all algorithms
* Inspect the **Pipeline** of al algorithms
