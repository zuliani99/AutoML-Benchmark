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

#### Changes that have to be made to run AutoKeras:
* Search the file **auto_model.py** and modify these functions:
```python
def predict(self, x, batch_size=32, verbose=1, custom_objects={}, **kwargs):
    if isinstance(x, tf.data.Dataset) and self._has_y(x):
        x = x.map(lambda x, y: x)
    self._check_data_format((x, None), predict=True)
    dataset = self._adapt(x, self.inputs, batch_size)
    pipeline = self.tuner.get_best_pipeline()
    if custom_objects:
        model = self.tuner.get_best_model(custom_objects=custom_objects)
    else:
        model = self.tuner.get_best_model()
    dataset = pipeline.transform_x(dataset)
    dataset = tf.data.Dataset.zip((dataset, dataset))
    y = model.predict(dataset, **kwargs)
    y = utils.predict_with_adaptive_batch_size(
        model=model, batch_size=batch_size, x=dataset, verbose=verbose, **kwargs
    )
    return pipeline.postprocess(y)
        
def evaluate(self, x, y=None, batch_size=32, verbose=1, custom_objects={},**kwargs):
    self._check_data_format((x, y))
    if isinstance(x, tf.data.Dataset):
        dataset = x
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
    x = self._adapt(x, self.inputs, batch_size)
    y = self._adapt(y, self._heads, batch_size)
    dataset = tf.data.Dataset.zip((x, y))
    pipeline = self.tuner.get_best_pipeline()
    dataset = pipeline.transform(dataset)
    if custom_objects:
        model = self.tuner.get_best_model(custom_objects=custom_objects)
    else:
        model = self.tuner.get_best_model()
    return utils.evaluate_with_adaptive_batch_size(
        model=model, batch_size=batch_size, x=dataset, verbose=verbose, **kwargs
    )
        
 def export_model(self, custom_objects={}):
    if custom_objects:
        return self.tuner.get_best_model(custom_objects=custom_objects)
    else:
        return self.tuner.get_best_model()
```

* Search the file **tuner.py** and modify the function:
```python
def get_best_model(self, custom_objects={}):
    with hm_module.maybe_distribute(self.distribution_strategy):
        if custom_objects:
            model = tf.keras.models.load_model(self.best_model_path, custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(self.best_model_path)
    return model
```

## Usage
To run the app execute the following line of code:
```bash
python3 start.py
```
Open your favourite browser and go to: [http://0.0.0.0:8050/](http://0.0.0.0:8050/). Here you will be albe to interact with the application

### Type of tests
There are five types of operations:

1. **OpenML Benchmark:** Here you can choose the number of dataframes each for classification task and for regression task and the number of instances that a dataframe at least has. This command will start a benchmark using openml dataframes

2. **Kaggle Benchmark:** Here you can choose multiple kaggle's dataframes for running a benchmark on them

3. **Test Benchmark:** Here you can run a benchmark on a specific dataframe by insering the *dataframe id* and using a single *algorithm* ot all of them by selecting a options

4. **Past Results OpenML:** Here you can navigate between past *OpenML* benchmark by selecting a specific date

5. **Past Results Kaggle:** Here you can navigate between past *Kaggle* benchmark by selecting a specific date

### Actions available 
In all operation these action are available:
* Analize the results of _Calssification Tasks_ and _Regression Tasks_ by a **Table** visualization, **Bar Charts** visualization and **Scatter Plot** visualization
* See the **Lifetime** of all algorithms
* Inspect the **Pipeline** of al algorithms
