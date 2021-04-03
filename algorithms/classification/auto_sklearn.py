from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import autosklearn.classification


def autoSklearn_class(df):
  y = df.iloc[:, -1:]
  X = df.iloc[:, 0:df.shape[1]-1]
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  #automl = AutoSklearn2Classifier(
  automl = autosklearn.classification.AutoSklearnClassifier(
      time_left_for_this_task=1*60,
      per_run_time_limit=30,
      n_jobs=-1,
      ensemble_size=1,
      initial_configurations_via_metalearning=0
  )
  #con le ultime due righe di solito si va a fare overfitting
  automl.fit(X_train, y_train)
  print(automl.sprint_statistics())
  print(automl.show_models())
  y_pred = automl.predict(X_test)
  return (accuracy_score(y_test, y_pred))