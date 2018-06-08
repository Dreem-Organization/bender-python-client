# Getting started with bender

First you need to create an account on https://bender.rythm.co

Then to install

```
pip install bender-client
```

Open a python console and do:

```python
from bender import Bender

bender = Bender()
```

This will ask for your email and password. The client will use these to login and retrieve a TOKEN.
This TOKEN is personal, it should not be shared, it will be stored in your home folder as
".bender_token".

You will not be asked for your login/password again until it expires.

:warning: Your TOKEN is personal. You should not give it or add it to any public repository.


### Get started with an experiment:

You can manage and create your experiments from ui at https://bender.rythm.co

You can also use:

```python
from bender import Bender

bender = Bender()

bender.new_experiment(
  name='My Experiment',
  description='This is a Bender experiment',
  metrics=['test_accuracy', 'train_accuracy'],
  dataset='my_dataset.csv',
  dataset_parameters={
    'CV_folds': '10',
    'version': '2006'
  }
)

bender.experiment.name
>>> 'My Experiment',
bender.experiment.id
>>> "51b52d11-926d-4b3f-9e76-c341a94a010c"
```

If you created your experiment online, or want to retrieve a previous experiment just do:

```python
from bender import Bender

bender = Bender()

bender.set_experiment("51b52d11-926d-4b3f-9e76-c341a94a010c")
```

or alternatively

```python
from bender import Bender

bender = Bender(experiment_id="51b52d11-926d-4b3f-9e76-c341a94a010c")
```

### Create an Algo

Once you have an experiment you can attach algos to it.

You can manage and create your algos from ui at https://bender.rythm.co

You can also use:

```python

# Describe parameters (name is mandatory, category and search_space are optional)
parameters = [
    {
      "name": 'n_estimators',
      "category": "uniform",
      "search_space": {
        "low": 1,
        "high": 1000,
        "step": 1,
      }
    },
    {
      "name": 'criterion',
      "category": "categorical",
      "search_space": {
        "values": ["gini", "entropy"]
      },
    },
    {
      "name": 'max_depth',
      "category": "loguniform",
      "search_space": {
        "low": 1e2,
        "high": 1e5,
        "step": 10,
      }
    },
    {
      "name": 'learning_rate',
      "category": "lognormal",
      "search_space": {
        "mu": 1e-3,
        "sigma": 1e1,
      }
    }
]

bender.new_algo(
    name='RandomForest',
    parameters=parameters,
    description={
     "version": "1",
     "description": "Basic random forest from scikit-learn."
    }
)

bender.algo.name
>>> "RandomForest"
bender.algo.id
>>> "62198422-0b79-4cae-a2a4-30969f147ad7"
```

If you created your algo online, or want to retrieve a previous algo just do:


```python

bender.set_algo("62198422-0b79-4cae-a2a4-30969f147ad7")
```

Or even

```python
from bender import Bender

bender = Bender(algo_id="62198422-0b79-4cae-a2a4-30969f147ad7")
```

This will set both the algo and experiment attributes.

:warning: You need to set an experiment or you won't be able to create an algo.

### Get a suggestion

If you defined the search space and category for each parameter of your algo, bender will be able to suggest you new parameters to try.

You will have to tell bender wich metric he should optimize (and weither it is a loss or a 
reward metric)

In our previous example, we want to optimize parameter to get the highest "test_accuracy" possible.

```python

from bender import Bender

bender = Bender(algo_id="62198422-0b79-4cae-a2a4-30969f147ad7")

>>> bender.suggest(metric="test_accuracy", is_loss=False)
{"n_estimators": 126, "criterion": "giny"}
>>> bender.suggest(metric="test_accuracy", is_loss=False)
{"n_estimators": 785, "criterion": "giny"}
>>> bender.suggest(metric="test_accuracy", is_loss=False)
{"n_estimators": 21, "criterion": "entropy"}

```


### Send a trial

Once you have evaluated your model with a peculiar set of parameters you should register it
with bender's trial.

Possible optimizers are:
- random
- parzen_estimator
- model_based_estimator

Here is a complete scenario with previous experiment and algo:

```python

from bender import Bender

bender = Bender(algo_id="62198422-0b79-4cae-a2a4-30969f147ad7")

parameters = bender.suggest(metric="test_accuracy", is_loss=False, optimizer="model_based_estimator")

test_accuracy, train_accuracy = my_function(data, parameters)

bender.new_trial(
  parameters=parameters,
  results={"test_accuracy": test_accuracy, "train_accuracy": train_accuracy}
)

```

You can find you experiments, algos and trials at https://bender.rythm.co
