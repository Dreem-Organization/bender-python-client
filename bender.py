import requests

CURRENT_EXPERIMENT = None
BASE_URL = "http://127.0.0.1:8000"
BASE_URL = 'https://api.rythm.co/v1/dreem/bender'
TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI2ZDExNTY0YzczM2U0MDhkYTRiYzVlZWYxNjE5NTMxZiIsImV4cCI6MTQ3MTQ2MDE4NiwicGVybWlzc2lvbnMiOiJoZWFkYmFuZD10ZWFtO25vY3Rpcz1hZG1pbjtkcmVlbWVyPXRlYW07Y3VzdG9tZXI9dGVhbTtkYXRhc2V0PXRlYW07bmlnaHRyZXBvcnQ9dGVhbTtkYXRhdXBsb2FkPWFkbWluO2RhdGFzYW1wbGU9dGVhbTthbGdvcnl0aG09dGVhbTtwcm9kdWN0X3Rlc3Rpbmc9dGVhbSJ9.JRDPQVQGZWvd9C6UMNtG2Q0tDxbMgqSk21r6UI8C38w'


class Bender():
    """
    Main class for bender
    Usage Pattern:
    --------------
    bender = Bender(author=YOUR_PSEUDONYM, experiment=OPTIONAL_EXPERIMENT_ID)
    # If no experiment id provided, create one
    bender.experiment.create(
      name='My Experiment',
      description='This is a Bender experiment',
      metrics=['accuracy'],
      dataset='my_dataset.csv',
      dataset_parameters={'parameter': 'value'}
    )
    bender.algo.create(
        name='RandomForest',
        parameters=['n_estimators', 'criterion',]
    )
    bender.trial.new(
        parameters={'n_estimators': 30, 'criterion': 'gini'}
        results={'accuracy': 0.877}
        comment='not bad'
    )
    """
    def __init__(self, author, experiment=None, algo=None, latest_algo=False):
        self.author = author
        self.experiment = Experiment(experiment)
        self.algo = Algo(author=self.author, experiment=self.experiment, algo=algo,
                         latest_algo=latest_algo)
        self.trial = Trial(author=self.author, experiment=self.experiment, algo=self.algo)


class Experiment():
    """Experiment class for Bender """
    def __init__(self, experiment):
        self.id = experiment
        self.name = None
        self.description = None
        self.metrics = None
        self.author = None
        self.dataset = None
        self.dataset_parameters = None

        if experiment is not None:
            self.get(experiment)

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.description = data.get('description')
        self.metrics = data.get('metrics')
        self.author = data.get('author')
        self.dataset = data.get('dataset')
        self.dataset_parameters = data.get('dataset_parameters')

    def get(self, experiment_id):
        """Retrieve experiment instance"""
        r = requests.get(
            url='%s/experiments/%s/' % (BASE_URL, experiment_id),
            headers={"Authorization": "Bearer {0}".format(TOKEN)}
            )
        if r.status_code == 200:
            self.populate(r.json())
        else:
            raise BenderFailed('Could not retrieve experiment.')

    def create(self, name, description, metrics, author, dataset, dataset_parameters):
        """
        Please provide the following:
        - name: string
        - description: string
        - metrics: list of strings
        - dataset: string
        - dataset_parameters: dict
        """
        r = requests.post(
            url='%s/experiments/' % (BASE_URL),
            json={
              'name': name,
              'description': description,
              'metrics': metrics,
              'dataset': dataset,
              'author': author,
              'dataset_parameters': dataset_parameters},
            headers={"Authorization": "Bearer {0}".format(TOKEN)})

        if r.status_code == 201:
            self.populate(r.json())
        else:
            print(r.text)
            raise BenderFailed('Failed to create experiment.')

    def __str__(self):
        if self.name is None:
            print('Please create or get an experiment.')


class Algo():
    """ Algo class for Bender """
    def __init__(self, experiment, author, algo, latest_algo):
        self.id = None
        self.name = None
        self.parameters = None
        self.experiment = experiment
        self.experiment_id = experiment.id

        if algo is not None:
            self.get(algo)

        elif self.experiment is not None and latest_algo is True:
            self.get_latest_used_algo(experiment.id, author)

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.experiment_id = data.get('experiment')
        self.parameters = data.get('parameters')

    def get_latest_used_algo(self, experiment_id, author):
        r = requests.get(
            url='%s/latest_algo_for_experiment/%s/'
            % (BASE_URL, experiment_id),
            headers={"Authorization": "Bearer {0}".format(TOKEN)}
        )

        if r.status_code == 200:
            self.populate(r.json()[0])
        else:
            raise BenderFailed('Could not retrieve latest algo.')

    def create(self, name, parameters):
        """
        Please provide the following:
        - name: string
        - parameters: list of strings
        """
        data = {'name': name,
                'parameters': parameters,
                'experiment': self.experiment.id}

        r = requests.post(url='%s/algos/' % BASE_URL, json=data,
                          headers={"Authorization": "Bearer {0}".format(TOKEN)})

        if r.status_code == 201:
            self.populate(r.json())
        else:
            raise BenderFailed('Could not create Algo')

    def get(self, algo_id):
        """Retrieve algo instance"""
        r = requests.get(url='%s/algos/%s/' % (BASE_URL, algo_id),
                         headers={"Authorization": "Bearer {0}".format(TOKEN)})
        if r.status_code == 200:
            self.populate(r.json())
        else:
            raise BenderFailed('Could not retrieve experiment.')

    def __str__(self):
        if self.name is None:
            raise BenderFailed('Please create or get an algorithm.')
        else:
            return self.name


class Trial():
    """ Trial class for bender """
    def __init__(self, author, experiment=None, algo=None):
        self.experiment = experiment
        self.algo = algo
        self.author = author
        self.parameters = None
        self.results = None
        self.comment = None

    def populate(self, parameters, results, comment):
        self.parameters = parameters
        self.results = results
        self.comment = comment

    def new(self, parameters, results, comment=None):
        if (self.experiment is not None and self.algo is not None):
            data = {'experiment': self.experiment.id,
                    'algo': self.algo.id,
                    'author': self.author,
                    'parameters': parameters,
                    'results': results,
                    'comment': comment}
            r = requests.post(url='%s/trials/' % BASE_URL, json=data,
                              headers={"Authorization": "Bearer {0}".format(TOKEN)})
            if r.status_code == 201:
                self.populate(parameters, results, comment)
                print('Trial successfully send.')
                return r.json()
            else:
                raise BenderFailed(
                    "Could not send trial.\nPlease make sure you provided the following:\
                    \nParameters: %s.\nResults: %s."
                    % (', '.join(self.algo.parameters), ', '.join(self.experiment.metrics))
                )


class BenderFailed(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error
