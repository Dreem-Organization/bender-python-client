import requests

CURRENT_EXPERIMENT = None
BASE_URL = "http://127.0.0.1:8000"


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
    def __init__(self, author, experiment=None, latest_algo=False):
        self.author = author
        self.experiment = Experiment(experiment)
        self.algo = Algo(latest_algo, author=self.author, experiment=self.experiment)
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
        r = requests.get(url='%s/experiments/%s/' % (BASE_URL, experiment_id))
        if r.status_code == 200:
            self.populate(r.json())
        else:
            print('Could not retrieve experiment')
            return r.content

    def create(self, name, description, metrics, author, dataset, dataset_parameters):
        """
        Please provide the following:
        - Name: string
        - Description: string
        - Metrics: list of strings
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
              'dataset_parameters': dataset_parameters})

        if r.status_code == 201:
            self.populate(r.json())
            print('Created experiment "%s" with id:%s' % (self.name, self.id))

        else:
            print('Failed to create experiment.')
            return r.content

    def __str__(self):
        if self.name is None:
            print('Please create or get an experiment.')


class Algo():
    """ Algo class for Bender """
    def __init__(self, experiment, author, latest_algo):
        self.id = None
        self.name = None
        self.experiment = experiment.id

        if latest_algo is True:
            self.get_lastest_used_algo(experiment.id, author)

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.experiment = data.get('experiment')

    def get_lastest_used_algo(self, experiment_id, author):
        r = requests.get(
            url='%s/lastest_algo_for_experiment/%s'
            % (BASE_URL, experiment_id)
        )

        if r.status_code == 200:
            self.populate(r.json()[0])
            print('Retrieved last Algo.')

    def create(self, name, parameters):
        """
        Please provide the following:
        - name: string
        - parameters: list of strings
        """
        data = {'name': name,
                'parameters': parameters,
                'experiment': self.experiment}

        r = requests.post(url='%s/algos/' % BASE_URL, json=data)

        if r.status_code == 201:
            self.populate(r.json())
            print('Algo Created')
        else:
            print('Could not create Algo')
            print(r.content)

    def get(self, algo_id):
        """Retrieve algo instance"""
        r = requests.get(url='%s/algos/%s/' % (BASE_URL, algo_id))
        if r.status_code == 200:
            self.populate(r.json())
        else:
            print('Could not retrieve experiment')
            return r.content

    def __str__(self):
        if self.name is None:
            return 'Please create or get an algorithm.'
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

            r = requests.post(url='%s/trials/' % BASE_URL, json=data)
            if r.status_code == 201:
                self.populate(parameters, results, comment)
                print('Trial successfully send.')
                return r.json()
            else:
                print('Could not send trial.')
                return r.content
