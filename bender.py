import requests

CURRENT_EXPERIMENT = None
BASE_URL = "http://127.0.0.1:8000"


"""
 USAGE PATTERN
--------------

from bender import bender
bd = Bender(author="soukiassianb", experiment=28)
bd.algo.create(name='Random Forest', parameters={})
bd.new_trial(parameters={}, results={}, comment='Nice')
"""


class Experiment():
    """ Experiment class for Bender """
    def __init__(self, experiment, algo):
        self.id = experiment
        self.name = None
        self.description = None
        self.metrics = None
        self.author = None
        self.dataset = None
        self.dataset_parameters = None

        self.algo = algo

        if experiment is not None:
            self.get(experiment)

    def get(self, experiment_id):
        r = requests.get(url='%s/experiments/%s/' % (BASE_URL, experiment_id))
        if r.status_code == 200:
            self.populate(r.json())
        else:
            print('Could not retrieve experiment')
            return r.content

    def create(self, name, description, metrics, author, dataset, dataset_parameters):
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

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.description = data.get('description')
        self.metrics = data.get('metrics')
        self.author = data.get('author')
        self.dataset = data.get('dataset')
        self.dataset_parameters = data.get('dataset_parameters')

        self.algo.experiment = data.get('id')

    def __str__(self):
        if self.name is None:
            print('Please create or get an experiment.')


class Algo():
    """ Algo class for Bender """
    def __init__(self, experiment=None):
        self.id = None
        self.name = None
        self.experiment = experiment

    def create(self, name, parameters):
        data = {'name': name,
                'parameters': parameters,
                'experiment': self.experiment}
        r = requests.post(url='%s/algos/' % BASE_URL, json=data)
        if r.status_code == 201:
            self.populate(r.json())
            print('ALLL OK')
        else:
            print('Could not create Algo')
            print(r.content)

    def get(self, algo_id):
        r = requests.get(url='%s/algos/%s/' % (BASE_URL, algo_id))
        if r.status_code == 200:
            self.populate(r.json())
        else:
            print('Could not retrieve experiment')
            return r.content

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.experiment = data.get('experiment')

    def __str__(self):
        if self.name is None:
            print('Please create or get an algorithm.')


class Bender():
    """Main class for bender"""
    def __init__(self, author, experiment=None):
        self.author = author
        self.algo = Algo(experiment)
        self.experiment = Experiment(experiment, algo=self.algo)

    def new_trial(self, parameters, results, comment=None):
        if (self.experiment.id is not None and self.algo.id is not None):
            data = {'experiment': self.experiment.id,
                    'author': self.author,
                    'parameters': parameters,
                    'results': results,
                    'comment': comment,
                    'algo': self.algo.id}
            r = requests.post(url='%s/trials/' % BASE_URL, json=data)
            if r.status_code == 201:
                data = r.json()
            else:
                return r
