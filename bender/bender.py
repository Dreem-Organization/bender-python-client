import requests


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

    BASE_URL = 'https://bender-api.rythm.co'

    def __repr__(self):
        return "Bending Unit {}".format(self.user_id)

    def _say_hello(self):
        return "Bite my shinny metal ass!"

    def __init__(self, token, algo_id=None, experiment_id=None):
        r = requests.get(
            url='{}/user/'.format(self.BASE_URL),
            headers={"Authorization": "JWT {}".format(token)}
        )
        if r.status_code != 200:
            raise BenderError("Invalid token, check {} for more informations".format(self.BASE_URL))

        self.username = r.json()["username"]
        self.user_id = r.json()["pk"]
        self.session = requests.Session()
        self.session.headers.update({'Authorization': 'JWT {}'.format(token)})

        self.algo = None
        if algo_id is not None:
            self.set_algo(algo_id=algo_id)

        self.experiment = None
        if algo_id is None and experiment_id:
            self.set_experiment(experiment_id=experiment_id)

    def list_experiments(self):
        r = self.session.get(
            url='{}/api/experiments/?owner={}'.format(self.BASE_URL, self.username)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        print("Experiment list")
        name = self.experiment.name if self.experiment else ""
        for experiment in r.json()["results"]:
            print("  -{} {}: {}".format(
                ">" if experiment["name"] == name else "",
                experiment["name"],
                experiment['id']))

    def list_shared_experiments(self):
        r = self.session.get(
            url='{}/api/experiments/?shared_experiments={}'.format(self.BASE_URL, self.username)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        print("Shared Experiment list")
        name = self.experiment.name if self.experiment else ""
        for experiment in r.json()["results"]:
            print("  -{} {}: {}".format(
                ">" if experiment["name"] == name else "",
                experiment["name"],
                experiment['id']))

    def set_experiment(self, experiment_id):
        r = self.session.get(
            url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id),
        )
        if r.status_code != 200:
            raise BenderError('Could not retrieve experiment.')
        self.experiment = Experiment(**r.json())

    def delete_experiment(self, experiment_id):
        r = self.session.delete(
            url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))
        else:
            print("Experiment deleted!")

    def new_experiment(self,
                       name,
                       metrics,
                       description=None,
                       dataset=None,
                       dataset_parameters=None,
                       **kwargs):
        """
        Please provide the following:
        - name: string
        - description: string
        - metrics: list of strings
        - dataset: string
        - dataset_parameters: dict
        - author: string
        """
        if type(metrics) != list:
            if type(metrics) == str:
                metrics = [metrics]
            else:
                raise BenderError("Need to give a list of metrics: e.g: ['metric_1', 'metric_2']")

        r = self.session.post(
            url='{}/experiments/'.format(self.BASE_URL),
            json={
                'name': name,
                'description': description,
                'metrics': metrics,
                'dataset': dataset,
                'dataset_parameters': dataset_parameters},
        )

        if r.status_code == 201:
            self.set_experiment(r.json()["pk"])
        else:
            raise BenderError('Failed to create experiment: {}'.format(r.content))


class Experiment():
    """Experiment class for Bender """

    def __init__(self, id, name, description, metrics, owner, dataset, dataset_parameters, **kwargs):
        self.id = id
        self.name = name
        self.description = description
        self.metrics = metrics
        self.dataset = dataset
        self.dataset_parameters = dataset_parameters

    def __repr__(self):
        self.name


class Algo():
    """ Algo class for Bender """

    def __init__(self, algo_id):
        if algo_id is not None:
            self.get(algo_id)

    def populate(self, data):
        self.id = data.get('id')
        self.name = data.get('name')
        self.experiment_id = data.get('experiment')
        self.parameters = data.get('parameters')
        self.description = data.get('description')

    def get_latest_used_algo(self, algo_id):
        r = self.session.get(
            url='{}/algos/{}/'.format(self.BASE_URL, algo_id),
        )

        if r.status_code == 200:
            self.populate(r.json()[0])
        else:
            raise BenderError('Could not retrieve algo.')

    def create(self, name, parameters, description={}):
        """
        Please provide the following:
        - name: string
        - parameters: list of dict
        """
        if type(parameters) != list or len(parameters) == 0:
            raise BenderError("Need to give a list of parameters see doc for more info.")
        else:
            for parameter in parameters:
                if type(parameter) != dict:
                    raise BenderError("Need to give a list of parameters see doc for more info.")

        if type(description) != dict:
            raise BenderError("Need to give a dict object.")

        data = {
            'name': name,
            'parameters': parameters,
            'description': description
        }

        r = self.session.post(url='{}/algos/'.formatself.BASE_URL, json=data,
                              )

        if r.status_code == 201:
            self.populate(r.json())
        else:
            raise BenderError('Could not create Algo: {}'.format(r.content))

    def get(self, algo_id):
        """Retrieve algo instance"""
        r = self.session.get(url='{}/algos/{}/'.format(self.BASE_URL, algo_id),
                             )
        if r.status_code == 200:
            self.populate(r.json())
        else:
            raise BenderError('Could not retrieve experiment.')

    def __str__(self):
        if self.name is None:
            raise BenderError('Please create or get an algorithm.')
        else:
            return self.name


class Trial():
    """ Trial class for bender """

    def __init__(self, trial_id):
        if trial_id is not None:
            self.get(trial_id)

    def populate(self, data):
        self.parameters = data.get('parameters')
        self.results = data.get('results')
        self.comment = data.get('comment')
        self.id = data.get('id')

    def new(self, parameters, results, comment=None):
        if (self.experiment is not None and self.algo is not None):
            if (len(set(self.algo.parameters) & set(parameters.keys())) == len(self.algo.parameters)
                    and len(set(self.experiment.metrics) & set(results.keys())) == len(self.experiment.metrics)):
                r = self.session.post(
                    url='{}/trials/'.formatself.BASE_URL,
                    json={'algo': self.algo.id,
                          'parameters': parameters,
                          'results': results,
                          'comment': comment},
                    headers={"Authorization": "Bearer {}".format(self.token)}
                )
                if r.status_code == 201:
                    self.populate(data=r.json())
                    print('Trial successfully send.')
                    return r.json()

            raise BenderError(
                "Could not send trial.\nPlease make sure you provided the following:\
                \nParameters: {}.\nResults: {}."
                .format(', '.join(self.algo.parameters), ', '.join(self.experiment.metrics))
            )
        else:
            raise BenderError('You must provide an Experiment and Algo before sending new trials.')


class BenderError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error


if __name__ == "__main__":

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InZ0b3RvIiwiZXhwIjoxNTAxMTcyODc5LCJlbWFpbCI6InZhbGVudGluQHJ5dGhtLmNvIiwidXNlcl9pZCI6MTB9.lruHE-kxjsaaPEnJCXCYz84vYaNgfav3UczIMf33ms0"
    bender = Bender(token)
    bender.list_experiments()
    bender.set_experiment("77d68410-2279-4746-8939-79dc71fbf876")
