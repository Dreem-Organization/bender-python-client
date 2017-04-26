from .utils import new_api_session, remove_saved_token


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

    def __init__(self, algo_id=None, experiment_id=None):
        self.session, self.username, self.user_id = new_api_session(url=self.BASE_URL)

        self.algo = None
        self.experiment = None

        if algo_id is not None:
            self.set_algo(algo_id=algo_id)

        if algo_id is None and experiment_id:
            self.set_experiment(experiment_id=experiment_id)

    def revoke_credentials(self):
        remove_saved_token()
        self.session = None
        self.username = None
        self.session, self.username = new_api_session(url=self.BASE_URL)

    def list_experiments(self):
        r = self.session.get(
            url='{}/api/experiments/?owner={}'.format(self.BASE_URL, self.username)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        print("\nExperiment list")
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

        print("\nShared Experiment list")
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
        self.algo = None

    def delete_experiment(self, experiment_id):
        r = self.session.delete(
            url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id)
        )
        if r.status_code != 204:
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
        if type(metrics) != list:
            if type(metrics) == str:
                metrics = [metrics]
            else:
                raise BenderError("Need to give a list of metrics: e.g: ['metric_1', 'metric_2']")

        r = self.session.post(
            url='{}/api/experiments/'.format(self.BASE_URL),
            json={
                'name': name,
                'description': description,
                'metrics': metrics,
                'dataset': dataset,
                'dataset_parameters': dataset_parameters},
        )

        if r.status_code == 201:
            self.set_experiment(r.json()["id"])
        else:
            raise BenderError('Failed to create experiment: {}'.format(r.content))

    def share_experiment(self, experiment_id, username):
        r = self.session.get(
            url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id),
        )
        if r.status_code != 200:
            raise BenderError('Could not retrieve experiment.')

        r = self.session.patch(
            url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id),
            json={
                    'shared_with' : [username],
            },
        )

    def list_algos(self):
        if self.experiment is None:
            raise BenderError("You need to set up an experiment.")

        r = self.session.get(
            url='{}/api/algos/?owner={}&&experiment={}'.format(
                self.BASE_URL, self.username, self.experiment.id)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        print("\nAlgo list")
        name = self.algo.name if self.algo else ""
        for algo in r.json()["results"]:
            print("  -{} {}: {}".format(
                ">" if algo["name"] == name else "",
                algo["name"],
                algo['id']))

    def set_algo(self, algo_id):
        r = self.session.get(
            url='{}/api/algos/{}/'.format(self.BASE_URL, algo_id),
        )
        if r.status_code != 200:
            raise BenderError('Could not retrieve algo.')
        data = r.json()
        if self.experiment is None or data["experiment"] != self.experiment.id:
            self.set_experiment(data["experiment"])
        self.algo = Algo(**r.json())

    def delete_algo(self, algo_id):
        r = self.session.delete(
            url='{}/api/algos/{}/'.format(self.BASE_URL, algo_id)
        )
        if r.status_code != 204:
            raise BenderError("Error: {}".format(r.content))
        else:
            print("algo deleted!")

    def new_algo(self, name, parameters, description=None, **kwargs):

        if self.experiment is None:
            raise BenderError("Set experiment!")

        r = self.session.post(
            url='{}/api/algos/'.format(self.BASE_URL),
            json={
                'name': name,
                'description': description,
                'parameters': parameters,
                'experiment': self.experiment.id
            }
        )

        if r.status_code != 201:
            raise BenderError('Failed to create experiment: {}'.format(r.content))
        self.set_algo(r.json()["id"])
        if self.algo.is_search_space_defined is False:
            print("Search space is not defined properly. Suggestion won't work.")

    def suggest(self, metric, is_loss, optimizer="parzen_estimator"):
        if self.algo is None:
            raise BenderError("Set experiment!")

        if metric not in self.experiment.metrics:
            raise BenderError("Metrics need to be in {}".format(self.experiments.metrics))

        if self.algo.is_search_space_defined is False:
            raise BenderError("Must define a search space properly.")

        r = self.session.post(
            url='{}/api/algos/{}/suggest/'.format(self.BASE_URL, self.algo.id),
            json={
                'metric': metric,
                'is_loss': is_loss,
                'optimizer': optimizer,
            }
        )
        if r.status_code != 200:
            raise BenderError('Failed to suggest trial: {}'.format(r.content))
        return r.json()

    def list_trials(self):
        if self.algo is None:
            raise BenderError("You need to set up an algo.")

        r = self.session.get(
            url='{}/api/trials/?algo={}'.format(self.BASE_URL, self.algo.id)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        return r.json()['results']

    def update_trial(self, trial_id, parameters=None, results=None, comment=None):
        # First we check that the trial id is valid
        r = self.session.get(
            url='{}/api/trials/{}/'.format(self.BASE_URL, trial_id),
        )
        if r.status_code != 200:
            raise BenderError("Invalid trial_id")
        # Then, we fill a dict that contains changing info 
        json = {}
        if parameters is not None:
            json['parameters'] = parameters        
        if results is not None:
            json['results'] = results
        if comment is not None:
            json['comment'] = comment
        # API request to update trial info
        r = self.session.patch(
            url='{}/api/trials/{}/'.format(self.BASE_URL, trial_id),
            json=json,
        )

    def list_k_best_trials(self, metric, is_loss, k=10, summary=True):
        if self.algo is None:
            raise BenderError("You need to set up an algo.")

        if metric not in self.experiment.metrics:
            raise BenderError("Metrics need to be in {}".format(self.experiment.metrics))

        r = self.session.get(
            url='{}/api/trials/?algo={}&&o_results={}{}&&limit={}'.format(
                self.BASE_URL,
                self.algo.id,
                "" if is_loss else "-",
                metric,
                k)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        if summary:
            results = [
                {
                    "parameters": result["parameters"],
                    "comment": result["comment"],
                    "results": result["results"],
                }
                for result in r.json()['results']
            ]
        else:
            results = r.json()['results']

        return results

    def set_trial(self, trial_id):
        r = self.session.get(
            url='{}/api/trials/{}/'.format(self.BASE_URL, trial_id),
        )
        if r.status_code != 200:
            raise BenderError('Could not retrieve trial.')
        data = r.json()
        if self.experiment is None or data["experiment"] != self.experiment.id:
            self.set_experiment(data["experiment"])
        self.trial = Trial(**r.json())

    def delete_trial(self, trial_id):
        r = self.session.delete(
            url='{}/api/trials/{}/'.format(self.BASE_URL, trial_id)
        )
        if r.status_code != 204:
            raise BenderError("Error: {}".format(r.content))
        else:
            print("trial deleted!")

    def new_trial(self, results, parameters, comment=None, **kwargs):
        if self.algo is None:
            raise BenderError("Set an algo.")

        r = self.session.post(
            url='{}/api/trials/'.format(self.BASE_URL),
            json={
                'algo': self.algo.id,
                'parameters': parameters,
                'results': results,
                'comment': comment,
            },
        )

        if r.status_code == 201:
            self.set_trial(r.json()["id"])
        else:
            raise BenderError('Failed to create experiment: {}'.format(set(r.content)))

        return r.json()["id"]


class Experiment():
    """Experiment class for Bender """

    def __init__(self,
                 id,
                 name,
                 description,
                 metrics,
                 owner,
                 dataset,
                 dataset_parameters,
                 **kwargs):
        self.id = id
        self.name = name
        self.description = description
        self.metrics = metrics
        self.dataset = dataset
        self.dataset_parameters = dataset_parameters

    def __repr__(self):
        return str(self.name)


class Algo():
    """ Algo class for Bender """

    def __init__(self, id, name, experiment, parameters, description, is_search_space_defined, **kwargs):
        self.id = id
        self.name = name
        self.parameters = parameters
        self.description = description
        self.is_search_space_defined = is_search_space_defined

    def __str__(self):
        return str(self.name)


class Trial():
    """ Trial class for bender """

    def __init__(self, parameters, results, comment, id, **kwargs):
        self.parameters = parameters
        self.results = results
        self.comment = comment
        self.id = id


class BenderError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error


if __name__ == "__main__":

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InZ0b3RvIiwiZXhwIjoxNTAxMTcyODc5LCJlbWFpbCI6InZhbGVudGluQHJ5dGhtLmNvIiwidXNlcl9pZCI6MTB9.lruHE-kxjsaaPEnJCXCYz84vYaNgfav3UczIMf33ms0"
    bender = Bender(token)
    bender.list_experiments()
    bender.set_experiment("066ca930-8e4d-4ce7-a142-77088a403347")
    bender.list_experiments()
    bender.list_algos()
    bender.set_algo("9106820f-5da0-4ad9-8eb0-8f951d09f05d")
    bender.list_experiments()
    bender.list_algos()
    bender.new_trial(results={"loss": 2}, parameters={"param1": 1, "param2": 2})
