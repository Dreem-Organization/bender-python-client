from .utils import new_api_session, remove_saved_token
import urllib


class Bender:

    BASE_URL = 'https://bender-api.dreem.com'

    def __repr__(self):
        return "Bending Unit {}".format(self.user_id)

    @staticmethod
    def _say_hello():
        return "Bite my shinny metal ass!"

    def __init__(self, algo_id=None, experiment_id=None):
        self.session, self.username, self.user_id = new_api_session(url=self.BASE_URL)

        self.algo = None
        self.experiment = None

        if algo_id is not None:
            self.set_algo(algo_id=algo_id)

        if algo_id is None and experiment_id:
            self.set_experiment(experiment_id=experiment_id)

    def list_experiments(self):
        r = self.session.get(
            url='{}/api/experiments/?owner={}'.format(self.BASE_URL, self.username)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        experiment_list = []
        for experiment in r.json()["results"]:
            experiment_list.push({"name": experiment["name"], "id": experiment["id"]})
        return experiment_list

    def set_experiment(self, name=None, experiment_id=None):
        data = None
        if experiment_id is not None:
            r = self.session.get(
                url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id),
            )
            if r.status_code != 200:
                raise BenderError('Could not retrieve experiment.')
            data = r.json()

        elif name is not None:
            r = self.session.get(
                url='{}/api/experiments/?owner={}&name={}'.format(
                    self.BASE_URL,
                    self.username,
                    urllib.parse.quote(name)
                )
            )
            if r.status_code != 200 or r.json()["count"] != 1:
                raise BenderError('Could not retrieve experiment.')
            data = r.json()["results"][0]

        else:
            raise BenderError("Provide a name or experiment_id!")

        self.experiment = Experiment(**data)
        self.algo = None

    def create_experiment(
        self,
        name,
        metrics,
        description=None,
        dataset=None,
        dataset_parameters=None,
        **kwargs
        ):

        r = self.session.get(
            url='{}/api/experiments/?owner={}&name={}'.format(self.BASE_URL,
                                                              self.username,
                                                              urllib.parse.quote(name)
                                                              )
        )

        if r.status_code == 200 and r.json()["count"] == 1:
            self.set_experiment(experiment_id=r.json()["results"][0]["id"])
            print("Experiment already exist with that name and is now set as the current experiment.")
        else:
            if type(metrics) != list:
                raise BenderError("Need to give a list of metrics.")

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
                self.set_experiment(experiment_id=r.json()["id"])
            else:
                raise BenderError('Failed to create experiment: {}'.format(r.content))

    def get_experiment(self):
        return self.experiment

    def delete_experiment(self, name=None, experiment_id=None):
        if experiment_id is not None:
            r = self.session.delete(
                url='{}/api/experiments/{}/'.format(self.BASE_URL, experiment_id)
            )
            if r.status_code != 204:
                raise BenderError("Error: {}".format(r.content))
            else:
                print("Experiment deleted!")

        elif name is not None:
            r = self.session.delete(
                url='{}/api/experiments/?owner={}&name={}'.format(
                    self.BASE_URL,
                    self.username,
                    urllib.parse.quote(name)
                )
            )
            if r.status_code != 204:
                raise BenderError("Error: {}".format(r.content))
            else:
                print("Experiment deleted!")

        else:
            raise BenderError("Provide a name or experiment_id!")

    def list_algos(self):
        if self.experiment is None:
            raise BenderError("You need to set up an experiment.")

        r = self.session.get(
            url='{}/api/algos/?experiment={}'.format(
                self.BASE_URL, self.experiment.id)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))
        
        algo_list = []
        for algo in r.json()["results"]:
            algo_list.push({"name": algo["name"], "id": algo["id"]})
        return algo_list

    def set_algo(self, name=None, algo_id=None):
        r = None
        if algo_id is not None:
            r = self.session.get(
            url='{}/api/algos/{}/'.format(self.BASE_URL, algo_id),
            )
            if r.status_code != 200:
                raise BenderError('Could not retrieve algo.')
            data = r.json()

        elif name is not None:
            r = self.session.get(
                url='{}/api/algos/?experiment={}&name={}'.format(
                    self.BASE_URL,
                    self.experiment.id,
                    urllib.parse.quote(name)
                )
            )
            if r.status_code != 200 or r.json()["count"] != 1:
                raise BenderError('Could not retrieve algo.')
            data = r.json()["results"][0]

        else:
            raise BenderError("Provide a name or algo_id!")

        if self.experiment is None or data["experiment"] != self.experiment.id:
            self.set_experiment(experiment_id=data["experiment"])
        self.algo = Algo(**r.json())
    
    def create_algo(self, name, hyper_parameters, description=None, **kwargs):
        if self.experiment is None:
            raise BenderError("Set experiment!")

        r = self.session.get(
            url='{}/api/algos/?experiment={}&name={}'.format(
                self.BASE_URL,
                self.experiment.id,
                urllib.parse.quote(name)
            )
        )

        if r.status_code == 200 and r.json()["count"] == 1:
            self.set_algo(algo_id=r.json()["results"][0]["id"])
            print("Algo already exist with that name and is now set as the current algo.")
        else:
            r = self.session.post(
                url='{}/api/algos/'.format(self.BASE_URL),
                json={
                    'name': name,
                    'description': description,
                    'parameters': hyper_parameters,
                    'experiment': self.experiment.id
                }
            )

            if r.status_code != 201:
                raise BenderError('Failed to create experiment: {}'.format(r.content))
            self.set_algo(r.json()["id"])
            if self.algo.is_search_space_defined is False:
                print("Search space is not defined properly. Suggestion won't work.")
            
    def get_algo(self):
        return self.algo

    def delete_algo(self, name=None, algo_id=None):
        r = None
        if algo_id is not None:
            r = self.session.delete(
            url='{}/api/algos/{}/'.format(self.BASE_URL, algo_id),
            )
            if r.status_code != 200:
                raise BenderError('Could not retrieve algo.')
            data = r.json()

        elif name is not None:
            r = self.session.delete(
                url='{}/api/algos/?experiment={}&name={}'.format(
                    self.BASE_URL,
                    self.experiment.id,
                    urllib.parse.quote(name)
                )
            )
            if r.status_code != 200 or r.json()["count"] != 1:
                raise BenderError('Could not retrieve algo.')
            data = r.json()["results"][0]

        else:
            raise BenderError("Provide a name or algo_id!")

        if r.status_code != 204:
            raise BenderError("Error: {}".format(r.content))
        else:
            print("algo deleted!")

    def list_trials(self):
        if self.algo is None:
            raise BenderError("You need to set up an algo.")

        r = self.session.get(
            url='{}/api/trials/?algo={}'.format(self.BASE_URL, self.algo.id)
        )
        if r.status_code != 200:
            raise BenderError("Error: {}".format(r.content))

        data = r.json()
        results = data["results"]
        while True:
            if data["next"] is not None:
                data = self.session.get(data["next"]).json()
                results.extend(data["results"])
            else:
                break

        return results

    def create_trial(self, results, hyper_parameters, weight=1, comment=None, **kwargs):
        if self.algo is None:
            raise BenderError("Set an algo.")

        r = self.session.post(
            url='{}/api/trials/'.format(self.BASE_URL),
            json={
                'algo': self.algo.id,
                'parameters': hyper_parameters,
                'results': results,
                'comment': comment,
                'weight': 1,
            },
        )

        if r.status_code != 201:
            raise BenderError('Failed to create trial: {}'.format(r.content))

    def delete_trial(self, trial_id):
        r = self.session.delete(
            url='{}/api/trials/{}/'.format(self.BASE_URL, trial_id)
        )
        if r.status_code != 204:
            raise BenderError("Error: {}".format(r.content))
        else:
            print("Trial deleted!")
    
    def suggest(self, metric, optimizer="parzen_estimator"):
        if self.algo is None:
            raise BenderError("Set experiment!")

        if any(len(m["metric_name"]) == metric for m in self.experiment.metrics):
            raise BenderError("Metrics need to be in {}".format(self.experiment.metrics))

        if self.algo.is_search_space_defined is False:
            raise BenderError("Must define a search space properly.")

        r = self.session.post(
            url='{}/api/algos/{}/suggest/'.format(self.BASE_URL, self.algo.id),
            json={
                'metric': metric,
                'optimizer': optimizer,
            }
        )
        if r.status_code != 200:
            raise BenderError('Failed to suggest trial: {}'.format(r.content))
        return r.json()

    def revoke_credentials(self):
        remove_saved_token()
        self.session = None
        self.username = None
        self.session, self.username, self.user_id = new_api_session(url=self.BASE_URL)


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


class Algo:
    """ Algo class for Bender """

    def __init__(self, id, name, experiment, parameters, description, is_search_space_defined, **kwargs):
        self.id = id
        self.name = name
        self.parameters = parameters
        self.description = description
        self.is_search_space_defined = is_search_space_defined

    def __repr__(self):
        return str(self.name)


class BenderError(Exception):
    def __init__(self, error):
        self.error = error

    def __repr__(self):
        return self.error


if __name__ == "__main__":
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InZ0b3RvIiwiZXhwIjoxNTAxMTcyODc5LCJlbWFpbCI6InZhbGVudGluQHJ5dGhtLmNvIiwidXNlcl9pZCI6MTB9.lruHE-kxjsaaPEnJCXCYz84vYaNgfav3UczIMf33ms0"
    bender = Bender(token)
    bender.list_experiments()
    bender.set_experiment(experiment_id="066ca930-8e4d-4ce7-a142-77088a403347")
    bender.list_experiments()
    bender.list_algos()
    bender.set_algo("9106820f-5da0-4ad9-8eb0-8f951d09f05d")
    bender.list_experiments()
    bender.list_algos()
    bender.new_trial(results={"loss": 2}, parameters={"param1": 1, "param2": 2})
