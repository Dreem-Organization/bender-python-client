import requests
import json

CURRENT_EXPERIMENT = None
BASE_URL = "http://127.0.0.1:8000"


class Experiment():
    def __init__(self, name, description, metrics, dataset, dataset_parameters):
        self.data = {
            'name': name,
            'description': description,
            'metrics': json.dumps(metrics),
            'dataset': dataset,
            'dataset_parameters': json.dumps(dataset_parameters)
        }
        self.id = self.create(self.data)

    def create(self, data):
        print(data)
        r = requests.post(url='%s/experiments/' % (BASE_URL), data=data)
        if r.status_code == 201:
            print('Created experiment "%s" with id:%s' % (r.json()['name'], r.json()['id']))
            return r.json()['id']
        else:
            print('Failed to create experiment.')
            return

    def new_algo(self, name, parameters):
        data = {'name': name,
                'parameters': json.dumps(parameters),
                'experiment': self.id}
        r = requests.post(url='%s/algos/' % BASE_URL, data=data)
        if r.status_code == 201:
            data = r.json()
            return Algo(id=data['id'], name=data['name'], experiment=self.id)


class Algo():
    def __init__(self, id, name, experiment):
        self.id = id
        self.name = name
        self.experiment = experiment

    def new_trial(self, experiment, author, parameters, results, comment=None):
        data = {'experiment': self.experiment,
                'author': author,
                'parameters': json.dumps(parameters),
                'results': json.dumps(results),
                'comment': comment,
                'algo': self.id}
        r = requests.post(url='%s/trials/' % BASE_URL, data=data)
        if r.status_code == 201:
            data = r.json()
        else:
            return r
