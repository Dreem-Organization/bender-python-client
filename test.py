import unittest
import json
from benderclient import Bender
bender = Bender()

exp = 'pre-comit-test-script-experiment'
alg = 'pre-comit-test-script-algo'


class BenderTest(unittest.TestCase):
    def test_a_create_experiment(self):
        print("-> Creating experiment...")
        bender.create_experiment(
            name=exp,
            description='NA',
            metrics=[{"metric_name": "a", "type": "reward"}],
            dataset='NA'
        )
        print("-> Creating same experiment...")
        bender.create_experiment(
            name=exp,
            description='NA',
            metrics=[{"metric_name": "a", "type": "reward"}],
            dataset='NA'
        )

    def test_b_list_experiments(self):
        print("-> Listing experiments...")
        experiments = bender.list_experiments()
        self.assertEqual(type(experiments), list)

    def test_c_set_experiment(self):
        print("-> Set experiment by name...")
        bender.set_experiment(name=bender.get_experiment().name)
        print("-> Set experiment by id...")
        bender.set_experiment(experiment_id=bender.get_experiment().id)

    def test_d_create_algo(self):
        print("-> Creating algo...")
        bender.create_algo(
            name=alg,
            hyperparameters=[{"name": "NA", "category": "categorical",
                              "search_space": {"values": [3, 5, 7]}}],
            description='NA'
        )
        print("-> Creating same algo...")
        bender.create_algo(
            name=alg,
            hyperparameters=[{"name": "NA", "category": "categorical",
                              "search_space": {"values": [3, 5, 7]}}],
            description='NA'
        )

    def test_d_list_algos(self):
        print("-> Listing Algos...")
        algos = bender.list_algos()
        self.assertEqual(type(algos), list)

    def test_e_set_algo(self):
        print("-> Set algo by name...")
        bender.set_algo(name=bender.get_algo().name)
        print("-> Set algo by id...")
        bender.set_algo(algo_id=bender.get_algo().id)

    def test_f_create_trial(self):
        print("-> Creating trial...")
        bender.create_trial(
            results={'a': 1},
            hyperparameters={'NA': 3},
            comment='NA'
        )

    def test_g_list_trials(self):
        print("-> Listing Trials...")
        trials = bender.list_trials()
        self.assertEqual(type(trials), list)

    def test_h_suggest(self):
        print("-> Getting suggestion...")
        bender.suggest(metric="a")

    def test_i_delete_trial(self):
        print("-> Delete trial by id...")
        bender.delete_trial(bender.list_trials()[0]['id'])

    def test_j_delete_algo(self):
        print("-> Delete algo by id...")
        bender.delete_algo(bender.get_algo().id)

    def test_k_delete_experiment(self):
        print("-> Delete experiment by id...")
        bender.delete_experiment(bender.get_experiment().id)


if __name__ == '__main__':
    unittest.main()
