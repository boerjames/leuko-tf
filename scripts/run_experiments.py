# Example of the main script which trains a bunch of CNNs
# Usage: python run_experiments.py

import os
from DataLoader import DataLoader
from NetworkBuilder import NetworkBuilder
from NetworkTrainer import NetworkTrainer
from ResultsManager import ResultsManager

def run():
    n_experiments = 1
    data_shape = [40, 40, 3]
    verbose = True
    save_data = True
    save_path = os.path.join(os.pardir, 'save_data/')

    dl = DataLoader(path="/tmp/data",
                    data_shape=data_shape,
                    percent_train=0.8,
                    normalize=False,
                    verbose=verbose,
                    save_data=save_data,
                    save_path=save_path)

    data = dl.load_images()
    n_class = data["n_class"]

    nb = NetworkBuilder(input_shape=data_shape,
                        n_class=n_class,
                        verbose=verbose)


    nt = NetworkTrainer(data=data,
                        verbose=verbose)


    rm = ResultsManager(database_name=':memory:',
                        battery_name='test_battery',
                        save_path=save_path)

    for exp_id in range(n_experiments):
        if verbose:
            print("RUNNING EXPERIMENT {}".format(exp_id))

        network = nb.build_network()

        train_accuracy, test_accuracy = nt.train_network(network=network,
                                                         optimization_algorithm='rmsprop',
                                                         learning_rate=0.001,
                                                         training_epochs=20,
                                                         early_stop=30,
                                                         batch_size=256)

        rm.add(exp_id, train_accuracy, test_accuracy)

    rm.export()

if __name__ == '__main__':
    run()