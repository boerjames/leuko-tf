# Example of the main script which trains a bunch of CNNs
# Usage: python run_experiments.py

from DataLoader import DataLoader
from NetworkBuilder import NetworkBuilder
from NetworkTrainer import NetworkTrainer
from ResultsManager import ResultsManager

data_shape = [40, 40, 3]
verbose = True

dl = DataLoader(path="/tmp/data",
                data_shape=data_shape,
                percent_train=0.8,
                verbose=verbose,
                save_data=False)

data = dl.load_images()
n_class = data["n_class"]

nb = NetworkBuilder(input_shape=data_shape,
                    n_class=n_class,
                    verbose=verbose)


nt = NetworkTrainer(data=data)


rm = ResultsManager(database_name=':memory:', battery_name='test_battery')

for x in range(1):
    network = nb.build_network()
    print(network["x"])
    print(network["y"])
    print(network["prediction"])
    print()

    train_accuracy, test_accuracy = nt.train_network(network=network,
                                                     optimization_algorithm='rmsprop',
                                                     learning_rate=0.001,
                                                     training_epochs=150,
                                                     early_stop=30,
                                                     batch_size=256)

    rm.add(train_accuracy, test_accuracy)

rm.export()