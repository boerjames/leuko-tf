from DataLoader import DataLoader
from NetworkBuilder import NetworkBuilder
from NetworkTrainer import NetworkTrainer
from ResultsManager import ResultsManager

data_shape = [40, 40, 3]

dl = DataLoader(path="/tmp/data", input_shape=data_shape, percent_train=0.8, verbose=False, save_data=False)
data = dl.load_images()
print(data["train_images"].shape)
print(data["train_labels"].shape)
print(data["test_images"].shape)
print(data["test_labels"].shape)
print(data["n_images"])
print()

nb = NetworkBuilder(input_shape=data_shape, n_class=2, verbose=False)
network = nb.build_network()
print(network["x"])
print(network["y"])
print(network["prediction"])
print()

nt = NetworkTrainer(data=data)
nt.train_network(network=network, learning_rate=0.001, training_epochs=100, batch_size=100)