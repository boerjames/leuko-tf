from NetworkBuilder import NetworkBuilder
from ResultsManager import ResultsManager
from DataLoader import DataLoader

data_shape = [40, 40, 3]

nb = NetworkBuilder(input_shape=data_shape, n_class=2)
network = nb.build_network()

dl = DataLoader(path="/tmp/data", input_shape=data_shape, percent_train=0.8, save_data=True)
dl.load_images()