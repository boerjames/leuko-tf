from NetworkBuilder import NetworkBuilder
from ResultsManager import ResultsManager
import DataLoader

nb = NetworkBuilder(input_size=[40, 40, 3], n_class=2)
input, output, network = nb.build_network()

