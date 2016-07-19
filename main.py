from NetworkBuilder import NetworkBuilder
from ResultsManager import ResultsManager

nb = NetworkBuilder(input_size=[40, 40, 3], n_class=2)
nb.build_network()

