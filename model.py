from mesa import Model
import networkx as nx


class CovidModel(Model):
    network: nx.Graph

    def __init__(self, num_nodes=500, avg_degree=3, recovery_chance=0.01):
        """
        Initializes the COVID model.

        :param num_nodes: The number of nodes in the model
        :param avg_degree: The average degree of nodes in the model
        :param recovery_chance: The chance that an infected node recovers during a given time step
        """
        self.recovery_chance = recovery_chance
        
        self.network = nx.erdos_renyi_graph(num_nodes, avg_degree)


