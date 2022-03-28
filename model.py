from typing import Any

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx

from project.agent import InfectionState, CovidAgent


class CovidModel(Model):
    network: nx.Graph
    infection_prob: float
    recovery_prob: float

    def __init__(
        self,
        num_nodes=500,
        avg_degree=3,
        infection_prob=0.1,
        recovery_prob=0.01,
        *args: Any,
        **kwargs: Any
    ):
        """
        Initializes the COVID model.

        :param num_nodes: The number of nodes in the model
        :param avg_degree: The average degree of nodes in the model
        :param infection_prob: Probability of an agent infecting a connected agent during a single time step
        :param recovery_chance: The chance that an infected node recovers during a single time step
        """

        super().__init__(*args, **kwargs)
        self.recovery_prob = recovery_prob
        self.infection_prob = infection_prob

        self.schedule = RandomActivation(self)
        self.network = nx.erdos_renyi_graph(num_nodes, avg_degree)
        self.grid = NetworkGrid(self.network)

        self.datacollector = DataCollector(
            {
                "Susceptible": "num_susceptible",
                "Infected": "num_infected",
            }
        )

        # Initialize agents
        for i, node in enumerate(self.network.nodes):
            # Infect only one node
            infection_state = (
                InfectionState.SUSCEPTIBLE if i > 0 else InfectionState.INFECTED
            )
            agent = CovidAgent(i, self, infection_state)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.datacollector.collect(self)

    def step(self) -> None:
        pass

    @property
    def num_susceptible(self) -> int:
        return self.num_with_state(InfectionState.SUSCEPTIBLE)

    @property
    def num_infected(self) -> int:
        return self.num_with_state(InfectionState.INFECTED)

    def num_with_state(self, state: InfectionState) -> int:
        # TODO: Figure out how to implement this
        pass
