from collections import Counter
from typing import cast
from time import time

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx

from agent import CovidAgent, InfectionState
from variant import CovidVariant


class CovidModel(Model):
    G: nx.Graph
    grid: NetworkGrid
    infection_prob: float
    recovery_prob: float
    death_prob: float
    gain_resistance_prob: float
    resistance_level: float
    mutation_prob: float

    variant_code_map: dict[str, CovidVariant]

    def __init__(
        self,
        num_nodes=500,
        avg_degree=10,
        infection_prob=0.1,
        recovery_prob=0.05,
        death_prob=0.001,
        gain_resistance_prob=0.01,
        resistance_level=1.0,
        mutation_prob=0.01,
    ):
        """
        Initializes the COVID model.

        :param num_nodes: The number of nodes in the model
        :param avg_degree: The average degree of nodes in the model
        :param infection_prob: Probability of an agent infecting a
        connected agent during a single time step
        :param recovery_prob: The chance that an infected agent recovers
        during a single time step
        :param death_prob: The chance that an infected agent dies during a
        single time step
        :param gain_resistance_prob: The chance that a susceptible agent will
        gain resistance during a single time step (e.g. through vaccination)
        :param resistance_level: The probability that a resistant agent will
        resist an infection relative to a susceptible agent. Can be interpreted
        as vaccine efficacy/protection against re-infection
        :param mutation_prob: The probability that a given bit of a virus'
        genetic code will flip
        """

        super().__init__()
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.death_prob = death_prob
        self.gain_resistance_prob = gain_resistance_prob
        self.resistance_level = resistance_level
        self.mutation_prob = mutation_prob

        self.schedule = RandomActivation(self)
        edge_probability = avg_degree / num_nodes
        self.G = nx.erdos_renyi_graph(num_nodes, edge_probability)
        self.grid = NetworkGrid(self.G)

        self.datacollector = DataCollector(
            {
                "Susceptible": "num_susceptible",
                "Infected": "num_infected",
                "Resistant": "num_resistant",
                "Dead": "num_dead",
                "Variants": "variant_frequency",
            }
        )

        # Initialize agents
        for i, node in enumerate(self.G.nodes):
            agent: CovidAgent
            if i == 0:
                variant = CovidVariant(self, self.infection_prob, self.death_prob)
                self.variant_code_map = {
                    variant.genetic_code.get_bitvector_in_hex(): variant
                }
                agent = CovidAgent(i, self, InfectionState.INFECTED, variant)
            else:
                agent = CovidAgent(i, self, InfectionState.SUSCEPTIBLE)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.variant_file_name = f"output/variant_data-{int(time())}.csv"
        with open(self.variant_file_name, "w") as f:
            f.write("Variant,Time Step,Cases,Infectivity Rate,Death Rate\n")

    def step(self) -> None:
        self.datacollector.collect(self)
        self.update_variant_csv()
        self.schedule.step()

    def update_variant_csv(self):
        """
        A hacky way to dump variant data to a file so we can visualize it elsewhere
        """
        with open(self.variant_file_name, "a") as f:
            for variant, frequency in self.variant_frequency:
                f.write(
                    f"{variant.name},{self.schedule.steps},{frequency},{variant.base_infection_prob:3.2f},{variant.base_death_prob:3.2f}\n"
                )
        pass

    @property
    def num_susceptible(self) -> int:
        return self.num_with_state(InfectionState.SUSCEPTIBLE)

    @property
    def num_infected(self) -> int:
        return self.num_with_state(InfectionState.INFECTED)

    @property
    def num_resistant(self) -> int:
        return self.num_with_state(InfectionState.RESISTANT)

    @property
    def num_dead(self) -> int:
        return self.num_with_state(InfectionState.DEAD)

    def num_with_state(self, state: InfectionState) -> int:
        return sum(1 for agent in self.agents if agent.state == state)

    @property
    def agents(self) -> list[CovidAgent]:
        return cast(list[CovidAgent], self.grid.get_all_cell_contents())

    @property
    def variant_frequency(self) -> list[tuple[CovidVariant, int]]:
        counter: dict[CovidVariant, int] = Counter(
            (
                agent.infection_variant
                for agent in self.agents
                if agent.state == InfectionState.INFECTED
            )
        )
        return cast(
            list[tuple[CovidVariant, int]],
            sorted(counter.items(), key=lambda t: t[1], reverse=True),
        )

    @property
    def summary(self) -> str:
        return (
            f"Susceptible: {self.num_susceptible}"
            f"\nInfected: {self.num_infected}"
            f"\nResistant: {self.num_resistant}"
            f"\nDead: {self.num_dead}"
        )
