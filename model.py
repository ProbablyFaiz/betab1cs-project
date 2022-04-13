from collections import Counter, defaultdict
import csv
from typing import cast
from time import time

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx

from agent import CovidAgent, InfectionState
from variant import CovidVariant

DEFAULT_MODEL_PARAMS = {
    "num_nodes": 500,
    "avg_degree": 10,
    "infection_prob": 0.1,
    "recovery_prob": 0.05,
    "death_prob": 0.001,
    "gain_resistance_prob": 0.01,
    "resistance_level": 1.0,
    "mutation_prob": 0.001,
    "genome_bits": 4,
}


class CovidModel(Model):
    G: nx.Graph
    grid: NetworkGrid
    infection_prob: float
    recovery_prob: float
    death_prob: float
    gain_resistance_prob: float
    resistance_level: float
    mutation_prob: float
    genome_bits: float

    variant_code_map: dict[str, CovidVariant]
    variant_freq_series: list[list[tuple[CovidVariant, int]]]
    variant_immunity_series: list[dict[CovidVariant, float]]

    dump_data: bool

    def __init__(
        self,
        num_nodes=DEFAULT_MODEL_PARAMS["num_nodes"],
        avg_degree=DEFAULT_MODEL_PARAMS["avg_degree"],
        infection_prob=DEFAULT_MODEL_PARAMS["infection_prob"],
        recovery_prob=DEFAULT_MODEL_PARAMS["recovery_prob"],
        death_prob=DEFAULT_MODEL_PARAMS["death_prob"],
        gain_resistance_prob=DEFAULT_MODEL_PARAMS["gain_resistance_prob"],
        resistance_level=DEFAULT_MODEL_PARAMS["resistance_level"],
        mutation_prob=DEFAULT_MODEL_PARAMS["mutation_prob"],
        genome_bits=DEFAULT_MODEL_PARAMS["genome_bits"],
        dump_variant_data=False,
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
        self.genome_bits = genome_bits
        self.dump_data = dump_variant_data

        self.schedule = RandomActivation(self)
        edge_probability = avg_degree / num_nodes
        self.G = nx.erdos_renyi_graph(num_nodes, edge_probability)
        self.grid = NetworkGrid(self.G)

        self.start_time = int(time() * 1000)
        self.variant_freq_series = []
        self.variant_immunity_series = []

        self.datacollector = DataCollector(
            {
                "Susceptible": "num_susceptible",
                "Infected": "num_infected",
                "Resistant": "num_resistant",
                "Dead": "num_dead",
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

    def step(self) -> None:
        self.datacollector.collect(self)
        if self.dump_data:
            self.variant_freq_series.append(self.variant_frequency)
            self.variant_immunity_series.append(self.variant_immunity_levels)
            if self.schedule.steps > 0 and self.schedule.steps % 50 == 0:
                self.dump_variant_data()
        self.schedule.step()

    def dump_variant_data(self):
        """
        A hacky way to dump variant data to a file so we can visualize it elsewhere
        """
        info_file_name = f"output/variant-info-{self.start_time}.csv"
        with open(info_file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Variant Code",
                    "Maximum Cases",
                    "Infection Probability",
                    "Death Probability",
                    "Avg. Immunity at First Case",
                ]
            )
            # Tracks the maximum number of cases for each variant and the time step at which it occurred
            max_cases_dict = defaultdict(lambda: (-1, -1))
            first_case_step_dict = {}
            for i, freq_info in enumerate(self.variant_freq_series):
                for variant, frequency in freq_info:
                    if variant not in first_case_step_dict:
                        first_case_step_dict[variant] = i
                    if frequency > max_cases_dict[variant][0]:
                        max_cases_dict[variant] = (frequency, i)
            for variant, (max_cases, max_case_step) in max_cases_dict.items():
                # Get the time step of the first case of the variant
                writer.writerow(
                    [
                        variant.name,
                        max_cases,
                        f"{variant.base_infection_prob:4.3f}",
                        f"{variant.base_death_prob:5.4f}",
                        f"{self.variant_immunity_series[first_case_step_dict[variant]][variant]:5.4f}",
                    ]
                )
            f.flush()
        time_series_file_name = f"output/variant-time-series-{self.start_time}.csv"
        variant_order = sorted(
            [variant for variant in max_cases_dict.keys()],
            key=lambda v: int(v.name, 16),
        )
        with open(time_series_file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Time Step"] + [v.name for v in variant_order])
            for i, freq_info in enumerate(self.variant_freq_series):
                curr_row = [i]
                freq_info_dict = defaultdict(lambda: 0, freq_info)
                for variant in variant_order:
                    curr_row.append(freq_info_dict[variant])
                writer.writerow(curr_row)
            f.flush()

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
    def variant_immunity_levels(self) -> dict[CovidVariant, float]:
        """
        Tracks average level of immunity for each variant during the current time step.
        Very expensive to compute ( O(num_nodes*num_variants) ), but valuable to collect.
        """
        variant_immunity_levels = {}
        total_immunity = 0.0
        for variant in self.variant_code_map.values():
            avg_immunity = sum(
                agent.resistance_level(variant) for agent in self.agents
            ) / len(self.agents)
            variant_immunity_levels[variant] = avg_immunity
            total_immunity += avg_immunity
        avg_immunity = total_immunity / len(self.variant_code_map)
        for variant, immunity in variant_immunity_levels.items():
            variant_immunity_levels[variant] = immunity / avg_immunity if avg_immunity > 0 else 0
        return variant_immunity_levels

    @property
    def summary(self) -> str:
        return (
            f"Susceptible: {self.num_susceptible}"
            f"\nInfected: {self.num_infected}"
            f"\nResistant: {self.num_resistant}"
            f"\nDead: {self.num_dead}"
        )
