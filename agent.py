from enum import Enum
from typing import cast, TYPE_CHECKING

from mesa import Agent

from variant import CovidVariant

if TYPE_CHECKING:  # Avoids circular import issues
    from model import CovidModel


class InfectionState(int, Enum):
    """
    An enum representing the possible states of a CovidAgent
    """

    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2
    DEAD = 3


# noinspection PyChainedComparisons
class CovidAgent(Agent):
    model: "CovidModel"
    state: InfectionState
    infection_variant: CovidVariant | None

    immune_memory: list[CovidVariant]

    def __init__(
        self,
        unique_id: int,
        model: "CovidModel",
        initial_state: InfectionState,
        infection_variant: CovidVariant = None,
    ):
        super().__init__(unique_id, model)
        self.state = initial_state
        self.immune_memory = []
        self.infection_variant = infection_variant

    def step(self) -> None:
        if self.state == InfectionState.INFECTED:
            # Either the agent recovers or spreads the virus with some probability
            if self.random.random() < self.model.recovery_prob:
                # Recovered agents become resistant
                self.state = InfectionState.RESISTANT
                self.immune_memory.append(self.infection_variant)
                self.infection_variant = None
            elif (
                self.random.random() < self.model.death_prob
                and self.random.random() > self.death_resistance_level
            ):
                self.state = InfectionState.DEAD
            else:
                self.infect_neighbors()
        elif self.state == InfectionState.SUSCEPTIBLE:
            if self.random.random() < self.model.gain_resistance_prob:
                self.state = InfectionState.RESISTANT
                self.immune_memory.append(
                    CovidVariant(0, 0)
                )  # Vaccine immunity, not contagious

    def infect_neighbors(self) -> None:
        """
        Infect susceptible neighbors with probability model.infection_prob
        """
        for neighbor in self.neighbors:
            neighbor.try_infect(self.infection_variant)

    def try_infect(self, variant: CovidVariant) -> None:
        if (
            self.state in (InfectionState.SUSCEPTIBLE, InfectionState.RESISTANT)
            and self.random.random() < variant.base_infection_prob
            and self.random.random() > self.infection_resistance_level(variant)
        ):
            self.state = InfectionState.INFECTED
            self.infection_variant = variant.child_variant(self.model.mutation_prob)

    def infection_resistance_level(self, variant: CovidVariant) -> float:
        return (
            max(
                remembered_variant.similarity(variant)
                for remembered_variant in self.immune_memory
            )
            if len(self.immune_memory)
            else 0
        )

    @property
    def death_resistance_level(self) -> float:
        return (
            max(
                remembered_variant.similarity(self.infection_variant)
                for remembered_variant in self.immune_memory
            )
            if len(self.immune_memory)
            else 0
        )

    @property
    def neighbors(self) -> list["CovidAgent"]:
        # Cast does nothing, just keeps the type-checker happy
        return cast(
            list[CovidAgent],
            self.model.grid.get_cell_list_contents(
                self.model.grid.get_neighbors(self.unique_id)
            ),
        )
