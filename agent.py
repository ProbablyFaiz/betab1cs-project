from enum import Enum
from typing import cast, TYPE_CHECKING

from mesa import Agent

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


class CovidAgent(Agent):
    model: "CovidModel"
    state: InfectionState

    def __init__(
        self, unique_id: int, model: "CovidModel", initial_state: InfectionState
    ):
        super().__init__(unique_id, model)
        self.state = initial_state

    def step(self) -> None:
        if self.state == InfectionState.INFECTED:
            # Either the agent recovers or spreads the virus with some probability
            if self.random.random() < self.model.recovery_prob:
                # Assume for now that recovered agents become resistant
                self.state = InfectionState.RESISTANT
            elif self.random.random() < self.model.death_prob:
                self.state = InfectionState.DEAD
            else:
                self.infect_neighbors()
        elif self.state == InfectionState.SUSCEPTIBLE:
            if self.random.random() < self.model.gain_resistance_prob:
                self.state = InfectionState.RESISTANT

    def infect_neighbors(self) -> None:
        """
        Infect susceptible neighbors with probability model.infection_prob
        """
        for neighbor in self.neighbors:
            if self.should_infect_neighbor(neighbor):
                neighbor.state = InfectionState.INFECTED

    def should_infect_neighbor(self, neighbor: "CovidAgent") -> bool:
        # noinspection PyChainedComparisons
        return (
            neighbor.state == InfectionState.SUSCEPTIBLE
            and self.random.random() < self.model.infection_prob
        ) or (
            neighbor.state == InfectionState.RESISTANT
            and self.random.random() < self.model.infection_prob
            and self.random.random() > self.model.resistance_level
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
