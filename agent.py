from enum import Enum
from typing import cast

from mesa import Agent

from project.model import CovidModel


class InfectionState(int, Enum):
    """
    An enum representing the possible states of a CovidAgent
    """

    SUSCEPTIBLE = 0
    INFECTED = 1


class CovidAgent(Agent):
    model: CovidModel
    state: InfectionState

    def __init__(
        self, unique_id: int, model: CovidModel, initial_state: InfectionState
    ):
        super().__init__(unique_id, model)
        self.state = initial_state

    def step(self) -> None:
        # Since we handle infection of susceptibles on the infector side, we don't
        # need to implement any logic here for susceptible agents
        if self.state == InfectionState.INFECTED:
            # Either the agent recovers or spreads the virus with some probability
            if self.random.random() < self.model.recovery_prob:
                self.state = InfectionState.SUSCEPTIBLE
            else:
                self.infect_neighbors()

    def infect_neighbors(self) -> None:
        """
        Infect susceptible neighbors with probability model.infection_prob
        """
        for neighbor in self.neighbors:
            if (
                neighbor.state == InfectionState.SUSCEPTIBLE
                and self.random.random() < self.model.infection_prob
            ):
                neighbor.state = InfectionState.INFECTED

    @property
    def neighbors(self) -> list["CovidAgent"]:
        # Cast does nothing, just keeps the type-checker happy
        return cast(
            list[CovidAgent],
            self.model.grid.get_cell_list_contents(
                self.model.grid.get_neighbors(self.unique_id)
            ),
        )
