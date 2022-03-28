from enum import Enum

from mesa import Agent

from project.model import CovidModel


class InfectionState(int, Enum):
    """
    An enum representing the possible states of a CovidAgent
    """

    SUSCEPTIBLE = 0
    INFECTED = 1


class CovidAgent(Agent):
    state: InfectionState

    def __init__(
        self, unique_id: int, model: CovidModel, initial_state: InfectionState
    ):
        super().__init__(unique_id, model)
        self.state = initial_state

    def step(self) -> None:
        pass
