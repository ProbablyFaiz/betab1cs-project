import networkx as nx

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from model import CovidModel
from agent import CovidAgent, InfectionState

STATE_COLOR_MAP = {
    InfectionState.INFECTED: "#cc0000",
    InfectionState.SUSCEPTIBLE: "#f6b26b",
    InfectionState.RESISTANT: "#6aa84f",
    InfectionState.DEAD: "#000000",
}


def network_portrayal(network: nx.Graph):
    def node_color(agent: CovidAgent) -> str:
        return STATE_COLOR_MAP[agent.state]

    def edge_color(agent1: CovidAgent, agent2: CovidAgent) -> str:
        return "#e8e8e8"

    def edge_width(agent1: CovidAgent, agent2: CovidAgent) -> int:
        return 2

    def get_agents(source, target):
        return network.nodes[source]["agent"][0], network.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agent),
            "tooltip": f"{agent.unique_id}: {agent.state.name}"
        }
        for (_, (agent,)) in network.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in network.edges
    ]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500, library="d3")
chart = ChartModule(
    [
        {"Label": "Infected", "Color": STATE_COLOR_MAP[InfectionState.INFECTED]},
        {"Label": "Susceptible", "Color": STATE_COLOR_MAP[InfectionState.SUSCEPTIBLE]},
        {"Label": "Resistant", "Color": STATE_COLOR_MAP[InfectionState.RESISTANT]},
        {"Label": "Dead", "Color": STATE_COLOR_MAP[InfectionState.DEAD]},
    ]
)


class ModelInfo(TextElement):
    def render(self, model: CovidModel):
        return (
            f"{model.num_infected} infected, "
            f"{model.num_susceptible} susceptible, "
            f"{model.num_resistant} resistant, "
            f"{model.num_dead} dead"
        )


server = ModularServer(CovidModel, [network, ModelInfo(), chart], "COVID-19 Model")
