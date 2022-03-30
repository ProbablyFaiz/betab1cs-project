import networkx as nx

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
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
            "tooltip": f"{agent.unique_id}: {agent.state.name}",
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

model_params = {
    "num_nodes": UserSettableParameter(
        "number",
        name="Number of Agents",
        value=500,
        min_value=1,
        max_value=1000,
        step=50,
        description="Number of Agents",
    ),
    "avg_degree": UserSettableParameter(
        "number",
        name="Avg. Node Degree",
        value=10,
        min_value=1,
        max_value=50,
        step=1,
        description="Avg. Node Degree",
    ),
    # TODO: Add such a parameter (possibly with different variants for each individual)
    # "initial_outbreak_size": UserSettableParameter(
    #     "slider",
    #     name="Initial Outbreak Size",
    #     value=1,
    #     min_value=1,
    #     max_value=20,
    #     step=1,
    #     description="Initial Outbreak Size",
    # ),
    "infection_prob": UserSettableParameter(
        "number",
        name="Base Infection Probability",
        value=0.1,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        description=(
            "Probability of an agent infecting a neighboring agent during a single time"
            " step for the initial variant"
        ),
    ),
    "death_prob": UserSettableParameter(
        "number",
        name="Base Death Probability",
        value=0.001,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        description=(
            "Probability of an infected agent dies during a single time step for the"
            " initial variant"
        ),
    ),
    "recovery_prob": UserSettableParameter(
        "number",
        name="Recovery Probability",
        value=0.05,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description=(
            "Probability that an infected agent recovers during a single time step"
        ),
    ),
    "gain_resistance_prob": UserSettableParameter(
        "number",
        "Gain Resistance Probability",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        description=(
            "Probability that a susceptible agent will become resistant to the initial"
            " variant during a given time step"
        ),
    ),
    "resistance_level": UserSettableParameter(
        "number",
        "Resistance Level",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description=(
            "Probability that a resistant agent will resist an infection relative to a"
            " susceptible agent. Can be interpreted as vaccine efficacy/protection"
            " against re-infection"
        ),
    ),
    "mutation_prob": UserSettableParameter(
        "number",
        "Mutation Probability",
        value=0.001,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        description=(
            "Probability that a given bit of a virus' genetic code will flip during"
            " reproduction"
        ),
    ),
    "genome_bits": UserSettableParameter(
        "slider",
        "Genome Bits",
        value=4,
        min_value=4,
        max_value=16,
        step=4,
        description=(
            "Number of bits in each agent's genome. Must be a multiple of 4. 2^n"
            " variants are possible for a genome_bits value n"
        ),
    ),
}


class ModelInfo(TextElement):
    def render(self, model: CovidModel):
        return (
            f"{model.num_infected} infected, "
            f"{model.num_susceptible} susceptible, "
            f"{model.num_resistant} resistant, "
            f"{model.num_dead} dead<br>"
            "Dominant variants: "
            f"{', '.join([f'{variant.name} ({freq} cases, {variant.base_infection_prob:3.2f} infectivity)' for (variant, freq) in model.variant_frequency[:3]])}"
        )


server = ModularServer(
    CovidModel, [network, ModelInfo(), chart], "COVID-19 Model", model_params
)
