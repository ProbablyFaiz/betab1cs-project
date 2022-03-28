from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from model import CovidModel

# Function that describes how each cell will be portrayed in the visualization
def portrayCell(cell):
    assert cell is not None
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": cell.x,
        "y": cell.y,
        "Color": "black" if cell.state == cell.PRO else "white",
    }


# Make a world that is 100x100, on a 500x500 display.
canvas_element = CanvasGrid(portrayCell, 100, 100, 500, 500)

server = ModularServer(
    CovidModel, [canvas_element], "Covid Infection Model", {"height": 100, "width": 100}
)
