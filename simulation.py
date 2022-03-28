# A place where we can run basic simulations
from multiprocessing import Pool, cpu_count

from model import CovidModel

NUM_NODES = 500
AVG_DEGREE = 4
INFECTION_PROB = 0.1
RECOVERY_PROB = 0.01
GAIN_RESISTANCE_PROB = 0.01
RESISTANCE_LEVEL = 0.7

MAX_STEPS = 500
INFECTION_THRESHOLD = 0.9

NUM_PROCESSES = max(int(cpu_count() / 2), 1)
NUM_SIMULATIONS = 500


def run_simulation(debug_output=False) -> int | None:
    """
    Runs the COVID-19 model until the maximum number of steps, the disease has
    been eradicated, or the infection threshold has been reached.

    :return: The number of steps simulated for the model before termination
    """
    covid_model = CovidModel(
        NUM_NODES, AVG_DEGREE, INFECTION_PROB, RECOVERY_PROB, GAIN_RESISTANCE_PROB, RESISTANCE_LEVEL
    )
    steps = 0
    while (
        steps < MAX_STEPS
        and covid_model.num_infected
        > 0  # If COVID is completely eradicated, we can stop
        and termination_condition_unmet(covid_model)
    ):
        covid_model.step()
        steps += 1
    if debug_output:
        print(f"Model after {steps} steps:")
        print(covid_model.summary)
    if termination_condition_unmet(covid_model):
        return None
    return steps


def termination_condition_unmet(covid_model: CovidModel) -> bool:
    return covid_model.num_infected / len(covid_model.agents) < INFECTION_THRESHOLD


if __name__ == "__main__":
    print("Running simulations...")
    with Pool(NUM_PROCESSES) as p:
        step_counts = p.map(run_simulation, (False for _ in range(NUM_SIMULATIONS)))
    terminated_step_counts = [c for c in step_counts if c is not None]
    num_unterminated = sum(1 for c in step_counts if c is None)
    average_step_count = (
        sum(terminated_step_counts) / len(terminated_step_counts)
        if len(terminated_step_counts) > 0
        else None
    )
    print(
        f"""Results of {NUM_SIMULATIONS} simulations:
Parameters
----------
Infection probability: {INFECTION_PROB}, average degree: {AVG_DEGREE},
Infection threshold: {INFECTION_THRESHOLD}, maximum time steps: {MAX_STEPS}

Infectivity threshold reached: {len(terminated_step_counts)} (average steps: {average_step_count})
Not reached: {num_unterminated}"""
    )
    print(f"")
