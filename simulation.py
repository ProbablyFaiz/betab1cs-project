# A place where we can run basic simulations
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from model import CovidModel

MAX_STEPS = 500
INFECTION_THRESHOLD = 1.0

NUM_PROCESSES = max(int(cpu_count() / 2), 1)
NUM_SIMULATIONS = 1

pool = None


def run_simulation(model_params: dict, debug_output=False) -> int | None:
    """
    Runs the COVID-19 model until the maximum number of steps, the disease has
    been eradicated, or the infection threshold has been reached.

    :return: The number of steps simulated for the model before termination
    """
    covid_model = CovidModel(**model_params)
    steps = 0
    while (
        steps < MAX_STEPS
        # If COVID is completely eradicated, we can stop
        and covid_model.num_infected > 0
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


def run_simulation_set(model_params: dict):
    display_dict = defaultdict(lambda: "default", model_params)
    print(
        f"""
---------------------------------------------------------------------
Running {NUM_SIMULATIONS} simulations with configuration:
Infection probability: {display_dict["infection_prob"]}, average degree: {display_dict["avg_degree"]}, death probability: {display_dict["death_prob"]}
Gain resistance probability: {display_dict["gain_resistance_prob"]}, resistance level: {display_dict["resistance_level"]}, recovery probability: {display_dict["recovery_prob"]}
Infection threshold: {INFECTION_THRESHOLD}, maximum time steps: {MAX_STEPS}"""
    )
    step_counts = pool.map(
        run_simulation, (model_params for _ in range(NUM_SIMULATIONS))
    )
    terminated_step_counts = [c for c in step_counts if c is not None]
    num_unterminated = sum(1 for c in step_counts if c is None)
    average_step_count = (
        sum(terminated_step_counts) / len(terminated_step_counts)
        if len(terminated_step_counts) > 0
        else None
    )
    print(
        f"""
Results
-------
Infection threshold reached: {len(terminated_step_counts)} (average steps: {average_step_count})
Not reached: {num_unterminated}
"""
    )


if __name__ == "__main__":
    pool = Pool(NUM_PROCESSES)
    run_configurations = [
        {
            "avg_degree": 10,
            "infection_prob": 0.1,
            "gain_resistance_prob": 0.01,
            "recovery_prob": 0.02,
            "resistance_level": 0.7,
            "death_prob": 0.0,
        },
    ]
    for config in run_configurations:
        run_simulation_set(config)
    pool.close()
