# A place where we can run basic simulations
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

from model import CovidModel, DEFAULT_MODEL_PARAMS

MAX_STEPS = 1500
INFECTION_THRESHOLD = 1.0

NUM_PROCESSES = max(int(cpu_count() / 2), 1)
NUM_SIMULATIONS = 15

pool = None


def run_simulation(model_params: dict, debug_output=True) -> int | None:
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
    if model_params.get("dump_variant_data", False):
        covid_model.dump_variant_data()
    if termination_condition_unmet(covid_model):
        return None
    return steps


def termination_condition_unmet(covid_model: CovidModel) -> bool:
    return covid_model.num_infected / len(covid_model.agents) < INFECTION_THRESHOLD


def get_model_param(params: dict, key: str):
    return params[key] if key in params else DEFAULT_MODEL_PARAMS[key]


def run_simulation_set(model_params: dict):
    mp = partial(get_model_param, model_params)
    print(
        f"""
---------------------------------------------------------------------
Running {NUM_SIMULATIONS} simulations with configuration:
Infection probability: {mp("infection_prob")}, average degree: {mp("avg_degree")}, death probability: {mp("death_prob")}
Gain resistance probability: {mp("gain_resistance_prob")}, resistance level: {mp("resistance_level")}, recovery probability: {mp("recovery_prob")}
Mutation rate: {mp("mutation_prob")}, infection threshold: {INFECTION_THRESHOLD}, maximum time steps: {MAX_STEPS}"""
    )
    step_counts = list(map(
        run_simulation, (model_params for _ in range(NUM_SIMULATIONS))
    ))
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
    # pool = Pool(NUM_PROCESSES)
    run_configurations = [
        {
            "mutation_prob": 0.001,
            "genome_bits": 8,
            "dump_variant_data": False,
        },
    ]
    for config in run_configurations:
        run_simulation_set(config)
    # pool.close()
