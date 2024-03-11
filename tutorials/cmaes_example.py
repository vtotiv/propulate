import random
import argparse
import logging
import pathlib

from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import BasicCMA, ActiveCMA, CMAPropagator
from propulate.utils import set_logger_config
from function_benchmark import get_function_search_space


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print(
            "#################################################\n"
            "# PROPULATE: Parallel Propagator of Populations #\n"
            "#################################################\n"
        )

    parser = argparse.ArgumentParser(
        prog="Simple Propulator example",
        description="Set up and run a basic Propulator optimization of mathematical functions.",
    )
    parser.add_argument(  # Function to optimize
        "--function",
        type=str,
        choices=[
            "bukin",
            "eggcrate",
            "himmelblau",
            "keane",
            "leon",
            "rastrigin",
            "schwefel",
            "sphere",
            "step",
            "rosenbrock",
            "quartic",
            "bisphere",
            "birastrigin",
            "griewank",
        ],
        default="sphere",
    )
    parser.add_argument(
        "--generations", type=int, default=1000
    )  # Number of generations
    parser.add_argument(
        "--seed", type=int, default=0
    )  # Seed for Propulate random number generator
    parser.add_argument("--adapter", type=str, default="basic")
    parser.add_argument("--verbosity", type=int, default=1)  # Verbosity level
    parser.add_argument(
        "--checkpoint", type=str, default="./"
    )  # Path for loading and writing checkpoints.
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--logging_int", type=int, default=10)
    config = parser.parse_args()

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"{config.checkpoint}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    rng = random.Random(
        config.seed + comm.rank
    )  # Separate random number generator for optimization.
    function, limits = get_function_search_space(
        config.function
    )  # Get callable function + search-space limits.

    # Set up evolutionary operator.
    if config.adapter == "basic":
        adapter = BasicCMA()
    elif config.adapter == "active":
        adapter = ActiveCMA()
    else:
        raise ValueError("Adapter can be either 'basic' or 'active'.")

    propagator = CMAPropagator(adapter, limits, rng=rng)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=function,
        propagator=propagator,
        rng=rng,
        island_comm=comm,
        generations=config.generations,
        checkpoint_path=config.checkpoint,
    )

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=config.logging_int, debug=config.verbosity)
    propulator.summarize(top_n=config.top_n, debug=config.verbosity)
