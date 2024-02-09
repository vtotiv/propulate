import csv
from time import time

from mpi4py import MPI

import numpy as np
from propulate import Surrogate
from propulate.population import Individual


class StaticSurrogate(Surrogate):
    """
    Surrogate model using the best known run as baseline.
    After the first run, each subsequent loss is compared to the baseline.
    Every run with a loss outside the margin of the baseline is cancelled.

    This model assumes regular yields between training runs,
    otherwise the indices of the baseline run won't match.
    """

    def __init__(self, dataset_name: str, margin: float = 0.8):
        print("Static Surrogate - init")

        super().__init__()

        self.synthetic_id: int = 0
        self.margin: float = margin

        # cancel is only allowed after the first complete run
        # first_run keeps track of that
        self.first_run: bool = True

        # baseline is the best known run
        self.baseline: np.ndarray = np.zeros((0), dtype=float)
        self.current_run: np.ndarray = np.zeros((0), dtype=float)

        # --------------------
        # set up csv logging
        # --------------------
        # set dataset name for file name
        self.dataset_name: str = dataset_name
        self.rank: int = MPI.COMM_WORLD.rank
        # remember the current epoch
        self.epoch: int = 0
        # and individual
        self.ind: Individual = None
        # change this for each surrogate
        self.surrogate_name: str = "StaticSurrogate"
        # -----------------------

    def start_run(self, ind: Individual):
        print("Static Surrogate - start run - ind", ind.keys(), ind.values())

        # reset to new run
        self.synthetic_id = 0
        # reset current run with correct size
        self.current_run = np.zeros((self.baseline.size), dtype=float)

        # ----- csv logging -----
        # reset epoch
        self.epoch = 0
        # remeber ind for generation and params
        self.ind = ind
        # -----------------------

    def update(self, loss: float):
        print("Static Surrogate - update - loss", loss)

        if self.first_run:
            self.baseline = self.current_run.copy()
            self.first_run = False
            return

        # check if current run is better than baseline
        if self.baseline[-1] > self.current_run[-1]:
            self.baseline = self.current_run.copy()

    def cancel(self, loss: float) -> bool:
        print("Static Surrogate - cancel - loss", loss)

        # ----- csv logging -----
        self._log_to_csv(loss)
        self.epoch += 1
        # -----------------------

        self.synthetic_id += 1

        # cancel is only allowed after the first complete run
        if self.first_run:
            self.current_run = np.append(self.current_run, loss)
            return False

        # append loss to current run
        self.current_run[self.synthetic_id - 1] = loss

        print("Static Surrogate - cancel - compare with baseline", self.baseline[self.synthetic_id - 1], "and loss", loss * self.margin)
        # cancel if current run is outside margin of baseline
        if self.baseline[self.synthetic_id - 1] < loss * self.margin:
            return True

        return False

    def merge(self, data: np.ndarray):
        print("Static Surrogate - merge")

        # no prior data to merge with
        if self.first_run:
            self.baseline = data.copy()
            self.first_run = False
            return

        # merged runs final loss is better than baseline
        if self.baseline[-1] < data[-1]:
            self.baseline = data.copy()

    def data(self) -> np.ndarray:
        print("Static Surrogate - data")

        # return best run so far
        return self.baseline

    def _log_to_csv(
        self,
        avg_validation_loss,
    ) -> None:
        file_name = f"{self.dataset_name}_log_{self.rank}.csv"

        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                time(),
                self.rank,
                self.surrogate_name,
                self.ind.generation,
                self.epoch,
                '#'.join([f"{k}={v}" for k, v in self.ind.items()]),
                avg_validation_loss])
