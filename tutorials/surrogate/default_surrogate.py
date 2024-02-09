import csv
from time import time

from mpi4py import MPI

from propulate import Surrogate
from propulate.population import Individual


class DefaultSurrogate(Surrogate):
    """
    This surrogate model does nothing.
    It exists for testing purposes and comparing performance decline
    due to the additional overhead of using a surrogate.
    """

    def __init__(self, dataset_name: str):
        super().__init__()

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
        self.surrogate_name: str = "DefaultSurrogate"
        # -----------------------

    def start_run(self, ind: Individual):
        # ----- csv logging -----
        # reset epoch
        self.epoch = 0
        # remeber ind for generation and params
        self.ind = ind
        # -----------------------

    def update(self, loss: float):
        pass

    def cancel(self, loss: float) -> bool:
        # ----- csv logging -----
        self._log_to_csv(loss)
        self.epoch += 1
        # -----------------------

        return False

    def merge(self, data: float):
        pass

    def data(self) -> float:
        return 0.0

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
