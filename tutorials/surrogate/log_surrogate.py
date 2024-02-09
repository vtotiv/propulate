import csv
from time import time

from mpi4py import MPI

from propulate import Surrogate
from propulate.population import Individual


class LogSurrogate(Surrogate):
    """
    Just a few debug prints to see what's going on.
    """

    def __init__(self, dataset_name: str):
        super().__init__()
        self.synthetic_id: int = 0
        self.best_loss: float = 10000.0

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
        self.surrogate_name: str = "LogSurrogate"
        # -----------------------

        print("LogSurrogate initialized")

    def start_run(self, ind: Individual):
        self.synthetic_id = 0
        print(f"LogSurrogate - Start run called on individual with keys: {ind.keys()} and values: {ind.values()}")

        # ----- csv logging -----
        # reset epoch
        self.epoch = 0
        # remeber ind for generation and params
        self.ind = ind
        # -----------------------

    def update(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
        print(f"LogSurrogate - Updated on id {self.synthetic_id} with loss: {loss} and best loss: {self.best_loss}")

    def cancel(self, loss: float) -> bool:
        print(f"LogSurrogate - Cancel called on id {self.synthetic_id} with loss: {loss}")

        # ----- csv logging -----
        self._log_to_csv(loss)
        self.epoch += 1
        # -----------------------

        self.synthetic_id += 1
        return False

    def merge(self, data: float):
        print(f"LogSurrogate - Merge called with best loss: {self.best_loss} and data: {data}")
        if data < self.best_loss:
            self.best_loss = data

    def data(self) -> float:
        print(f"LogSurrogate - Data called with best loss: {self.best_loss}")
        return self.best_loss

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
