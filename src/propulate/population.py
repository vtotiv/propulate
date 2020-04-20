# TODO multi objective optimization?
# TODO invalidate loss, when entry is modified so this does not have to be done by the propagator
class Individual(dict):
    def __init__(self, generation=None, rank=None, traits=[]):
        super(Individual, self).__init__(list())
        self.generation = generation
        self.rank = rank
        self.loss = None  # NOTE set to None instead of inf since there are no comparisons
        return
