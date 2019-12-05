import secrets as sr
from enum import Enum

"""
nucleotides
"""


class Nucleo(Enum):
    A = 0
    G = 1
    T = 2
    C = 3


"""
possible mutations
"""


class Change(Enum):
    modify = 0
    insert = 1
    delete = 2


"""
genome abstraction barrier
"""


class Genome:
    def __init__(self, size):
        self.__size = size
        self.__data = self.__r_genome()

    def __r_nucleo(self):
        return Nucleo(sr.randbelow(len(Nucleo))).name

    def __r_genome(self):
        return str().join(map(str, [self.__r_nucleo() for i in range(self.__size)]))

    def __replacer(self, v, x):
        self.__data = self.__data[0:x] + v + self.__data[x:]

    def __r_insert(self, x):
        self.__replacer(self.__r_nucleo(), x)

    def __r_delete(self, x):
        self.__replacer(str(), x)

    def __pair_map(self, v):
        return {"A": "T", "G": "C", "T": "A", "C": "G"}.get(v)

    def __r_modify(self, x):
        self.__replacer(self.pair_map(self.__data[x]), x)

    def __switcher(self, v, x):
        return {
            "0": lambda x: self.__r_insert(x),
            "1": lambda x: self.__r_delete(x),
            "2": lambda x: self.__r_modify(x),
        }.get(v)(x)

    def get_data(self):
        return self.__data

    def r_change(self):
        loc = sr.randbelow(len(self.__data))
        num = sr.randbelow(len(Change))

        mut = Change(num).value

        self.__switcher(str(mut), loc)


"""
finder for number of necessary mutations for s
"""


def n_finder(s):
    g = Genome(len(s))
    n = 0

    while g.get_data() != s:
        g.r_change()
        n += 1

    return n
