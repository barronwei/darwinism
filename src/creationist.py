import random as rand
import secrets as sec
from enum import Enum

"""
This is a representation of nucleotides through an enumeration.
This forces any value that is a nucleotide to have one of the following values: A, G, T, C!
"""

class Nucleo(Enum):
    A = 0
    G = 1
    T = 2
    C = 3
    
"""
This is a representation of possible mutations that a nucleotide can go through.
This forces any value that is a mutation to be a conversion, insertion, or deletion.
"""
    
class Change(Enum):
    modify = 0
    insert = 1
    delete = 2
    
"""
This is a representation of a genome, which has only two accessible methods: get_data and r_change.
In other words, we cannot tamper with the genome deterministically. This has to be done through
r_change, which generates a random mutation. This is a representation of creationist ideology.
"""

class Genome:
    def __init__(self, size):
        self.__size = size
        self.__data = self.__r_genome()
        
    def __r_nucleo(self):
        return Nucleo(sec.randbelow(len(Nucleo))).name
        
    def __r_genome(self):
        return str().join(map(str, [self.__r_nucleo() for i in range(self.__size)]))
    
    def __replacer(self, v, x):
        self.__data = self.__data[0:x] + v + self.__data[x+1:]
    
    def __r_insert(self, x):
        self.__replacer(self.__r_nucleo(), x)
        
    def __r_delete(self, x):
        self.__replacer(str(), x)
        
    def __pair_map(self, v):
        return {
            'A': 'T',
            'G': 'C',
            'T': 'A',
            'C': 'G',
        }.get(v)
        
    def __r_modify(self, x):
        self.__replacer(self.__pair_map(self.__data[x]), x)
    
    def __switcher(self, v, x):
        return {
            '0': lambda x: self.__r_insert(x),
            '1': lambda x: self.__r_delete(x),
            '2': lambda x: self.__r_modify(x),
        }.get(v)(x)
    
    def get_data(self):
        return self.__data

    def r_change(self):
        loc = sec.randbelow(len(self.__data))
        num = sec.randbelow(len(Change))
        
        mut = Change(num).value
        
        self.__switcher(str(mut), loc)
        
"""
This function finds the number of necessary mutations a random genome needs to go through until
that genome matches s.
"""
        
def n_finder(s):
    g = Genome(len(s))
    n = 0
    
    while (g.get_data() != s):
        g.r_change()
        n += 1

    return n

    
"""
This is a representation of a simple genome that can only go through conversions.
"""

class Genome_s:
    def __init__(self, s):
        self.__size = len(s)
        self.__data = self.__d_genome(s)
        
    def __d_genome(self, s):
        res = [self.__pair_map(n) if bool(rand.getrandbits(1)) else n for n in s]
        return str().join(map(str, res))
    
    def __replacer(self, v, x):
        self.__data = self.__data[0:x] + v + self.__data[x+1:]
    
    def __pair_map(self, v):
        return {
            'A': 'T',
            'G': 'C',
            'T': 'A',
            'C': 'G',
        }.get(v)
        
    def __r_modify(self, x):
        self.__replacer(self.__pair_map(self.__data[x]), x)
    
    def get_data(self):
        return self.__data

    def r_change(self):
        loc = sec.randbelow(self.__size)
        self.__r_modify(loc)
        
"""
This function finds the number of necessary mutations a random simple genome needs to go 
through until that simple genome matches s.
"""
        
def n_finder_s(s):
    g = Genome_s(s)
    n = 0
    
    while (g.get_data() != s):
        g.r_change()
        n += 1

    return n
