import numpy as np
import scipy.stats

"""
This function returns false for the 0 string.
"""

def false_s(s):
    return s != '0'

"""
This function returns a list of modes in x.
"""

def mode_fn(x):
    return [i for i in set(x) if x.count(i) == max(map(x.count, x))]

"""
This function recursively looks for a valid value in protein
strand s at index i in the priority queue arr. This function
returns an invalid value if a valid one is not available.
"""

def r_check(arr, s, i):
    if false_s(arr[s][i]):
        return arr[s][i]
    
    next = s + 1
    
    if next == len(arr):
        return '0'
    
    r_check(arr, next, i)
    

"""
This function returns the most common amino acid among the
proteins in arr with ancestral priority of list p at index i.
"""

def checker(arr, p, i):
    r = [s[i] for s in arr]
    m = mode_fn(r)
    
    if len(m) == 1:
        return m[0] if false_s(m[0]) else r_check(p, 0, i)
    
    lst = [c for c in p if c[i] in m]
    
    return r_check(lst, 0, i)

    
"""
This function predicts the ancestral protein of the proteins in arr
that are all of the same length. Use fillers to accommodate lengths.
"""

def predict(arr, p):
    m = max(map(len, arr))
    pro = [checker(arr, p, i) for i in range(m)]
    return str().join(pro)

"""
This function prints out the locations in s that are NA.
"""

def find_na(s):
    for i, c in enumerate(s):
        if c == '0':
            print(i)
           
"""
This function returns the shorter strand.
"""
            
def shorter(a, b):
    return a if len(a) < len(b) else b

"""
This function assists in the comparison between strands of proteins.
"""

def compare(a, b):
    l = len(shorter(a, b))
    c = [a[i] == b[i] for i in range(l)]
    return sum(c)

"""
This function provides a list of the number of similarities between
s and each of the strands in arr.
"""

def counter(arr, s):
    return [compare(pro, s) for pro in arr]

"""
This function assists with recursively inserting placeholders into
s at the locations in lst.
"""

def r_fills(s, lst):
    if lst:
        return r_fills(s[:lst[0]] + '0' + s[lst[0]:], lst[1:])
    else:
        return s
    
"""
This function simulates the insertion of n placeholders to compare 
the similar proteins between s and strands in arr. This simulation
returns the indices of where n placeholders should be for the
maximum number of similarities.
"""
    
def fillers(arr, s, n):
    empty = [len(s) + 1 for i in range(n)]
    value = np.zeros(empty)
    it = np.nditer(value, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        it[0] = sum(counter(arr, r_fills(s, it.multi_index)))
        it.iternext()

    return np.unravel_index(value.argmax(), value.shape)

