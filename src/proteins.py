from statistics import mode
import numpy as np

def checker(arr, p, i):
    try:
        return mode([pro[i] for pro in arr])
    except:
        return p[0][i]

def predict(arr, p):
    m = max(map(len, arr))
    pro = [checker(arr, p, i) for i in range(m)]
    return str().join(pro)

def find_na(s):
    for i, c in enumerate(s):
        if c == '0':
            print(i)
            
def shorter(a, b):
    return a if len(a) < len(b) else b

def compare(a, b):
    l = len(shorter(a, b))
    c = [a[i] == b[i] for i in range(l)]
    return sum(c)

def counter(arr, s):
    return [compare(pro, s) for pro in arr]

def r_fills(s, lst):
    if lst:
        return r_fills(s[:lst[0]]+ '0' + s[lst[0]:], lst[1:])
    else:
        return s
    
def fillers(arr, s, n):
    empty = [len(s) for i in range(n)]
    value = np.zeros(empty)
    it = np.nditer(value, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        it[0] = sum(counter(arr, r_fills(s, it.multi_index)))
        it.iternext()

    return np.unravel_index(value.argmax(), value.shape)
