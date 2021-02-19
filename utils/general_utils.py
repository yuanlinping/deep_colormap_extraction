from random import shuffle
import numpy as np

def shuffle_list(*ls):  # native python based
    """
    :param ls:
    :return:
    >>> a = [0,1,2,3,4]
    >>> b = [5,6,7,8,9]
    >>> a1,b1 = shuffle_list(a,b)
    >>> print(a1,b1)
    >>> a = [0,1,2,3,4]
    >>> b = [5,6,7,8,9]
    >>> c = [10,11,12,13,14]
    >>> a1,b1,c1 = shuffle_list(a,b,c)
    >>> print(a1,b1,c1)
    """
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)




