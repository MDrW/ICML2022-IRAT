import numpy as np


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.
    Example usage:
        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...
    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.
    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def stack_dict_list(dict_list):
    ret = dict()
    if not dict_list:
        return ret
    keys = dict_list[0].keys()
    for k in keys:
        eg = dict_list[0][k]
        if isinstance(eg, dict):
            v = stack_dict_list([x[k] for x in dict_list])
        else:
            v = np.array([x[k] for x in dict_list])
        ret[k] = v

    return ret
