class AttrDict(dict):
    def __getattr__(self, key):
        return self[key] if key in self else self.__getattribute__(key)

    __setattr__ = dict.__setitem__

    def copy(self) -> 'AttrDict':
        return AttrDict(self)
