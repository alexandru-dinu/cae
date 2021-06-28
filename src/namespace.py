import json


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, xs):
        if isinstance(xs, dict):
            return Namespace(**xs)

        if isinstance(xs, (list, tuple)):
            return [self.__elt(x) for x in xs]

        return xs

    def __str__(self):
        return json.dumps(self.__get_nested(), indent=4)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.__dict__)

    def __get_nested(self) -> dict:
        out = {}

        for k, v in self.__dict__.items():
            # nested
            if isinstance(v, Namespace):
                out[k] = "<self>" if v is self else Namespace.__get_nested(v)

            # non-primitive type, call its str method
            elif hasattr(v, "__dict__"):
                out[k] = str(v)

            # primitives
            else:
                out[k] = v

        return out

    def is_empty(self):
        return len(self) == 0

    def to_dict(self) -> dict:
        return self.__get_nested()

    def to_file(self, fname) -> None:
        with open(fname, "wt") as fp:
            fp.write(str(self))
