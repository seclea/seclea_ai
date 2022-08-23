class CMixin:
    _c: str

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val

    def serialize(self):
        return {'bd': 'hi'}


class AMixin(CMixin):
    def a(self):
        print('hi a: ', self.c)
        self.c = 'k'

    def serialize(self):
        return {'bc': 'hi'}


class BMixin(CMixin):
    def b(self):
        print('hi b: ', self.c)

    def serialize(self):
        return {'bk': 'hi'}


class ABMixin(AMixin, BMixin, CMixin):
    def ab(self):
        print(self.c)
        self.a()
        self.b()
        print(self.c)

    def serialize(self):
        AMixin.serialize(self)
        BMixin.serialize(self)
        CMixin.serialize(self)
        return {**{'b': 'hi'},
                **AMixin.serialize(self),
                **BMixin.serialize(self),
                **CMixin.serialize(self)
                }


def process_n(n):
    """
    When n is even print "i'm even"
    when n is divisible by 5 print " minecraft!"
    otherwise print n

    @param n: (integer) the number
    """
    # Bedirhan write your code here
    if n % 2 == 0:
        return ('im even')
    elif n % 5 == 0:
        return ('minecraft')
    else:
        return (n + 1)


# this is a simple function you can reference from
def multiply_by_2(n):
    return n * 2


class ExampleClass:
    def __init__(self, n):
        self.n = n

    def do_something(self):
        self.n = multiply_by_2(self.n)
        print(self.n)


def _assemble_kwargs(**kwargs):
    return dict([(key, val) for key, val in kwargs.items() if val is not None])


print(_assemble_kwargs(hi=None, hi2=2, hi3='3'))
