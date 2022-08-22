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
        return('im even')
    elif n % 5 == 0:
        return('minecraft')
    else:
        return(n+1)

# this is a simple function you can reference from
def multiply_by_2(n):
    return n*2

class ExampleClass:
    def __init__(self,n):
        self.n=n

    def do_something(self):
        self.n=multiply_by_2(self.n)
        print(self.n)

if __name__ == "__main__":
    print("====== example function output ===============")
    # example use of a function
    a = multiply_by_2(5)
    print(a)

    print("====== example class output ===============")
    # example use of a class:
    object_a = ExampleClass(6)
    object_a.do_something()
    object_a.do_something()

    print("========")
    for i in range(10):
        print(process_n(i))


""" expected output for n= 1,2,3,4,5,6,7,8,9,10:
1
i'm even
3
i'm even
minecraft !
i'm even
7
i'm even
9
i'm even minecraft!


current output: THIS IS DIFFERENT ^
im even
2
im even
4
im even
minecraft
im even
8
im even
10

"""
