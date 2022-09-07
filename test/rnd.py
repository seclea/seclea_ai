class CMixin:
    _c: str

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val

    def serialize(self):
        return {"bd": "hi"}


class AMixin(CMixin):
    def a(self):
        print("hi a: ", self.c)
        self.c = "k"

    def serialize(self):
        return {"bc": "hi"}


class BMixin(CMixin):
    def b(self):
        print("hi b: ", self.c)

    def serialize(self):
        return {"bk": "hi"}


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
        return {
            **{"b": "hi"},
            **AMixin.serialize(self),
            **BMixin.serialize(self),
            **CMixin.serialize(self),
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
        # having nested if statements is not good practice
        # you can also combine the logic of "im even" and "minecraft" (see examples bellow
        if n % 5 == 0:
            return "im even minecraft"
        else:
            return "im even"
    elif n % 5 == 0:
        return "minecraft"
    else:
        return n


def process_n_2(n):
    # this is better but still too many if statements if we have more conditions
    to_print = ""
    if n % 2 == 0:
        to_print += "i'm even "
    if n % 5 == 0:
        to_print += "minecraft "
    if len(to_print) == 0:
        to_print = str(n)
    return to_print


def process_n_3(n):
    # this allows us to expand for many condition cleanly
    condition_dict = {2: "i'm even ", 5: "minecraft ", 6: "hello ", 12: "goodbye "}
    result = ""

    for key, val in condition_dict.items():
        if n % key == 0:
            result += val
    return result if result != "" else n


def driver(fn):
    for i in range(10):
        print(fn(i))


a = "asd"
print(a[:10])
if __name__ == "__main__":
    print("======[ bedirhan's ]=======")
    driver(process_n)
    print("======[ octavio's 1]=======")
    driver(process_n_2)
    print("======[ octavio's 2]=======")
    driver(process_n_3)


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
