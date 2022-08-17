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

#
k = ABMixin()
k.c = 'hola'
k.ab()
