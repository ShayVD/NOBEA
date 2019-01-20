

class Benchmark(object):

    def __init__(self, function, domain, dimensions, min_value, name):
        self._function = function
        self._domain = domain
        self._dimensions = dimensions
        self._min_value = min_value
        self._name = name

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        self._function = function

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        self._dimensions = dimensions

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, min_value):
        self._min_value = min_value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name