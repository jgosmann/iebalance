import brian.units


class EquationString(object):
    def __init__(self, separator):
        self.separator = separator

    def __call__(self, equations):
        return str(self.separator.join(equations))


def quantity((magnitude, unit)):
    return magnitude * getattr(brian.units, unit)


def quantity_list((l, unit)):
    return [quantity((magnitude, unit)) for magnitude in l]


class Configurable(object):
    def __init__(self, config):
        self._config = config

    def _add_config_value(self, name, handler=None, default=None):
        try:
            value = self._config[name]
        except KeyError as e:
            if default is not None:
                value = default
            else:
                raise e
        if handler is not None:
            value = handler(value)
        setattr(self, name, value)
