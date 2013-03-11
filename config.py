import brian.units


def quantity(l):
    magnitude, unit = l
    return magnitude * getattr(brian.units, unit)


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
