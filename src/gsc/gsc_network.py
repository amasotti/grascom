"""
The Gradient Symbolic Computation Network class

"""


class Net(object):
    def __init__(self, grammar):
        #  Set up the settings dictionary
        self._define_settings()
        # The Harmonic Grammar
        self.grammar = grammar

        # Harmony constraints
        self.Hc = self.grammar.Hc
        self.Hcc = self.grammar.Hcc

    def _define_settings(self):
        """Set option variables to default values."""

        self.settings = {}
        # Bowl center
        self.settings['z'] = 0.3
        # bowl multiplier
        self.settings['q'] = 16.2
        # Maximum input in the C-Space : no constituent can be more than 100% present
        self.settings['maxInp'] = 1
