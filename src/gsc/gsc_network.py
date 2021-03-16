"""
The Gradient Symbolic Computation Network class

"""


class Net(object):
    def __init__(self):
        #  Set up the settings dictionary
        self._define_settings()

    def define_settings(self):
        """Set option variables to default values."""

        self.settings = {}
        self.settings['trace_varnames'] = [
            'act', 'H', 'H0', 'Q', 'q', 'T', 't', 'ema_speed', 'speed']
        self.settings['norm_ord'] = np.inf
        self.settings['coord'] = 'N'
        self.settings['ema_factor'] = 0.001
        self.settings['ema_tau'] = -1 / np.log(self.settings['ema_factor'])
        self.settings['T_init'] = 1e-3
        self.settings['T_min'] = 0.
        self.settings['T_decay_rate'] = 1e-3
        self.settings['q_init'] = 0.
        self.settings['q_max'] = 200.
        self.settings['q_rate'] = 10.
        self.settings['c'] = 0.5
        self.settings['bowl_center'] = 0.5
        self.settings['bowl_strength'] = None
        self.settings['beta_min_offset'] = 0.1
        self.settings['dt'] = 0.001
        self.settings['H0_on'] = True
        self.settings['H1_on'] = True
        self.settings['Hq_on'] = True
        self.settings['max_dt'] = 0.01
        self.settings['min_dt'] = 0.0005
        self.settings['q_policy'] = None

    def _update_settings(self, opts):
        """Update option variable values"""

        if opts is not None:
            for key in opts:
                if key in self.settings:
                    self.settings[key] = opts[key]
                    if key == 'ema_factor':
                        self.opts['ema_tau'] = -1 / np.log(self.opts[key])
                    if key == 'ema_tau':
                        self.opts['ema_factor'] = np.exp(-1 / self.opts[key])
                else:
                    raise ValueError("Check the updated infos")
