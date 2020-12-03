# these imports are the standard imports for required for derived lockboxes
from pyrpl.software_modules.lockbox import *
from pyrpl.software_modules.loop import *
import numpy as np
from ....widgets.module_widgets import AutoLockInputWidget
from ....modules import SignalLauncher
from qtpy import QtCore

class SignalLauncherAutoLockInput(SignalLauncher):
    """
    A SignalLauncher for the autolock
    """
    update_autolock_plot = QtCore.Signal()
    
# Any InputSignal must define a class that contains the function "expected_signal(variable)" that returns the expected
# signal value as a function of the variable value. This function ensures that the correct setpoint and a reasonable
# gain is chosen (from the derivative of expected_signal) when this signal is used for feedback.
class AutoLockInput(InputSignal):
    """ An input signal that is interpreted using previously recorded data from
    a calibration scan. """
    
    _widget_class = AutoLockInputWidget
    _signal_launcher = SignalLauncherAutoLockInput
    _setup_attributes = ["autolock_setpoint_x", "points_on_slope",
                         "search_pattern_xmin", "search_pattern_xmax"]
    _gui_attributes = ["autolock_setpoint_x", "points_on_slope",
                         "search_pattern_xmin", "search_pattern_xmax"]
    autolock_setpoint_x = FloatProperty(default=0., doc = "position of the "
                                        "autolock setpoint on the x-axis")
    points_on_slope = IntProperty(default=100, min=0, doc = "number of "
                                  "datapoints around the setpoint to use "
                                  "when fitting the slope of the errorsignal")
    search_pattern_xmin = FloatProperty(default=-1, call_setup=True,
                                              doc = "xmin value of the pattern"
                                              "that is used when searching the "
                                              "lockpoint")
    search_pattern_xmax = FloatProperty(default=1, call_setup=True,
                                              doc = "xmax value of the pattern"
                                              "that is used when searching the "
                                              "lockpoint")
    
    def __init__(self, parent, name=None):
        self.scope_data_function = lambda x: np.array(x)*0 # return a flat function by default
        super(AutoLockInput, self).__init__(parent, name=name)
        
    def expected_signal(self, variable):
        return self.scope_data_function(variable)

# TODO: think of something that is more clever than the default numeric derivative on discrete data
#    def expected_slope(self, variable):
#        return 2.0 * self.custom_gain_attribute * self.lockbox.custom_attribute * variable
    def _setup(self):
        self._signal_launcher.update_autolock_plot.emit()
        
    def sweep_acquire(self):
        """
        Here we need to acquire the actuator signal along with the errorsignal.
        """
        try:
            with self.pyrpl.scopes.pop(self.name) as scope:
                self.lockbox._sweep()
                if "sweep" in scope.states:
                    scope.load_state("sweep")
                else:
                    scope.setup(input1=self.signal(),
                                input2=self.lockbox.outputs[self.lockbox.default_sweep_output].pid.output_direct,
                                trigger_source=self.lockbox.asg.name,
                                trigger_delay=0,
                                duration=1./self.lockbox.asg.frequency,
                                ch1_active=True,
                                ch2_active=True,
                                average=True,
                                trace_average=1,
                                running_state='stopped',
                                rolling_mode=False)
                    scope.save_state("autosweep")
                error_signal, actuator_signal = scope.curve(timeout=1./self.lockbox.asg.frequency+scope.duration)
                times = scope.times
                error_signal -= self.calibration_data._analog_offset
                return times, error_signal, actuator_signal
        except InsufficientResourceError:
            # scope is blocked
            self._logger.warning("No free scopes left for sweep_acquire. ")
            return None, None
        
    def calibrate(self, autosave=True):
        """  """
        times, error_signal, actuator_signal = self.sweep_acquire()
        if error_signal is None:
            self._logger.warning('Aborting calibration because no scope is available...')
            return None
        self.calibration_data.get_stats_from_curve(error_signal)
        self.calibration_data.error_signal = error_signal
        self.calibration_data.actuator_signal = actuator_signal
        self.scope_data_function = lambda x: np.interp(x, actuator_signal, error_signal)
        self.plot_range = actuator_signal
        # log calibration values
        self._logger.info("%s calibration successful - Min: %.3f  Max: %.3f  Mean: %.3f  Rms: %.3f",
                          self.name,
                          self.calibration_data.min,
                          self.calibration_data.max,
                          self.calibration_data.mean,
                          self.calibration_data.rms)
        # update graph in lockbox
        self.lockbox._signal_launcher.input_calibrated.emit([self])

        # save data if desired
        if autosave:
            params1 = dict(self.calibration_data.setup_attributes) # make a copy
            params1['name'] = self.name+"_calibration_errorsignal"
            params2 = dict(params1) 
            params2['name'] = self.name+"_calibration_actuatorsignal"
            newcurve1 = self._save_curve(times, error_signal, **params1)
            newcurve2 = self._save_curve(times, actuator_signal, **params2)
            return newcurve1 # not sure which one to return here?
        else:
            return None
        
# this function was used for testing the function call feature of the stages
    def sweep_and_find_setpoint(self):
        print('Sweeping now')
        print('Determining output voltage at setpoint')
        offset = 1.5
        output = self.lockbox.outputs.output
        print(output)
        print(self.lockbox.sequence[0].outputs[output.name])
        self.lockbox.sequence[1].outputs[output.name].offset = offset

class AutoLockOutput(OutputSignal):
    pass

class AutoLock(Lockbox):
    """ 
    A lockbox that records a sweep of the errorsignal and lets the user choose 
    a lockpoint interactively. A segment of the errorsignal sweep can be 
    defined that is used to identify the lock point in a potentially 
    complicated errorsignal with several similar features. To avoid locking at
    a wrong position, a new sweep is recorded to determine a good starting 
    output value for the PID controller before every relock attempt. If desired
    several sweep steps with decreasing amplitude (given by 
    autolock_sweep_zoomfactor) can be performed.
    """

    inputs = LockboxModuleDictProperty(autolock_input=AutoLockInput,
                                       secondary_input=InputDirect)

    outputs = LockboxModuleDictProperty(autolock_output=AutoLockOutput, 
                                        secondary_output=OutputSignal)

    # sweep properties
    autolock_sweep_amplitude = FloatProperty(default=1., min=-1, max=1, call_setup=True)
    autolock_sweep_offset = FloatProperty(default=0.0, min=-1, max=1, call_setup=True)
    autolock_sweep_frequency = FrequencyProperty(default=10.0, call_setup=True)
    autolock_sweep_steps = IntProperty(default=1, min=1, call_setup=True,
                                       doc="How many sweeps to perform at "
                                       "different amplitude")
    autolock_sweep_zoomfactor = FloatProperty(default=0.5, min=1e-4, max=1,
                                              call_setup=True,
                                              doc="if sweep steps > 1, "
                                              "multiply amplitude with this "
                                              "factor every time")

    # list of attributes that are mandatory to define lockbox state. setup_attributes of all base classes and of all
#    # submodules are automatically added to the list by the metaclass of Module
    _setup_attributes = ["autolock_sweep_amplitude", "autolock_sweep_offset", 
                         "autolock_sweep_frequency", "autolock_sweep_steps", 
                         "autolock_sweep_zoomfactor"]
#    # attributes that are displayed in the gui. _gui_attributes from base classes are also added.
    _gui_attributes = ["autolock_sweep_amplitude", "autolock_sweep_offset", 
                         "autolock_sweep_frequency", "autolock_sweep_steps", 
                         "autolock_sweep_zoomfactor"]
    
    def __init__(self, parent, name=None):
        super(AutoLock, self).__init__(parent=parent, name=name)
        pass # nothing to do so far

    def lock(self, **kwargs):
        """
            Performs an autolock sweep first and adjusts the starting output 
            value (offset) for the first stage accordingly.
            
            After that, same as in Lockbox:
            Launches the full lock sequence, stage by stage until the end.
            optional kwds are stage attributes that are set after iteration through
            the sequence, e.g. a modified setpoint.
        """
        # here we should do a sweep, record the signal, do matching, 
        # display it in the input signal calibration graph, determine initial
        # output value for locking
        print("doing autolock sweep procedure now")
        # test setting initial output value
        self.sequence[1].outputs[self.outputs.autolock_output.name].offset = 1
        super(AutoLock, self).lock(**kwargs)

