# these imports are the standard imports for required for derived lockboxes
from pyrpl.software_modules.lockbox import *
from pyrpl.software_modules.loop import *
import numpy as np
from ....widgets.module_widgets import AutoLockInputWidget
from ....modules import SignalLauncher
from qtpy import QtCore
import dtw

import matplotlib.pyplot as plt

class SignalLauncherAutoCalibrateInput(SignalLauncher):
    """
    A SignalLauncher for the autolock
    """
    update_plots = QtCore.Signal()
    input_acquired = QtCore.Signal(list)
    

class AutoCalibrationData(CalibrationData):
    """ class to hold the calibration data of the autocalibration input """
    _setup_attributes = ["calibration_datasets", "output_at_lock_point", 
                         "slope_at_lock_point"]
    _gui_attributes = []
    
    calibration_datasets = DataProperty(default=[], doc="data acquired during "
                                                       "calibration scans")
    output_at_lock_point = FloatProperty(default=0, doc="output voltage at the"
                                                         " lockpoint")
    slope_at_lock_point = FloatProperty(default=1, doc="error signal change "
                                                       "per output voltage change"
                                                       "at the lockpoint")
    def clean_up_datasets(self):
        #TODO: remove overlapping regions in datasets
        print('cleaning up datasets')
        pass


    
    
class AutoCalibrateInput(InputSignal):
    """ 
    A lockbox input that records a sweep of the error signal and lets the user 
    choose a lock point interactively. A segment of this data is used as a 
    calibration search pattern. Whenever a new calibration is performed, this
    pattern is used to reliably find the same lock point within the signal.
    The slope of the error signal at the lock point is determined using a 
    linear fit. 
    
    If desired, several sweep steps with decreasing amplitude (given by 
    autolock_sweep_zoomfactor) can be performed during calibration to obtain 
    more accurate data close to the lock point.
    
    The output voltage at the lock point is written to the first locking stage
    as an offset, if the checkbox jump_to_lockpoint_in_first_stage is activated.
    """
    
    _widget_class = AutoCalibrateInputWidget
    _signal_launcher = SignalLauncherAutoCalibrateInput
    _setup_attributes = ["setpoint_x", "slope_interval",
                         "search_pattern_xmin", "search_pattern_xmax",
                         "definition_signal"]
    _gui_attributes = ["setpoint_x", "slope_interval",
                         "search_pattern_xmin", "search_pattern_xmax"]
    calibration_data = ModuleProperty(AutoCalibrationData)

    setpoint_x = FloatProperty(default=0., doc = "position of the lock point"
                                                 "on the x-axis in the "
                                                 "definition data")
    slope_interval = FloatProperty(default=0.1, min=0, doc = "interval "
                                  "on the x-axis for fitting the slope of the "
                                  "errorsignal around the setpoint")
    search_pattern_xmin = FloatProperty(default=-1, call_setup=True,
                                              doc = "xmin value of the pattern"
                                              "that is used when searching the "
                                              "lockpoint")
    search_pattern_xmax = FloatProperty(default=1, call_setup=True,
                                              doc = "xmax value of the pattern"
                                              "that is used when searching the "
                                              "lockpoint")
    setpoint_definition_data = DataProperty(default=[], 
                                            doc="error signal and actuator "
                                                "signal for defining the "
                                                "setpoint")
    jump_to_lockpoint_in_first_stage = BoolProperty(default=True, 
                                                 doc="whether to write the "
                                                     "setpoint output voltage "
                                                     "as the offset during "
                                                     "the first stage "
                                                     "after calibration")
    
    # calibration sweep properties
    calibration_sweep_amplitude = FloatProperty(default=1., min=-1, max=1)
    calibration_sweep_offset = FloatProperty(default=0.0, min=-1, max=1)
    calibration_sweep_frequency = FrequencyProperty(default=10.0)
    calibration_sweep_steps = IntProperty(default=1, min=1, 
                                       doc="How many sweeps to perform at "
                                       "different amplitude")
    calibration_sweep_zoomfactor = FloatProperty(default=0.5, min=1e-4, max=1,
                                              doc="if sweep steps > 1, "
                                              "multiply amplitude with this "
                                              "factor every time")
    
    def __init__(self, parent, name=None):
        self.data_function = lambda x: np.array(x)*0 # return a flat function by default
        super(AutoLockInput, self).__init__(parent, name=name)
        
    def expected_signal(self, variable):
        return self.data_function(variable)

    def expected_slope(self, variable):
        return self.calibration_data.slope_at_lock_point
    
    def _setup(self):
        self._signal_launcher.update_autolock_plot.emit()

    @property
    def sweep_output(self):
        return self.lockbox.outputs[self.lockbox.default_sweep_output]
    
    @property
    def lockpoint_search_pattern(self):
        actuator, error = self.setpoint_definition_data
        mask = (actuator > self.search_pattern_xmin) & \
               (actuator < self.search_pattern_xmax)
        return actuator[mask], error[mask]
        
    def _get_scope_data(self):
        """
        Get the scope data for a single trigger when a sweep is already running
        """
        try:
            with self.pyrpl.scopes.pop(self.name) as scope:
                scope.setup(input1=self.signal(),
                            input2=self.sweep_output.pid.output_direct,
                            trigger_source=self.lockbox.asg.name,
                            trigger_delay=0,
                            duration=1./self.lockbox.asg.frequency,
                            ch1_active=True,
                            ch2_active=True,
                            average=True,
                            trace_average=1,
                            running_state='stopped',
                            rolling_mode=False)
                # scope.save_state("calibrate")
                error_signal, actuator_signal = scope.curve(timeout=1./self.lockbox.asg.frequency+scope.duration)
                times = scope.times
                error_signal -= self.calibration_data._analog_offset
                # cut out the rising slope
                istart = np.argmin(actuator_signal)
                istop = np.argmax(actuator_signal[istart:])+istart
                error_signal = error_signal[istart:istop]
                actuator_signal = actuator_signal[istart:istop]
                times = times[istart:istop]
                return times, error_signal, actuator_signal
        except InsufficientResourceError:
            # scope is blocked
            self._logger.warning("No free scopes left for sweep_acquire. ")
            return None, None, None
        
    def get_setpoint_definition_data(self, autosave=True):
        """
        Acquire the actuator signal along with the errorsignal over the full 
        calibration sweep range.
        
        The purpose of this function is to record a full range sweep so that 
        the user can then select a setpoint and search pattern.
        """
        self.lockbox.auto_calibration_sweep(self.calibration_sweep_amplitude,
                                            self.calibration_sweep_offset,
                                            self.calibration_sweep_frequency)
        times, error_signal, actuator_signal = self._get_scope_data()
        self.lockbox.unlock() # after we have our data, sweeping can end
        if error_signal is None:
            self._logger.warning('Aborting acquisition because no scope is available...')
            return None
        self.setpoint_definition_data = [actuator_signal, error_signal]

        self.data_function = lambda x: np.interp(x, actuator_signal, 
                                                 error_signal)

        plt.figure()
        plt.plot(times, actuator_signal)
        plt.show()
        plt.figure()
        plt.plot(times, error_signal)
        plt.show()
        plt.figure()
        plt.plot(actuator_signal, self.data_function(actuator_signal), 'r-')
        plt.show()
        
        self.plot_range = np.linspace(min(actuator_signal), max(actuator_signal), 1000)

        self._logger.info("%s successfully obtained setpoint definition data",
                          self.name)
        # update graphs in widget
        self._signal_launcher.update_plots.emit()
        # save data if desired
        if autosave:
            params1 = dict(self.calibration_data.setup_attributes) # make a copy
            params1['name'] = self.name+"_setpoint_definition_errorsignal"
            params2 = dict(params1) 
            params2['name'] = self.name+"_setpoint_definition_actuatorsignal"
            newcurve1 = self._save_curve(times, error_signal, **params1)
            newcurve2 = self._save_curve(times, actuator_signal, **params2)
            return newcurve1 # not sure which one to return here?
        else:
            return None
        
    def calibrate(self, autosave=True):
        """
        Determine the position and slope of the lockpoint using the search
        pattern defined by the setpoint definition data.
        
        If autolock_sweep_steps > 1 we want to sequentially zoom into the 
        signal to get better resolution.
        """
        lb = self.lockbox # shortcut
        
        # check if we have setpoint definition data yet
        if len(self.setpoint_definition_data) == 0:
            self._logger.error("No setpoint definition dataset yet.")
            return None
        
        # initialize variables
        search_pattern = tw_search_pattern = self.lockpoint_search_pattern
        sweep_offset = self.calibration_sweep_offset
        sweep_amplitude = self.calibration_sweep_amplitude
        self.calibration_data.calibration_datasets = [] # reset to empty list
        self.calibration_data.calibrated_lockpoint = self.setpoint_x
        
        # calibration sweeps
        for i_sweep in self.calibration_sweep_steps:
            # scan and get signal
            lb.auto_calibration_sweep(sweep_amplitude,
                                      sweep_offset,
                                      self.calibration_sweep_frequency)
            times, error_signal, actuator_signal = self._get_scope_data()
            lb.unlock() # after we have our data, sweeping can end
            
            # pattern matching using dynamic time warping
            # first resample new signal to match the search pattern
            # this should be better in terms of speed and robustness
            dV = np.ptp(search_pattern[0])/(np.size(search_pattern[0])-1)
            actuator_signal_resampled = np.arange(min(actuator_signal), 
                                                  max(actuator_signal), dV)
            error_signal_resampled = np.interp(actuator_signal_resampled, actuator_signal, 
                                               error_signal)
            # match the search pattern to the resampled signal
            # use the dtw-python library
            result = dtw.dtw(search_pattern[1], error_signal_resampled,
                             open_end=True, open_begin=True, 
                             step_pattern='asymmetric')
            # result is an alignment object:
            # https://dynamictimewarping.github.io/py-api/html/api/dtw.D_T_W_.html#dtw.DTW
            # this is how we get the matching actuator values
            matching_actuator_values = actuator_signal_resampled[result.index2]
            # compute the "time-warp" transformation: maps actuator values
            # from the search pattern to actuator values in the new data
            time_warp = lambda x: np.interp(x, search_pattern[0], 
                                            matching_actuator_values)
            # apply this to the calibrated lock point
            # setpoint_x has to be inside the search pattern for this to work
            self.calibration_data.output_at_lock_point = time_warp(self.setpoint_x)
            # the timewarped search pattern should contain the original 
            # setpoint definition data, but with the x-axis scaled (time-warped)
            tw_search_pattern[0] = time_warp(tw_search_pattern[0])
            # update the search pattern for the next step
            # use the new data which is more accurate
            pattern_mask = (actuator_signal >= min(matching_actuator_values)) & \
                           (actuator_signal <= max(matching_actuator_values))
            search_pattern[0] = actuator_signal[pattern_mask]
            search_pattern[1] = error_signal[pattern_mask]
            # determine slope around lockpoint
            a_min = self.calibration_data.output_at_lock_point - slope_interval/2.
            a_max = self.calibration_data.output_at_lock_point + slope_interval/2.
            slope_mask = (actuator_signal >= a_min) & \
                         (actuator_signal <= a_max)
            slope = np.polyfit(actuator_signal[slope_mask], 
                               error_signal[slope_mask], deg=1)[0]
            self.calibration_data.slope_at_lock_point = slope
            
            # clean up data
            self.calibration_data.clean_up_datasets()
            
            # update sweep range for next step 
            sweep_amplitude *= self.calibration_sweep_zoomfactor
            # we always want to use the full amplitude for sweeping even if 
            # this shifts the lock point out of the center
            sweep_offset = np.clip(self.calibration_data.output_at_lock_point,
                                   self.sweep_output.min_voltage + amplitude,
                                   self.sweep_output.max_voltage - amplitude)
            
            self.lockbox._signal_launcher.update_plots.emit()
            
        times, error_signal, actuator_signal = self._get_scope_data()
        self.lockbox.unlock() # after we have our data, sweeping can end
        # cut out the rising slope
        istart = np.argmin(actuator_signal)
        istop = np.argmax(actuator_signal[istart:])+istart
        error_signal = error_signal[istart:istop]
        actuator_signal = actuator_signal[istart:istop]
        times = times[istart:istop]
        if error_signal is None:
            self._logger.warning('Aborting calibration because no scope is available...')
            return None
        self.calibration_data.get_stats_from_curve(error_signal)
        self.calibration_data.error_signal = error_signal
        self.calibration_data.actuator_signal = actuator_signal
        self.scope_data_function = lambda x: np.interp(x, actuator_signal, error_signal)
        self.plot_range = np.linspace(min(actuator_signal), max(actuator_signal), 1000)
        # log calibration values
        self._logger.info("%s calibration successful - Min: %.3f  Max: %.3f  Mean: %.3f  Rms: %.3f",
                          self.name,
                          self.calibration_data.min,
                          self.calibration_data.max,
                          self.calibration_data.mean,
                          self.calibration_data.rms)
        # update graph in lockbox
        self.lockbox._signal_launcher.input_calibrated.emit([self])
        # set stage0 offset
        if self.jump_to_lockpoint_in_first_stage:
            name = self.sweep_output.name
            stage0 = lb.sequence[1]
            stage0.outputs[name].offset = self.calibrated_lockpoint
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


class AutoLockOutput(OutputSignal):
    pass

class AutoCalibrateLockbox(Lockbox):
    """ 
    A lockbox that uses an autocalibrate input and offers the option to perform
    an autocalibration before every locking attempt.
    """

    inputs = LockboxModuleDictProperty(autocalibrate_input=AutoCalibrateInput)

    # list of attributes that are mandatory to define lockbox state. setup_attributes of all base classes and of all
#    # submodules are automatically added to the list by the metaclass of Module
    _setup_attributes = ["autolock_sweep_amplitude", "autolock_sweep_offset", 
                         "autolock_sweep_frequency", "autolock_sweep_steps", 
                         "autolock_sweep_zoomfactor"]
#    # attributes that are displayed in the gui. _gui_attributes from base classes are also added.
    _gui_attributes = ["autolock_sweep_amplitude", "autolock_sweep_offset", 
                         "autolock_sweep_frequency", "autolock_sweep_steps", 
                         "autolock_sweep_zoomfactor"]
    always_calibrate_before_locking = BoolProperty(default=True)
    
    def __init__(self, parent, name=None):
        super(AutoLock, self).__init__(parent=parent, name=name)
        pass # nothing to do so far
        
    def auto_calibration_sweep(self, amplitude, offset, frequency):
        # set up the output and arbitrary signal generator (asg)
        self.unlock()
        output = self.outputs[self.default_sweep_output]
        output.unlock(reset_offset=True) # for safety, this sets all pid parameters to zero
        output.pid.input = self.lockbox.asg
        self.asg.setup( amplitude=amplitude,
                        offset=offset,
                        frequency=frequency,
                        waveform='ramp', # autolock waveform is always a triangular (ramp)
                        trigger_source='immediately',
                        cycles_per_burst=0)
        output.pid.setpoint = 0.
        output.pid.p = 1.
        output.current_state = 'sweep'

    def lock(self, **kwargs):
        """
            If checkbox is enabled, calibrates all channels before locking.
            
            After that, same as in Lockbox:
            Launches the full lock sequence, stage by stage until the end.
            optional kwds are stage attributes that are set after iteration through
            the sequence, e.g. a modified setpoint.
        """
        self.calibrate_all()               
        super(AutoLock, self).lock(**kwargs)

