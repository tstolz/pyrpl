# these imports are the standard imports for required for derived lockboxes
from pyrpl.software_modules.lockbox import *
from pyrpl.software_modules.loop import *
import numpy as np
from ....widgets.module_widgets import AutoCalibrateInputWidget
from ....modules import SignalLauncher
from ....attributes import DataProperty
from qtpy import QtCore
import dtw

class SignalLauncherAutoCalibrateInput(SignalLauncher):
    """
    A SignalLauncher for the autocalibrate input.
    """
    update_plots = QtCore.Signal()
    

class AutoCalibrationData(CalibrationData):
    """ class to hold the calibration data of the autocalibration input """
    _setup_attributes = ["calibration_datasets", "lock_point_x", 
                         "slope_at_lock_point"]
    _gui_attributes = []
    
    calibration_datasets = DataProperty(default=[], doc="data acquired during "
                                                       "calibration scans")
    lock_point_x = FloatProperty(default=0, doc="output voltage to get to the "
                                              "lockpoint")
    lock_point_y = FloatProperty(default=0, doc="projected errorsignal at the "
                                              "lockpoint")
    slope_at_lock_point = FloatProperty(default=1, doc="error signal change "
                                                       "per output voltage change"
                                                       "at the lockpoint")
    scaled_search_pattern = DataProperty(default=[], 
                                             doc="search pattern scaled "
                                                 "to calibration scans")
    def clean_up_datasets(self):
        """ removes overlapping regions in datasets, keeping always the last
        recorded data. """
        # could do something like
        # xmin = np.inf
        # xmax = -np.inf
        # for i in range(len(self.calibration_datasets)-1, -1, -1):
        #     dset = self.calibration_datasets[i]
        #     keep = (dset[0] < xmin) | (dset[0] > xmax)
        #     xmin = min(dset[0])
        #     xmax = max(dset[0])
        #     self.calibration_datasets[i] = [dset[0][keep], dset[1][keep]]
        # but actually it is nice to see all the datasets for debugging
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
    _setup_attributes = ["lockpoint_x", "slope_interval",
                         "search_pattern_xmin", "search_pattern_xmax",
                         "jump_to_lockpoint_in_first_stage",
                         "calibration_sweep_amplitude",
                         "calibration_sweep_offset",
                         "calibration_sweep_frequency",
                         "calibration_sweep_steps",
                         "calibration_sweep_zoomfactor"]
    _gui_attributes = ["lockpoint_x", "slope_interval",
                         "search_pattern_xmin", "search_pattern_xmax",
                         "jump_to_lockpoint_in_first_stage",
                         "calibration_sweep_amplitude",
                         "calibration_sweep_offset",
                         "calibration_sweep_frequency",
                         "calibration_sweep_steps",
                         "calibration_sweep_zoomfactor"]
    calibration_data = ModuleProperty(AutoCalibrationData)

    lockpoint_x = FloatProperty(default=0., call_setup = True, increment=1e-4,
                               doc = "position of the lock point on the x-axis in the definition data")
    slope_interval = FloatProperty(default=0.1, min=0, call_setup = True,
                                   doc = "interval "
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
    setpoint_definition_data = DataProperty(default=[[],[]], 
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
        super(AutoCalibrateInput, self).__init__(parent, name=name)
        
    def expected_signal(self, variable):
        """
        We don't rely on a physical model of the system for the 
        AutoCalibrateLockbox. This means the physical quantity we want to 
        stabilize is actually the input voltage and we regard the output 
        voltage as our actuator that affects the input in a black-box-fashion. 
        Therefore, we don't want to have the additional conversion to and from
        the physical model variable that the other lockbox classes provide. 
        This means using an identity function here.
        """
        return variable 
    
    def expected_slope(self, variable):
        """
        Slope of the error signal voltage with respect to the output voltage. 
        This is required to estimate the external loop gain in output.py.
        """
        return self.calibration_data.slope_at_lock_point
    
    def _setup(self):
        # check if search pattern boundaries are sorted
        if self.search_pattern_xmin > self.search_pattern_xmax:
            tmp = self.search_pattern_xmin
            self.search_pattern_xmin = self.search_pattern_xmax
            self.search_pattern_xmax = tmp
        # check if setpoint lies within search pattern boundaries
        if self.lockpoint_x < self.search_pattern_xmin or self.lockpoint_x < self.search_pattern_xmin:
            self.lockpoint_x = (self.search_pattern_xmin + self.search_pattern_xmax)/2
        self._signal_launcher.update_plots.emit()

    @property
    def lockpoint_y(self):
        # this is just for plotting, so just interpolate
        if len(self.setpoint_definition_data[0])>0:
            return np.interp(self.lockpoint_x, *self.setpoint_definition_data)
        else:
            return 0
            
    @property
    def sweep_output(self):
        return self.lockbox.outputs[self.lockbox.default_sweep_output]
    
    @property
    def lockpoint_search_pattern(self):
        actuator, error = np.asarray(self.setpoint_definition_data)
        mask = (actuator > self.search_pattern_xmin) & \
               (actuator < self.search_pattern_xmax)
        return [actuator[mask], error[mask]]
        
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
        Acquire the error signal along with the actuator signal over the full 
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
        tw_search_pattern = np.copy(self.lockpoint_search_pattern)
        search_pattern = np.copy(self.lockpoint_search_pattern)
        sweep_offset = self.calibration_sweep_offset
        sweep_amplitude = self.calibration_sweep_amplitude
        self.calibration_data.calibration_datasets = [] # reset to empty list
        self.calibration_data.lock_point_x = self.lockpoint_x
        
        # calibration sweeps
        for i_sweep in range(self.calibration_sweep_steps):
            # scan and get signal
            lb.auto_calibration_sweep(sweep_amplitude,
                                      sweep_offset,
                                      self.calibration_sweep_frequency)
            times, error_signal, actuator_signal = self._get_scope_data()
            lb.unlock() # after we have our data, sweeping can end
            if error_signal is None:
                self._logger.warning('Aborting calibration because no scope is available...')
                return None
            
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
            # lockpoint_x has to be inside the search pattern for this to work
            self.calibration_data.lock_point_x = time_warp(self.calibration_data.lock_point_x)
            # the timewarped search pattern should contain the original 
            # setpoint definition data, but with the x-axis scaled (time-warped)
            tw_search_pattern[0] = time_warp(tw_search_pattern[0])
            self.calibration_data.scaled_search_pattern = tw_search_pattern
            
            # update the search pattern for the next step
            # use the new data which is more accurate
            pattern_mask = (actuator_signal >= min(matching_actuator_values)) & \
                           (actuator_signal <= max(matching_actuator_values))
            search_pattern = np.array([actuator_signal[pattern_mask], 
                                       error_signal[pattern_mask]])
            
            # determine slope around lockpoint
            a_min = self.calibration_data.lock_point_x - self.slope_interval/2.
            a_max = self.calibration_data.lock_point_x + self.slope_interval/2.
            slope_mask = (actuator_signal >= a_min) & \
                         (actuator_signal <= a_max)
            slope, offset = np.polyfit(actuator_signal[slope_mask], 
                               error_signal[slope_mask], deg=1)
            self.calibration_data.slope_at_lock_point = slope
            # lock_point_y is just for plotting, will not affect the actual
            # setpoint of the PID controller which is set in the stages
            self.calibration_data.lock_point_y = offset + slope*self.calibration_data.lock_point_x
            self.calibration_data.calibration_datasets.append([actuator_signal, error_signal])
            
            # clean up data
            self.calibration_data.clean_up_datasets()
            
            # update sweep range for next step 
            sweep_amplitude *= self.calibration_sweep_zoomfactor
            # we always want to use the full amplitude for sweeping even if 
            # this shifts the lock point out of the center
            sweep_offset = np.clip(self.calibration_data.lock_point_x,
                                   self.sweep_output.min_voltage + sweep_amplitude,
                                   self.sweep_output.max_voltage - sweep_amplitude)
            
            self._signal_launcher.update_plots.emit()
            
        # set stage0 offset
        if self.jump_to_lockpoint_in_first_stage:
            name = self.sweep_output.name
            stage0 = lb.sequence[0]
            stage0.outputs[name].offset = self.calibration_data.lock_point_x
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
            
class AutoCalibrateLockbox(Lockbox):
    """ 
    A lockbox that uses an autocalibrate input and offers the option to perform
    an autocalibration before every locking attempt.
    """

    inputs = LockboxModuleDictProperty(autocalibrate_input=AutoCalibrateInput)

    # list of attributes that are mandatory to define lockbox state. setup_attributes of all base classes and of all
#    # submodules are automatically added to the list by the metaclass of Module
    _setup_attributes = ["always_calibrate_before_locking"]
#    # attributes that are displayed in the gui. _gui_attributes from base classes are also added.
    _gui_attributes = ["always_calibrate_before_locking"]
    always_calibrate_before_locking = BoolProperty(default=True)
        
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
            If checkbox is enabled, calibrates autocalibrate input before locking.
            
            After that, same as in Lockbox:
            Launches the full lock sequence, stage by stage until the end.
            optional kwds are stage attributes that are set after iteration through
            the sequence, e.g. a modified setpoint.
        """
        if self.always_calibrate_before_locking:
            self.inputs.autocalibrate_input.calibrate()               
        super(AutoCalibrateLockbox, self).lock(**kwargs)

