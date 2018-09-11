from __future__ import print_function
from transitions import Machine
import logging
import inspect
import time
import datetime
# import functools
import sys


logging.basicConfig()


class Fridge(Machine):
    fridge_states = ['pump_hot_cooling_mainplate',
                     'pump_warming',
                     'pump_warming_cooling_mainplate',
                     'cooling_pump',
                     'cold',
                     'heat_switch_cooling',
                     'system_warm',
                     'warming_up',
                     'manual',
                     'unknown'
                    ]

    def heat_switch_off(self):
        self.fridge.heater_control('HS', False)
        print("heat switch off")

    def heat_switch_on(self):
        self.fridge.heater_control('HS', True)
        print("heat switch on")

    def pump_off(self):
        self.fridge.heater_control('HP', False)
        print("heat pump off")

    def pump_on(self):
        self.fridge.heater_control('HP', True)
        print("heat pump on")

    def pump_warm_up(self):
        try:
            self.fridge.HP_warm_up()  # This works for CTC
        except AttributeError:
            self.fridge.heater_control('HP', True)
        print("heat pump on to warmup")

    def compressor_control(self, state):
        if self.fridge.config['control_compressor']:
            self.fridge.compressor_control(state)

    def on_enter(self, methodname):
        name = methodname.split('on_enter_')[-1]

        newtime = time.time()
        dt = (newtime - self.time_last_transition)/60.
        self.time_last_transition = newtime
        msg = 'Enter state ' + \
              '%s, spent %.2f minutes in previous state' % (name, dt)
        if self.logger:
            self.logger.warning(msg)
        if self.console:
            self.console.writeline(time.asctime()+': '+msg)

    def on_enter_manual(self):
        self.on_enter(inspect.stack()[0][3])
        print(inspect.stack()[0][3])
        print('enter manual mode')

    def on_enter_heat_switch_cooling(self):
        # Turn off heat switch so that it can start cooling
        self.on_enter(inspect.stack()[0][3])
        self.heat_switch_off()

    def on_enter_pump_warming(self):
        # Turn on pump
        self.on_enter(inspect.stack()[0][3])
        self.pump_on()
        if 'timed_pump_heater' in self.config:
            if self.config['timed_pump_heater']:
                self.console.writeline('timed pump heater')
    def on_enter_pump_warming_cooling_mainplate(self):
        # Turn off pump to let mainplate cool
        self.on_enter(inspect.stack()[0][3])
        self.pump_off()

    def on_enter_pump_hot_cooling_mainplate(self):
        # Turn off pump it is hot enough to let mainplate cool
        self.on_enter(inspect.stack()[0][3])
        if 'pump_heater_on_while_hot' in self.config:
            if not self.config['pump_heater_on_while_hot']:
                self.pump_off()
        else:
            self.pump_off()

    def on_enter_cooling_pump(self):
        #  Start cooling to 1K by turning on heat switch
        self.pump_off()
        self.on_enter(inspect.stack()[0][3])
        self.heat_switch_on()

    def on_enter_cold(self):
        self.on_enter(inspect.stack()[0][3])

    def on_enter_system_warm(self):
        self.on_enter(inspect.stack()[0][3])
        self.fridge.heater_control('HS', False)
        self.fridge.heater_control('HP', False)
        #  Need to turn off the logging...
        #   did this in webFridge

    def on_enter_warming_up(self):
        self.on_enter(inspect.stack()[0][3])
        self.fridge.heater_control('HS', True)
        self.pump_warm_up()
        # self.fridge.heater_control('HP', True)
        self.compressor_control(False)

    def on_enter_unknown(self):
        self.on_enter(inspect.stack()[0][3])

    def __init__(self, fridge):
        states = ['pump_hot_cooling_mainplate',
                  'pump_warming',
                  'pump_warming_cooling_mainplate',
                  'cooling_pump',
                  'cold',
                  'heat_switch_cooling',
                  'system_warm',
                  'warming_up',
                  'manual',
                  'unknown'
                  ]

        Machine.__init__(self, states=states,
                         initial='manual')
        self.fridge = fridge
        self.config = fridge.config
        self.manual_mode = True
        self.time_last_transition = time.time()
        self.logger = None
        self.console = None
        # self.logger = logging.getLogger('fridge_machine')
        # self.logger.setLevel(logging.INFO)
        # self.logger.info('logging from fridge_machine')
        self.recycle_hour = self.config['recycle_time']
        self.recycle_period = \
            datetime.timedelta(days=self.config['recycle_period'])
        self.update_next_recycle_time()

    def update_next_recycle_time(self):
        self.next_recycle_time = datetime.datetime.now()
        if self.recycle_hour < 0:
            return
        new_time = self.next_recycle_time.replace(hour=self.recycle_hour,
                                                  minute=0, second=0)
        if self.next_recycle_time > new_time:
            self.next_recycle_time = new_time + self.recycle_period
        else:
            self.next_recycle_time = new_time

    def set_logger(self, logger):
        self.logger = logger

    def set_console(self, console):
        self.console = console

    def check_if_time_to_recycle(self):
        if self.recycle_hour >= 0:
            now = datetime.datetime.now()
            dt = now-self.next_recycle_time
            if dt.total_seconds() < 0:
                return False
            else:
                msg = 'Auto recycle at a certain hour'
                if self.logger:
                    self.logger.warning(msg)
                if self.console:
                    self.console.writeline(time.asctime()+': '+msg)
                self.next_recycle_time = \
                    self.next_recycle_time.replace(hour=self.recycle_hour,
                                                   minute=0) + \
                    self.recycle_period
                return True
        return False

    def update(self, datastr):
        temperatures = self.datastr_to_dict(datastr)
        print('In fridge_machine update state:', self.state)
        # self.logger.debug('fridge_state update %r' % self.state)
        #
        if self.is_manual():
            #  Don't do anythin
            pass
        elif self.is_unknown():
            # self.logger.debug('trying to figure the state things are in')
            #  Only move to cold if the state is unkown...
            try:
                if temperatures['1K'] < 1:
                    self.to_cold()
            except:
                pass
        elif self.is_cold():
            #  Check if temperature is cold and time to recycle
            #  else check if time to recycle
            if temperatures['1K'] < self.config['temps']['T1Khigh']:
                # things still cold
                #  But perhaps we should recycle anyway...
                #  check if recycle hour is not negative
                print('cold, but need to check if time to recycle')
                if self.check_if_time_to_recycle():
                    self.to_heat_switch_cooling()
            else:
                # Check if we should recycle or wait?
                if self.check_if_time_to_recycle():
                    self.to_heat_switch_cooling()
                elif self.recycle_hour < 0:  # recycle if recycle_time<0
                    self.to_heat_switch_cooling()
        elif self.is_heat_switch_cooling():
            #  Sometimes SWITCH thermometer is broken / not working
            #     if there is not thermometer, wait five minutes
            #     Check if heat switched has finised cooling off
            if 'SWITCH' in temperatures:
                if temperatures['SWITCH'] < self.config['temps']['Tswitch']:
                    #  Heat switch has cooled off, can start to heat the pump
                    self.to_pump_warming()
            else:
                # No switch thermometry Wait five minutes...
                newtime = time.time()
                dt = (newtime - self.time_last_transition)
                if dt > 300:
                    self.to_pump_warming()
        elif self.is_pump_warming():
            #   If possible check if the heat switch is getting to hot if so
            #      turn of pump heaters
            #   if not check if pump is to temperature
            if 'timed_pump_heater' in self.config:
                if self.config['timed_pump_heater']:
                    # No pump thermometry turn on pump for set time...
                    newtime = time.time()
                    pump_on_duration = 180
                    if 'pump_heater_length_time_on' in self.config:
                        pump_on_duration = \
                            self.config['pump_heater_length_time_on']
                    dt = (newtime - self.time_last_transition)
                    if dt > pump_on_duration:
                        self.to_pump_hot_cooling_mainplate()
                    return  # skip the rest of this case
            if 'SWITCH' in temperatures:
                if temperatures['SWITCH'] > self.config['temps']['Tswitch']:
                    #  heat switch is turning on, but pump is not hot enough
                    self.to_pump_warming_cooling_mainplate()
                elif temperatures['PUMP'] > self.config['temps']['Tpump']:
                    self.to_pump_hot_cooling_mainplate()
            elif temperatures['PUMP'] > self.config['temps']['Tpump']:
                    self.to_pump_hot_cooling_mainplate()
        elif self.is_pump_warming_cooling_mainplate():
            #  Check if heat switch has cooled off, can start to heat the pump
            if 'SWITCH' in temperatures:
                if temperatures['SWITCH'] < self.config['temps']['Tswitch']:
                    self.to_pump_warming()
        elif self.is_pump_hot_cooling_mainplate():
            # if temperatures['main_plate'] < self.config['temps']['T4Ksetpt']:
            # print('pump_hot_cooling_mainplate', temperatures['4K'],
            #       self.config['temps']['T4Ksetpt'])
            #  Check that the pump is still hot (within 5K of set pt)
            if 'PUMP' in temperatures:
                if temperatures['PUMP'] < (self.config['temps']['Tpump']-5):
                    self.console.writeline('need to turn on pump heater')
                    self.to_pump_warming()
            #  Check 4K plate and check to makes sure 1K pot is cold
            if temperatures['4K'] < self.config['temps']['T4Ksetpt']:
                if temperatures['1K'] < self.config['temps']['T4Ksetpt']+0.25:
                    self.to_cooling_pump()
                else:
                    target = self.config['temps']['T4Ksetpt']+0.25
                    self.console.writeline('1K plate is not at %.2f.' % target)
        elif self.is_cooling_pump():
            if temperatures['1K'] < self.config['temps']['T1Khigh']:
                self.to_cold()
        elif self.is_warming_up():
            if temperatures['4K'] > 270:
                self.to_system_warm()

    def datastr_to_dict(self, datastr):
        temperatures = {}
        fridge = self.fridge
        data_string_list = datastr.split(',')
        #  Check if second element is a float or not... it could be a human
        #  readable date
        try:
            float(data_string_list[1])
        except ValueError:
            del data_string_list[1]
        data_float_list = [float(value) for value in data_string_list]
        switch_list = ['pump_heater', 'switch_heater', 'compressor']
        # for name, value in zip(fridge.sensor_names, data_string_list[1:]):
        for name, value in zip(fridge.sensor_names, data_float_list[1:]):
            try:
                # temperatures[name.upper()] = float(value)
                temperatures[name.upper()] = value
                # add elements to the dictionary for LTS PTC controller
                if name.upper() == 'SW':
                    temperatures['SWITCH'] = temperatures['SW']
                if name.upper() == 'P':
                    temperatures['PUMP'] = temperatures['P']
                if name.upper() == 'MP':
                    temperatures['4K'] = temperatures['MP']
            except ValueError:
                print('ValueError: %r, %r' % (name, value))
                temperatures[name.upper()] = -1
            except Exception as e:
                print('Exception/Error', e)
            # except Error as e:
                print('Error', e)
                print("Unexpected error:", sys.exc_info()[0])
                print('name', name.upper())
                print('temperatures', value)
                print('Error in conversion: %r %r' % (name, value))
                temperatures[name.upper()] = -1
            if name in switch_list:
                temperatures[name.upper()] = temperatures[name.upper()] > 0
        # print(temperatures)
        return temperatures
