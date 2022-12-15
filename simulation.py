'''
simulation.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Last updated August 2020
'''

import json
import sys
import os
import heapq
import pandas
import random
import pickle
import datetime
import IPython
import numpy
import time
import bqplot
import ipywidgets
import operator
import data_handler
import environment
import agent
import number_of_choices_per_parameter
import traceback
from copy import deepcopy
sys.path.insert(0, 'nn/')
#from NN_utilities import load_model_sim


class Simulation():
    ''' The simulation class allows you to run the simulation.

    Example:
        sim = simulation.Simulation(path_to_parameter_file)
        sim.repeat_simulation()

    Args:
        path_to_parameter_file (str): The path to the input file
            containing input parameters for the simulation.

    Kwargs:
        measures (List[str]): A list of values to measure (usually for
            experiments). Possible measures are number_of_centers,
            number_of_charging_stations and walking_preparedness. #NOTE: update this list
        overwrite_parameters (Dict[str, Any]): Parameter values to use instead
            of the values specified in the input file. Often used for
            experiments. Parameters that can be overwritten can be found in the
            readme.

    Attributes:
        data_handler (DataHandler): Instance of the DataHandler class.
        environment (Environment): Instance of the Environment class.
        agents (Dict[str, Agent]): Agent ID's are used as keys with Agent
            instances as values.
        activity_heap (List[Tuple[DateTime, Agent]]): A sorted list containing
            a tuple of each agent and the time of their next activity.
        stop_condition (str): The name of the stop condition, either 'time' or
            'nr_activities_executed_per_agent'.
        stop_condition_parameters (Dict[str, Any]): The parameters needed for
            the specified stop condition. More information can be found in the
            documentation of the stop method.
        start_date_simulation (DateTime): The start date of the simulation.
        current_time (DateTime): The current time of the simulation.
        warmup_period (TimeDelta): The duration of the warmup period.
        agent_creation_method (str): The method used to create agents. Options
            are 'random', 'given' or 'previous'.
        agent_initialization (str): This attribute determines whether to load
            agents from memory or to create new ones and store them to memory.
            Options are 'load_and_use', 'create, 'create_and_store' and
            'create_and_use'.
        filepath_agent_database (str): Path to the agent database.
        info_printer (bool): Parameter to decide whether to print information
            about run times.
        previously_simulated (bool): Variable to indicate if this simulation
            has run before.
        progress_simulation (float): Fraction of the time of the simulation that
            is currently done.
        simulation_total_duration (TimeDelta): Duration of the simulation.
        measures (List[str]): A list of values to measure (usually for
            experiments). Possible measures are number_of_centers,
            number_of_charging_stations and walking_preparedness. #NOTE: update this list
        sensors (Dict[str, List]): A dictionary containing measures as keys and
            as values a list of the measure values (each value indicating a run).
            An additional key is:
            nr_activities_executed_per_agent (Dict[str, int]): The number of
                activities executed so far per agent.
        IDs_from_agent_database (bool): If this is put to True and agents are being
            created (agent_initialization is on create, create_and_use or
            create_and_store), then agent IDs are being loaded from a memory
            file 'data/experiment_results/all_agents.pkl' (if available).
    '''

    def __init__(self, path_to_parameter_file, measures = [],
        overwrite_parameters = {}):

        start = time.process_time()

        try:
            parameters = self._load_parameter_file(path_to_parameter_file)
        except Exception as e:
            print(e)
            sys.exit()
        if not self._load_and_check_attribute_parameters(parameters,
            overwrite_parameters):
            sys.exit()

        self._save_parameter_logfile(parameters, overwrite_parameters)

        if self.info_printer:
            print('INFO: Initialization')

        self.data_handler = data_handler.DataHandler(parameters['data_handler'],
            overwrite_parameters, self.info_printer)

        self._load_environment(parameters, overwrite_parameters)

        self.measures = measures
        self.sensors = {}
        for measure in measures:
            self.sensors[measure] = []
        self.sensors['nr_activities_executed_per_agent'] = []
        self.sensors['nr_activities_executed_per_cp'] = []

        if self.rollout_strategy_to_use != 'none' and \
            numpy.sum(list(self.number_of_CPs_to_add.values())) > 0:
            self._add_CPs()

        
        self._create_agents(parameters, overwrite_parameters)

        if self.agent_initialization in ['create', 'create_and_store']:
            if self.info_printer:
                print('\tINFO: Initialization took %s' %
                    self.data_handler.get_time_string(time.process_time() - start))
            return

        self._create_activity_heap()
        self.previously_simulated = False

        if self.info_printer:
            print('\tINFO: Initialization took %s' %
                self.data_handler.get_time_string(time.process_time() - start))

    def _save_parameter_logfile(self, parameters, overwrite_parameters):
        ''' Saves a copy of the parameters file with the overwrite parameters
            added to the log folder with a time stamp so that the experiment
            can be repeated.

        Args:
            parameters (Dict[str, Any]): The parameters of the input file.
            overwrite_parameters (Dict[str, Any]): The overwrite parameters.
                Parameters that can be overwritten are:
                info_printer, number_of_agents, bin_size_dist,
                clustering_lon_shift, clustering_lon_lat_scale,
                clustering_birch_threshold, minimum_nr_sessions_cs,
                minimum_nr_sessions_center and threshold_fraction_sessions.
        '''
        overwrite_parameters_list = ['info_printer', 'number_of_agents',
                    'bin_size_dist','clustering_lon_shift',
                    'clustering_lon_lat_scale','clustering_birch_threshold',
                    'minimum_nr_sessions_cs','minimum_nr_sessions_center',
                    'threshold_fraction_sessions']

        time_now = str(datetime.datetime.now().date()) + \
                                      '_' + str(datetime.datetime.now().time())[:-7]
        time_now = time_now.replace(':','_')
        time_now = time_now.replace('-','_')
        path_to_logfile = 'data/logs/simulation/{}_parameters.json'.format(time_now)

        for param in overwrite_parameters:
            print(param)
            if param in overwrite_parameters:
                parameters['{}'.format(param)] = \
                                        overwrite_parameters['{}'.format(param)]

        with open(path_to_logfile, 'w') as outfile:
            json.dump(parameters, outfile)

    def _load_parameter_file(self, path_to_file):
        ''' This method loads a json file containing the parameters
            used in the model.

        Args:
            path_to_file (str): Path to the json file.

        Returns:
            parameters (Dict[str, Any]): The content of the input file stored
                as a dictionary.
        '''

        try:
            parameters = json.load(open(path_to_file))
        except Exception as e:
            raise Exception('ERROR: Could not load parameter file (%s): %s' %
                (path_to_file, e))
        return parameters

    def _load_and_check_attribute_parameters(self, parameters, overwrite_parameters):
        ''' This method loads the parameters into the attributes of the class.
            Furthermore it checks whether the general and simulation contain
            valid values.

        Args:
            parameters (Dict[str, Any]): The parameters of the input file.
            overwrite_parameters (Dict[str, Any]): Parameter values to use
                instead of the values specified in the input file. Often used
                for experiments. Parameters that can be overwritten are:
                info_printer, number_of_agents, bin_size_dist,
                clustering_lon_shift, clustering_lon_lat_scale,
                clustering_birch_threshold, minimum_nr_sessions_cs,
                minimum_nr_sessions_center and threshold_fraction_sessions.

        Updates:
            info_printer
            start_date_simulation
            stop_condition
            stop_condition_parameters
            warmup_period
            agent_creation_method
            agent_initialization

        Returns:
            (bool): Returns True if all parameters are valid.
        '''

        if 'info_printer' in overwrite_parameters:
            self.info_printer = overwrite_parameters['info_printer']
        else:
            self.info_printer = parameters['info_printer']
        if not isinstance(self.info_printer, bool):
            print('WARNING: Parameter info_printer (%s) ' % self.info_printer +
                'is not a boolean. Time will not be printed.')
            self.info_printer = False
        
        
        if 'start_date_simulation' in overwrite_parameters:
            self.start_date_simulation = pandas.to_datetime(overwrite_parameters['start_date_simulation'],
                format = '%d-%m-%Y', errors = 'coerce')         
        else:
            self.start_date_simulation = pandas.to_datetime(parameters['simulation']['start_date_simulation'],
                format = '%d-%m-%Y', errors = 'coerce')
        if not isinstance(self.start_date_simulation, pandas.Timestamp):
            print('ERROR: start_date_simulation (%s) is not a DateTime.' % self.start_date_simulation)
            return False
        start_date_test_data = pandas.to_datetime(
            parameters['data_handler']['start_date_test_data'],
            format = '%d-%m-%Y', errors = 'coerce')
        if not isinstance(start_date_test_data, pandas.Timestamp):
            print('ERROR: start_date_test_data (%s) is not a DateTime.'
                % start_date_test_data)
            return False
        if start_date_test_data != self.start_date_simulation:
            print('WARNING: start_date_simulation (%s) ' % self.start_date_simulation +
            'is not the same as start_date_test_data ' +
            '(%s).' %start_date_test_data)

        if 'warmup_period_in_days' in overwrite_parameters:
            warmup_period_in_days = overwrite_parameters['warmup_period_in_days']
        else:
            warmup_period_in_days = parameters['simulation']['warmup_period_in_days']
        if not isinstance(warmup_period_in_days, int):
            print('ERROR: warmup_period_in_days (%s) should be an int.' %
                  warmup_period_in_days)
            return False
        if warmup_period_in_days < 0:
            print('ERROR: warmup_period_in_days (%s) should non negative.' %
                  warmup_period_in_days)
            return False
        self.warmup_period = pandas.to_timedelta('%d days'
            % warmup_period_in_days)
        
        
        ##### Need to update this even more, so it can handle more exceptions in overwrite parameters
        if 'stop_condition_parameters' in overwrite_parameters:
            self.stop_condition_parameters = overwrite_parameters['stop_condition_parameters'].copy()
            self.stop_condition_parameters['max_time'] = \
                        pandas.to_datetime(self.stop_condition_parameters['max_time'],
                            format = '%d-%m-%Y', errors = 'coerce')
            self.stop_condition = parameters['simulation']['stop_condition']
            
        else:
            if not isinstance(parameters['simulation']['stop_condition'], str) or \
                    (parameters['simulation']['stop_condition'] != 'time' and \
                    parameters['simulation']['stop_condition'] != 'nr_activities_executed_per_agent'):
                print('ERROR: Invalid stop_condition' +
                    '(%s). ' % parameters['simulation']['stop_condition'] +
                    'Options are \'time\' and \'nr_activities_executed_per_agent\'.')
                return False
            self.stop_condition = parameters['simulation']['stop_condition']
            self.stop_condition_parameters = dict()

            if self.stop_condition == 'time':
                if not 'max_time' in parameters['simulation']['stop_condition_parameters'] or \
                   not isinstance(pandas.to_datetime(parameters['simulation']['stop_condition_parameters']['max_time'],
                        format = '%d-%m-%Y', errors = 'coerce'), pandas.Timestamp):
                    print('ERROR: Invalid stop condition parameter (%s) for stop condition \'time\'.' %
                        parameters['simulation']['stop_condition_parameters'])
                    return False
                else:
                    self.stop_condition_parameters['max_time'] = \
                        pandas.to_datetime(parameters['simulation']['stop_condition_parameters']['max_time'],
                            format = '%d-%m-%Y', errors = 'coerce')
                    if self.stop_condition_parameters['max_time'] < self.start_date_simulation:
                        print('ERROR: max_time (%s) is before the start time of the simulation (%s)' %
                            (self.stop_condition_parameters['max_time'], self.start_date_simulation))
                        return False
                    if self.start_date_simulation + self.warmup_period >= self.stop_condition_parameters['max_time']:
                        print('ERROR: start_date_simulation (%s) + warmup_period (%s) is greater or equal to end_time (%s)' %
                              (self.start_date_simulation, self.warmup_period, self.stop_condition_parameters['max_time']))
                        return False

            if self.stop_condition == 'nr_activities_executed_per_agent':
                if not 'min_nr_activities_executed_per_agent' in parameters['simulation']['stop_condition_parameters'] or \
                   not isinstance(parameters['simulation']['stop_condition_parameters']['min_nr_activities_executed_per_agent'], int) or \
                   parameters['simulation']['stop_condition_parameters']['min_nr_activities_executed_per_agent'] <= 0:
                    print('ERROR: Invalid stop condition parameter (%s) ' %
                        parameters['simulation']['stop_condition_parameters'] +
                        'for stop condition \'nr_activities_executed_per_agent\'. ' +
                        'The parameter \'min_nr_activities_executed_per_agent\' ' +
                        'is needed with an positiveinteger value.')
                    return False
                else:
                    self.stop_condition_parameters['min_nr_activities_executed_per_agent'] = \
                        int(parameters['simulation']['stop_condition_parameters']['min_nr_activities_executed_per_agent'])

        if 'agent_creation_method' in overwrite_parameters:
            self.agent_creation_method = overwrite_parameters['agent_creation_method']
        else:
            self.agent_creation_method = parameters['simulation']['agent_creation_method']
        if not isinstance(self.agent_creation_method, str):
            print('ERROR: agent_creation_method (%s) is not a string.' %
                self.agent_creation_method)
            return False
        if not self.agent_creation_method in ['random', 'given', 'previous']:
            print('ERROR: agent_creation_method (%s) is not \'random\', \'given\' or \'previous\'.' %
                self.agent_creation_method)
            return False

        if 'agent_initialization' in overwrite_parameters:
            self.agent_initialization = overwrite_parameters['agent_initialization']
        else:
            self.agent_initialization = parameters['simulation']['agent_initialization']
        if not isinstance(self.agent_initialization, str):
            print('ERROR: agent_initialization (%s) is not a string.' %
                self.agent_initialization)
            return False
        if not self.agent_initialization in ['create', 'load_and_use', 'create_and_store', 'create_and_use']:
            print('ERROR: agent_initialization (%s) is not \'create\', \'load_and_use\', \'create_and_store\' or \'create_and_use\'.' %
                self.agent_initialization)
            return False

        if 'filepath_agent_database' in overwrite_parameters:
            self.filepath_agent_database = overwrite_parameters['filepath_agent_database']
        else:
            self.filepath_agent_database = parameters['simulation']['filepath_agent_database']
        if self.agent_initialization in ['create_and_store', 'load_and_use']:
            if not isinstance(self.filepath_agent_database, str):
                print('ERROR: filepath_agent_database (%s) is not a string.' %
                    self.filepath_agent_database)
                return False
            if not (os.path.isdir(self.filepath_agent_database) or os.path.isfile(self.filepath_agent_database)):
                print('ERROR: filepath_agent_database (%s) does not exist.' %
                    self.filepath_agent_database)
                return False

        if 'number_of_agents' in overwrite_parameters:
            self.number_of_agents = overwrite_parameters['number_of_agents']
        else:
            self.number_of_agents = parameters['simulation']['number_of_agents']
        if not isinstance(self.number_of_agents, int):
            print('ERROR: parameters[\'number_of_agents\'] (%s) is not an int.' %
                self.number_of_agents)
            return False
        if self.number_of_agents < 0:
            print('ERROR: parameters[\'number_of_agents\'] (%s) is not positive.' %
                self.number_of_agents)
            return False

        if 'agent_IDs' in overwrite_parameters:
            print('INFO: agent_IDs from overwrite_parameters')
            self.agent_IDs = overwrite_parameters['agent_IDs']
        else:
            self.agent_IDs = parameters['simulation']['agent_IDs']
            print('INFO: agent_IDs from parameter file')
        if not isinstance(self.agent_IDs, list):
            print('ERROR: agent_IDs (%s) are not a list.' % self.agent_IDs)
            return False

        if 'non_habitual_agents' in overwrite_parameters:
            self.non_habitual_agents = overwrite_parameters['non_habitual_agents']
        else:
            self.non_habitual_agents = parameters['simulation']['non_habitual_agents']
        if not isinstance(self.non_habitual_agents, dict):
            print('ERROR: non_habitual_agents (%s) is not a dictionary.')
            return False

        if self.non_habitual_agents.keys() != set(['Amsterdam', 'Den Haag', 'Rotterdam', 'Utrecht']):
            print('ERROR: non_habitual_agents (%s) should have keys %s' %
                (self.non_habitual_agents.keys(), set(['Amsterdam', 'Den Haag', 'Rotterdam', 'Utrecht'])))
            return False
        for key in self.non_habitual_agents:
            if not isinstance(self.non_habitual_agents[key], float) and \
                not isinstance(self.non_habitual_agents[key], int):
                print('ERROR: non_habitual_agents[%s] should have an int or float value.' %
                    (key, self.non_habitual_agents[key]))
                return False
            if self.non_habitual_agents[key] < 0:
                print('ERROR: non_habitual_agents[%s] should have value greather than 0 (%.2f).' %
                    (key, self.non_habitual_agents[key]))
                return False
        if 'non_habitual_agents_as_ratio_or_number' in overwrite_parameters:
            self.non_habitual_agents_as_ratio_or_number = overwrite_parameters['non_habitual_agents_as_ratio_or_number']
        else:
            self.non_habitual_agents_as_ratio_or_number = parameters['simulation']['non_habitual_agents_as_ratio_or_number']
        if not isinstance(self.non_habitual_agents_as_ratio_or_number, str):
            print('ERROR: non_habitual_agents_as_ratio_or_number (%s) is not a string.')
            return False

        if 'IDs_from_agent_database' in overwrite_parameters:
            self.IDs_from_agent_database = overwrite_parameters['IDs_from_agent_database']
            parameters['simulation']['IDs_from_agent_database'] = self.IDs_from_agent_database
        else:
            self.IDs_from_agent_database = parameters['simulation']['IDs_from_agent_database']
        if not isinstance(self.IDs_from_agent_database, bool):
            print('ERROR: IDs_from_agent_database (%s) is not a boolean.' %
                  self.IDs_from_agent_database)
            return False
        
        
        if 'delete_centers_over_time' in overwrite_parameters:
            self.delete_centers_over_time = overwrite_parameters['delete_centers_over_time']
            parameters['simulation']['delete_centers_over_time'] = self.delete_centers_over_time
        else:
            self.delete_centers_over_time = parameters['simulation']['delete_centers_over_time']
        if not isinstance(self.delete_centers_over_time, int):
            print('ERROR: delete_centers_over_time (%s) should have an int value.' %
                  self.delete_centers_over_time)
            return False
        
        if 'add_agents_during_simulation' in overwrite_parameters:
            self.add_agents_during_simulation = overwrite_parameters['add_agents_during_simulation']
            parameters['simulation']['add_agents_during_simulation'] = self.add_agents_during_simulation
        else:
            self.add_agents_during_simulation = parameters['simulation']['add_agents_during_simulation']
        if not isinstance(self.add_agents_during_simulation, int):
            print('ERROR: add_agents_during_simulation (%s) should have an int value.' %
                  self.add_agents_during_simulation)
            return False
        

        

        if not isinstance(parameters['simulation']['environment_from_memory'], bool):
            print('ERROR: environment_from_memory (%s) is not a boolean.' %
                  parameters['simulation']['environment_from_memory'])
            return False
        
        if 'reset_environment_sockets' in overwrite_parameters:
            self.reset_environment_sockets = overwrite_parameters['reset_environment_sockets']
#             parameters['simulation']['reset_environment_sockets'] = self.reset_environment_sockets
        else:
            self.reset_environment_sockets = parameters['simulation']['reset_environment_sockets']
        if not isinstance(self.reset_environment_sockets, bool):
            print('ERROR: reset_environment_sockets (%s) is not a boolean.' %
                  self.reset_environment_sockets)
            return False
        
        if 'reset_environment' in overwrite_parameters:
            self.reset_environment = overwrite_parameters['reset_environment']
            parameters['simulation']['reset_environment'] = self.reset_environment
        else:
            self.reset_environment = parameters['simulation']['reset_environment']
        if not isinstance(self.reset_environment, bool):
            print('ERROR: reset_environment (%s) is not a boolean.' %
                  self.reset_environment)
            return False
        
        if 'city' in overwrite_parameters:
            self.city_to_simulate = overwrite_parameters['city']
            parameters['simulation']['city'] = overwrite_parameters['city']
        else:
            self.city_to_simulate = parameters['simulation']['city']
        if not isinstance(self.city_to_simulate, str):
            print('ERROR: city_to_simulate (%s) is not a string.' %
                str(self.city_to_simulate))
            return False
        if self.city_to_simulate == "all":
            self.city_to_simulate = ["Amsterdam", "The Hague", "Rotterdam", "Utrecht"]

        if 'rollout_strategy_to_use' in overwrite_parameters:
            self.rollout_strategy_to_use = overwrite_parameters['rollout_strategy_to_use']
        else:
            self.rollout_strategy_to_use = parameters['simulation']['rollout_strategy_to_use']

        if self.rollout_strategy_to_use != 'none':
            if not self.rollout_strategy_to_use in ['most_unique_users_per_week',
                'most_kwh_charged_per_week', 'most_failed_connection_attempts', 'random']:
                print('ERROR: rollout_strategy_to_use (%s) is not in %s.' %
                      (self.rollout_strategy_to_use, ['most_unique_users_per_week',
                        'most_kwh_charged_per_week', 'most_failed_connection_attempts']))
                return False
            if not os.path.exists('data/simulation_pkls/%s_data_per_city_per_cp.pkl' % self.rollout_strategy_to_use):
                print('ERROR: %s does not exist. This is needed for the rollout strategy.' %
                    'data/simulation_pkls/%s_data_per_city_per_cp.pkl' % self.rollout_strategy_to_use)
                return False

        if self.rollout_strategy_to_use != 'none':
            if 'number_of_CPs_to_add' in overwrite_parameters:
                self.number_of_CPs_to_add = overwrite_parameters['number_of_CPs_to_add']
            else:
                self.number_of_CPs_to_add = parameters['simulation']['number_of_CPs_to_add']
            if not isinstance(self.number_of_CPs_to_add, dict):
                print('ERROR: number_of_CPs_to_add (%s) is not a dict.' %
                      self.number_of_CPs_to_add)
                return False
            if self.number_of_CPs_to_add.keys() != set(['Amsterdam', 'Den Haag', 'Rotterdam', 'Utrecht']):
                print('ERROR: number_of_CPs_to_add (%s) is not a dict.' %
                      self.number_of_CPs_to_add)
                return False
            for key in self.number_of_CPs_to_add:
                if not isinstance(self.number_of_CPs_to_add[key], int):
                    print('ERROR: number_of_CPs_to_add[%s] should have an int value.' %
                        (key, self.number_of_CPs_to_add[key]))
                    return False

        return True

    def _load_environment(self, parameters, overwrite_parameters):
        ''' This method loads the environment and sets it as the environment
            attribute.

        Args:
            parameters (Dict[str, Any]): The parameters of the input file.

        Updates:
            environment
        '''
        
        # check if we want to load or create the environment
        load_environment = True
        if ('environment_from_memory' in overwrite_parameters and
            not overwrite_parameters['environment_from_memory']):
                load_environment = False
        elif not parameters['simulation']['environment_from_memory']:
                load_environment = False
        
        if 'filepath_environment_file' in overwrite_parameters:
            self.filepath_environment_file = overwrite_parameters['filepath_environment_file']
        else:
            self.filepath_environment_file = parameters['simulation']['filepath_environment_file']
            
        if load_environment and self.agent_initialization in ['create_and_store', 'load_and_use']:
            if not isinstance(self.filepath_environment_file, str):
                print('ERROR: filepath_environment_file (%s) is not a string.' %
                    self.filepath_environment_file)
                sys.exit()
            if not (os.path.isdir(self.filepath_environment_file) or os.path.isfile(self.filepath_environment_file)):
                print('ERROR: filepath_environment_file (%s) does not exist.' %
                    self.filepath_environment_file)
                sys.exit()
        
        if load_environment:
            try:
                with open(self.filepath_environment_file, 'rb') as environment_file:
                    self.environment = pickle.load(environment_file)
                    if len(self.environment.css_info) > 0 and self.reset_environment_sockets:
                        print("\tINFO: Started to reset the amount of sockets of the charge points")
                        previous_time = time.process_time()
                        self.environment.reset_environment_sockets(self.start_date_simulation, \
                                                                   self.data_handler.get_sessions('general'))
                        print('\tINFO: Environment is reset in %s' % ( \
                   self.data_handler.get_time_string(time.process_time() - previous_time)))
                    elif not self.reset_environment_sockets:
                        print("\tINFO: reset_environment_sockets parameter is false, so the environment will not be reset")
                    else:
                        print("\tINFO: Environment is empty, so the environment will not be reset")
            except Exception as e:
                raise e
                print('WARNING: Environment cannot be loaded from memory, ' +
                    'creating environment.')
                self.environment = environment.Environment(
                    self.data_handler.get_sessions('general'), self.info_printer, self.start_date_simulation)
                with open ('data/simulation_pkls/environment.pkl', 'wb') as environment_file:
                    pickle.dump(self.environment, environment_file)
                print('INFO: Environment stored in %s' % ("data/simulation_pkls/environment.pkl"))
                
                
        ## if we want to create the environment      
        else:
            print('INFO: Environment started to be created at %s.' % datetime.datetime.now())
            self.environment = environment.Environment(
                self.data_handler.get_sessions('general'), self.info_printer, self.start_date_simulation)
            
            if 'filepath_environment_file' in overwrite_parameters:
                print('INFO: Environment will be stored in %s.' % self.filepath_environment_file)
            else:
                print('INFO: Environment filepath not defined in overwrite parameters. \n\t \
                Thus will be stored in %s.' % self.filepath_environment_file)
            with open(self.filepath_environment_file, 'wb') as environment_file:
                pickle.dump(self.environment, environment_file)
            print('INFO: Environment was created in %s.' % datetime.datetime.now())
        if 'distance_metric' in overwrite_parameters:
            self.environment.distance_metric = overwrite_parameters['distance_metric']
        else:
            self.environment.distance_metric = parameters['simulation']['distance_metric']
        
        self._set_parking_zone_data()
        
        
    def _set_parking_zone_data(self):
        geojson_amsterdam_parkeertarief = "data/parkingzones/amsterdam_parkeertariefgebied.geojson"
        geojson_amsterdam_vergunning = "data/parkingzones/amsterdam_vergunningsgebied.geojson"

        with open(geojson_amsterdam_vergunning) as f:
            self.environment.amsterdam_vergunning_data = json.load(f)

        with open(geojson_amsterdam_parkeertarief) as f:
            self.environment.amsterdam_parkeertarief_data = json.load(f)

        self.environment.parking_zones_amsterdam, self.environment.parking_zones_den_haag = \
            number_of_choices_per_parameter.get_zone_information()
        to_remove = []
        for i, zone in enumerate(self.environment.parking_zones_den_haag['features']):
            if zone['properties']['CODE'][-4:] != "(PT)":
                to_remove.append(zone)
            else:
                self.environment.parking_zones_den_haag['features'][i]['properties']['CODE'] = \
                    self.environment.parking_zones_den_haag['features'][i]['properties']['CODE'][:-5]
        for zone in to_remove:
            self.environment.parking_zones_den_haag['features'].remove(zone)

        self.environment.all_zones_den_haag = \
            number_of_choices_per_parameter.get_all_zones_den_haag(self.environment.parking_zones_den_haag)

        path_to_file = 'data/extra/Tarieven-Amsterdam-per-1-12-2016-v2.csv'
        with open(path_to_file) as f:
            pz_prices_amsterdam = pandas.read_csv(f, engine='python')

        day_conversion = {'MAANDAG': 0, 'DINSDAG': 1, 'WOENSDAG': 2, 'DONDERDAG': 3, 'VRIJDAG': 4, 'ZATERDAG': 5, 'ZONDAG': 6}
        self.environment.all_zones_amsterdam = {}
        for row_nr, row in pz_prices_amsterdam.iterrows():
            if row['EtmaalNaam'] == 'VRIJPARK':
                continue
            if row['GebiedCode'] not in self.environment.all_zones_amsterdam:
                self.environment.all_zones_amsterdam[row['GebiedCode']] = {day: [] for day in range(7)}
            try:
                self.environment.all_zones_amsterdam[row['GebiedCode']][day_conversion[row['EtmaalNaam']]].append({
                'start_time': int(row['Begintijd'].split(':')[0]),
                'end_time': int(row['Eindtijd'].split(':')[0]),
                'price': float(row[' Bedrag '])})
            except:
                pass

    def _add_CPs(self):
        with open('data/simulation_pkls/%s_data_per_city_per_cp.pkl' % self.rollout_strategy_to_use, 'rb') as data_file:
            data_per_cp = pickle.load(data_file)

        for city in self.number_of_CPs_to_add:
            cps = [cp for cp, count in data_per_cp[city].items() if self.environment.css_info[cp]['placement_date'] < self.start_date_simulation]
            counts = [count for cp, count in data_per_cp[city].items() if self.environment.css_info[cp]['placement_date'] < self.start_date_simulation]
            counts_norm = counts / numpy.sum(counts)

            cps_to_duplicate = numpy.random.choice(cps, size = self.number_of_CPs_to_add[city], replace = True, p = counts_norm)

            for cp in cps_to_duplicate:
                # print(self.environment.css_occupation[cp])
                self.environment.css_occupation[cp] = self.environment.css_occupation[cp]  + [None, None]
                # print(self.environment.css_occupation[cp])

                # print(self.environment.css_info[cp])
                self.environment.css_info[cp]['amount_of_sockets'] += 2
                # print(self.environment.css_info[cp])
            if self.info_printer:
                print('\tINFO: Added %d CPs in %s using rollout strategy %s' %
                    (self.number_of_CPs_to_add[city], city, self.rollout_strategy_to_use))



    def _create_agents(self, parameters, overwrite_parameters):
        ''' This method creates a dictionary of Agents and stores this in the
            agents attribute.

        Args:
            parameters (Dict[str, Any]): Parameters of the input file.

        Updates:
            agents
        '''

        self.agents = {}
        
        if self.number_of_agents != 0:
            if self.agent_creation_method == 'random':
                self._initialize_random_agents(parameters, overwrite_parameters)
            else:
                self._initialize_planned_agents(parameters, overwrite_parameters)

        self._initialize_non_habitual_agents(parameters, overwrite_parameters)

        if self.agent_initialization == 'create_and_store':
            print('\tINFO: Agents have been stored.')

        if self.agent_initialization in \
            ['create_and_use', 'store_and_use', 'load_and_use'] and not self.agents:
            print('ERROR: No valid agents for simulation.')

    def _initialize_planned_agents(self, parameters, overwrite_parameters):
        ''' This method creates a dictionary of Agents and stores this in the
            agents attribute. In this method we assume that the
            agent_creation_method option is either on 'given' or 'previous'.

        Args:
            parameters (Dict[str, Any]): Parameters of the input file.

        Updates:
            agents
        '''

        if self.agent_creation_method == 'given':
            parameters['simulation']['agent_creation_method'] = self.agent_creation_method
            if self.filepath_agent_database[-4:] == '.pkl':
                previous_time = time.process_time()
                print('\t\tINFO: Started loading pickle file with agents of file %s' % (self.filepath_agent_database))
                with open(self.filepath_agent_database, 'rb') as open_agent_file:
                    all_agents = pickle.load(open_agent_file)   
                print('\t\tINFO: Loaded %d agents in %s' % (len(all_agents), \
               self.data_handler.get_time_string(time.process_time() - previous_time)))
            else:
                raise Exception("ERROR: No valid input for filepath_agent_database in parameters: " + 
                                "agent_creation_method is given, so filepath_agent_database should be a pickle file")

            agent_IDs = list(all_agents.keys())
            
        if self.agent_creation_method == 'previous':
            with open('data/simulation_pkls/agents_of_latest_simulation.pkl', 'rb') as open_agent_file:
                agent_IDs = pickle.load(open_agent_file)
    
        if self.info_printer:
            print('\tINFO: Initializing (%s) %d %s agents' %
                (self.agent_initialization,
                len(agent_IDs), self.agent_creation_method))

        previous_time = time.process_time()
        for i, agent_ID in enumerate(agent_IDs):
            if 'car2go' in agent_ID:
                continue

            if (i + 1) % 10 == 0 and i != 0:
                if self.info_printer:
                    print('\tINFO: Initializing agent %d of %d' % (i + 1, len(agent_IDs)))
                    print('\tINFO: Initializing 10 agents in %s' %
                        self.data_handler.get_time_string(time.process_time() - previous_time ))
                previous_time = time.process_time()
            try:
                self.agents[agent_ID] = agent.Agent({'environment': self.environment,
                    'data_handler': self.data_handler, 'parameters': parameters['agent'],
                    'warmup_period': self.warmup_period, 'start_date_simulation': self.start_date_simulation,
                    'agent_initialization': self.agent_initialization,
                    'simulation_parameters' : parameters['simulation']},
                    self.info_printer, agent_ID, overwrite_parameters,
                    self.filepath_agent_database, measures = self.measures,
                    sim_sensors = self.sensors
                    # cluster_model_1 = self.cluster_model_1,
                    # cluster_model_2 = self.cluster_model_2,
                    # cluster_model_3 = self.cluster_model_3,
                    # 'cluster_model_4' : self.cluster_model_4},
                    # cluster_model_5 = self.cluster_model_5,
                    )
            except Exception as e:
                if str(e)[:7] != 'WARNING':
                    raise e
                
        with open ('data/simulation_pkls/agents_of_latest_simulation.pkl', 'wb') as agents_file:
            pickle.dump(list(self.agents.keys()), agents_file)

    def _initialize_random_agents(self, parameters, overwrite_parameters):
        ''' This method creates a dictionary of Agents and stores this in the
            agents attribute. These agents are loaded from the file database if
            agent_initialization of the input file is on load_and_use and
            otherwise are created. In this method we assume that the
            agent_creation_method option is on 'random'. Note that only agents
            in the file database can be loaded.

        Args:
            parameters (Dict[str, Any]): Parameters of the input file.

        Updates:
            agents
        '''

        if self.info_printer:
            print('\tINFO: Initializing (%s) %d random agents' %
                (self.agent_initialization,
                self.number_of_agents))

        current_number_of_agents = 0
        
        ### CHECK FOR CREATE AND USE
        if self.agent_initialization in ['create_and_store', 'create',
            'create_and_use']:
#             if self.IDs_from_agent_database: #NOTE: start/end data can also be in overwrite parameters    
#                 agent_file = 'data/experiment_results/experiment_max_agents_alltime.pkl'
#                 with open(agent_file, 'rb') as open_agent_file:
#                     all_agents = pickle.load(open_agent_file)
#                     random.shuffle(all_agents)
#             else:
#                 #all_agents = list(self.data_handler.training_data['ID'].unique())
            agent_counts = self.data_handler.training_data.groupby(['ID']).size()
            all_agents = list(agent_counts[agent_counts > self.data_handler.minimum_nr_sessions_center].index)
            random.shuffle(all_agents)
                
        elif self.agent_initialization == 'load_and_use':
            previous_time = time.process_time()
            if self.IDs_from_agent_database:
                all_agents = [files[:-4] for files in os.listdir(self.filepath_agent_database)
                if files[0] != '.']
                random.shuffle(all_agents)
            
            else:
                if self.filepath_agent_database[-4:] == '.pkl':
                    with open(self.filepath_agent_database, 'rb') as open_agent_file:
                        all_agents = pickle.load(open_agent_file)
                        
                else:
                    raise Exception("ERROR: No valid input for filepath_agent_database in parameters: " + 
                                    "agent_initialization is load_and_use, so input should be a pickle file")
            print('\t\tINFO: Loaded %d agents in %s' % (len(all_agents), \
                   self.data_handler.get_time_string(time.process_time() - previous_time)))
        elif self.agent_initialization not in ['create_and_store', 'create', 'create_and_use']:
            raise Exception("ERROR: No valid input for agent_initialization in input file.")
        
        
        previous_time = time.process_time()
        previous_stored_time = time.process_time()
        
        
            
        if self.agent_initialization in ['create', 'create_and_store', 'create_and_use'] or \
            self.IDs_from_agent_database:
            if self.agent_initialization == 'create_and_store' and (self.filepath_agent_database[-4:] == '.pkl' or not os.path.isdir(self.filepath_agent_database) or self.filepath_agent_database[-1:] != '/'):
                raise Exception("ERROR: No valid input for filepath_agent_database in parameters: " + 
                                "agent_initialization is create_and_store, so input should NOT be a (pickle) file, " +
                                "but a folder, where all agent_IDs & Data are going to be stored seperately")
            
            
            for i, agent_ID in enumerate(all_agents):
                if 'car2go' in agent_ID:
                    continue
                try:
                    self.agents[agent_ID] = agent.Agent({'environment': self.environment,
                                                         'data_handler': self.data_handler, 
                                                         'parameters': parameters['agent'],
                                                         'warmup_period': self.warmup_period, 
                                                         'start_date_simulation': self.start_date_simulation,
                                                         'agent_initialization': self.agent_initialization,
                                                         'simulation_parameters' : parameters['simulation']},
                                                        self.info_printer, agent_ID, overwrite_parameters,
                                                        self.filepath_agent_database, measures = self.measures,
                                                        sim_sensors = self.sensors
                                                       )
                    
             
                    


                except Exception as e:
                    if str(e)[:7] != 'WARNING':
                        if str(e)[:7] != "'routes":
                            raise e
                    print(e)
    #                     print(traceback.format_exc())
                    continue
        
                 ##### for creating only agents in a certain city
#                 agent_IDs = []
#                 if self.city_to_simulate in ["Amsterdam", "Rotterdam", "Utrecht", "Den Haag"]:
#                         for center_values in list(self.agents[agent_ID].centers_info.values()):
#                             if center_values['city'] == self.city_to_simulate:
#                                 if agent_ID not in agent_IDs:
#                                     agent_IDs.append(agent_ID)
#                                     current_number_of_agents += 1
                
                current_number_of_agents += 1
    
                if current_number_of_agents % 10 == 0:
                    if self.info_printer:
                        print('\t\tINFO: %d of %d agents are initialized' % (current_number_of_agents,
                                                                             self.number_of_agents))
                        print('\t\t\tINFO: %d of %d users are processed' % (i + 1, len(all_agents)))
                        print('\t\tINFO: Initialized %d agents in %s' % (10, self.data_handler.get_time_string(time.process_time() - previous_time)))
                        previous_time = time.process_time()
                if current_number_of_agents >= self.number_of_agents:
                    break

            if current_number_of_agents < self.number_of_agents:
                print('\t\tWARNING: Requested amount of agents not possible. ' +
                      'Continuing with simulation using %d agents.' %
                      current_number_of_agents)
                    
        ### use part of the agent_method to complete the agent object
        if self.agent_initialization in ['load_and_use']:
            if not self.IDs_from_agent_database:
                agent_IDs = list(all_agents.keys())
                agent_IDs = [agent_ID for agent_ID in agent_IDs if not 'car2go' in agent_ID]
                random.shuffle(agent_IDs)

                if self.city_to_simulate in ["Amsterdam", "Rotterdam", "Utrecht", "Den Haag"]:

                    ### check if the centers of the agent have a defined city, and input this for undefined centers
                    agent_IDs_check_city = [agent_object for agent_object in list(all_agents.values()) \
                                            for center in agent_object.centers_info \
                                            if'city' not in agent_object.centers_info[center]]
                    
                    for agent_object in agent_IDs_check_city:
                        agent_object.check_center_for_city()
                    
                    try:
                        agent_IDs = [agent_object.ID for agent_object in list(all_agents.values()) \
                                     for center in agent_object.centers_css \
                                     if agent_object.centers_info[center]['city'] == self.city_to_simulate]
                        print('\t\tINFO: There are %d agents available for %s' % \
                              (len(agent_IDs), self.city_to_simulate))
                        
                    except KeyError:
                        
                        print('\t\tWARNING: Undefined city for a center')

                agent_IDs = agent_IDs[:self.number_of_agents]
                self.habitual_users = len(agent_IDs)
                self.agents = {x: all_agents[x] for x in all_agents if x in agent_IDs}
                
                
            # import deleted centers and set end date dataframe
            if self.delete_centers_over_time:
                with open ('data/simulation_pkls/minus_centers_parkeervakken.pkl', 'rb') as open_agent_file:
                     delete_centers = pickle.load(open_agent_file)
                        
                if self.delete_centers_over_time > len(delete_centers):
                    print('Number of agents to delete is greater than number of generated centers from database. \
                          Thus continuing with the maximum amount of %s agents to delete over time.' % (len(delete_centers)))
                          
                    self.delete_centers_over_time = len(delete_centers)
#                 delete_centers = delete_centers[:self.delete_centers_over_time]
                date_rng = pandas.date_range(start=self.start_date_simulation, \
                                             end=self.stop_condition_parameters['max_time'], freq='H')
                date_dataframe = pandas.DataFrame(date_rng, columns=['date'])
            
            # copy some agents to be used as new agents
            if self.add_agents_during_simulation > 0:
                self._create_extra_agents_for_during_simulation(self.add_agents_during_simulation)
            
            count_delete_centers = 0
            for i, agent_ID in enumerate(self.agents):
                self.agents[agent_ID].environment = self.environment
                self.agents[agent_ID].data_handler = self.data_handler
                
                # reset all the cps in the walking distance based on the current amount of sockets
                if self.reset_environment == True or 'added' in agent_ID:
                    
                    for center in self.agents[agent_ID].centers_css:
                        self.agents[agent_ID].centers_css[center]['distance'] = \
                            self.agents[agent_ID].environment.get_nearby_css( \
                            self.agents[agent_ID].centers_css, center, \
                            self.agents[agent_ID].data_handler.training_data, \
                            self.agents[agent_ID].walking_preparedness, \
                            city = self.agents[agent_ID].centers_info[center]['city'])
                        
                # check if it is in a center that will be deleted and assign an end time for that agent
                if self.delete_centers_over_time:
                    if center in delete_centers and count_delete_centers < self.delete_centers_over_time:
                        number_in_line = count_delete_centers
#                         print("center in delete_centers")
                        number_in_line = round(number_in_line / self.delete_centers_over_time  * len(date_dataframe))
                        self.agents[agent_ID].end_date_agent = date_dataframe.iloc[number_in_line].loc['date']
                        count_delete_centers += 1
                    else:
#                         print(agent_ID)
                        self.agents[agent_ID].end_date_agent = self.stop_condition_parameters['max_time']
                else:
                    self.agents[agent_ID].end_date_agent = self.stop_condition_parameters['max_time']
                    
#                     self.agents[agent_ID]._load_centers(args['parameters'])
                self.agents[agent_ID].warmup_period = self.warmup_period
                self.agents[agent_ID].start_date_simulation = self.start_date_simulation
                self.agents[agent_ID].city_to_simulate = self.city_to_simulate
                if not 'added' in self.agents[agent_ID].ID:
                    
                    self.agents[agent_ID].start_date_agent = self.start_date_simulation

                    
                if not self.agents[agent_ID]._load_and_check_parameters(parameters['agent'],
                                                   overwrite_parameters,
                                                   self.agent_initialization):
                    sys.exit()
                if 'car2go' in agent_ID:
                    self.agents[agent_ID]._load_sessions_car2go()
                    
                else:
                    self.agents[agent_ID]._check_skip()
                  
                
                
                
                self.agents[agent_ID]._get_disconnection_duration()
                self.agents[agent_ID]._reset_results()
                transform_to = self.agents[agent_ID]._should_transform()
                self.agents[agent_ID]._transform_agent(transform_to)
                self.agents[agent_ID].set_initial_state()
                
              

        with open('data/simulation_pkls/agents_of_latest_simulation.pkl', 'wb') as agents_file:
            pickle.dump(list(self.agents.keys()), agents_file)


    def _initialize_non_habitual_agents(self, parameters, overwrite_parameters):
        ''' This method creates a dictionary of Agents and stores this in the
            agents attribute. In this method we assume that the
            agent_creation_method option is either on 'given' or 'previous'.

        Args:
            parameters (Dict[str, Any]): Parameters of the input file.

        Updates:
            agents
        '''

        print(self.agent_initialization)
        if self.agent_initialization in ['create_and_store']:
            # standard for each city 1 non habitual user is made
            self.non_habitual_agents = {'Amsterdam': 1, 'Den Haag': 1, 'Rotterdam': 1, 'Utrecht': 1}
        else:
            overall_city_counts = {'Amsterdam': 0, 'Den Haag': 0, 'Rotterdam': 0, 'Utrecht': 0}

            for agent_ID in self.agents:
                #if 'added' not in agent_ID:
#                     if 'added' not in agent_ID and 'car2go' not in agent.ID:

                    city_counts = self.agents[agent_ID].training_sessions['city'].value_counts().to_dict()
                    most_frequent_city = sorted(city_counts.items(), key=operator.itemgetter(1), reverse = True)[0][0]
                    # if len(city_counts) > 1:
                        # print(city_counts)
                        # print(most_frequent_city)
                    overall_city_counts[most_frequent_city] += 1
                    
                    
            if self.non_habitual_agents_as_ratio_or_number == "ratio":
                for city in self.non_habitual_agents:
                    self.non_habitual_agents[city] = int(self.non_habitual_agents[city] * overall_city_counts[city])
                    
            sum_city_counts = sum([overall_city_counts[city] for city in overall_city_counts])
            if self.non_habitual_agents_as_ratio_or_number == "number" and sum_city_counts > 0:
                print("IMPORTANT INFO: Parameter non_habitual_agents_as_ratio_or_number = 'number'. However, for now we will add a total number of %d over all cities. This is done by checking the ratio of the current agents in all the cities" % (self.non_habitual_agents['Amsterdam']))
                for city in self.non_habitual_agents:
                    self.non_habitual_agents[city] = int(self.non_habitual_agents[city] * overall_city_counts[city]/ sum_city_counts)
        
                    

                

        if self.info_printer:
            for city in self.non_habitual_agents:
                print('\tINFO: Initializing %d non_habitual agents in %s' %
                    (self.non_habitual_agents[city],
                    city))
        agent_names = {'Amsterdam': 'ams', 'Den Haag': 'den', 'Rotterdam': 'rot', 'Utrecht': 'utr'}
        # 
        agent_IDs = ['car2go_%s_%d' % (agent_names[city], i) for city in self.non_habitual_agents \
            for i in range(self.non_habitual_agents[city])]

        previous_time = time.process_time()
        for i, agent_ID in enumerate(agent_IDs):
            if (i + 1) % 200 == 0 and i != 0:
                if self.info_printer:
                    print('\tINFO: Initialized %d non_habitual agents out of %d' % (i + 1, len(agent_IDs)))
                    print('\tINFO: Initialized 200 agents in %s' %
                        self.data_handler.get_time_string(time.process_time() - previous_time ))
                previous_time = time.process_time()
            try:
                self.agents[agent_ID] = agent.Agent({'environment': self.environment,
                    'data_handler': self.data_handler, 'parameters': parameters['agent'],
                    'warmup_period': self.warmup_period, 'start_date_simulation': self.start_date_simulation,
                    'agent_initialization': self.agent_initialization,
                    'simulation_parameters' : parameters['simulation']},

                    self.info_printer, agent_ID, overwrite_parameters,
                    self.filepath_agent_database, measures = self.measures,
                    sim_sensors = self.sensors
                    # cluster_model_1 = self.cluster_model_1,
                    # cluster_model_2 = self.cluster_model_2,
                    # cluster_model_3 = self.cluster_model_3,
                    # 'cluster_model_4' : self.cluster_model_4,
                    # cluster_model_5 = self.cluster_model_5,
                    )
                ## This is ugly to have here. Could be improved.
                self.agents[agent_ID].start_date_agent = self.start_date_simulation
                self.agents[agent_ID].end_date_agent = self.stop_condition_parameters['max_time']
                
                
            except Exception as e:
                print('\t\tWARNING: %s' % e)
#                 if len(str(e)) > 18 and str(e)[18] == '0':
#                     answer = input('WARNING: The parameters in the input file ' +
#                         'might be off, do you want to continue? (y/n)')
#                     while answer not in ['n', 'y']:
#                         answer = input('Please respecify your answer: \'y\' or \'n\'.')
#                     if answer == 'n':
#                         print('ERROR: User interrupted the simulation.')
#                         sys.exit()
#                 if str(e)[0] != 'W':
#                     raise e


    def _create_extra_agents_for_during_simulation(self, agents_to_add):
        print("\tINFO: Adding %s agents by copying" % (agents_to_add))
        n_agents = len(self.agents)
        ratio_added_agents = agents_to_add/n_agents
        
        # if agents need to be copied more than once (if there are more than double of extra agents) 
        for i in range(0, int(numpy.ceil(ratio_added_agents))):
            agent_IDs = list(self.agents.keys())
            random.shuffle(agent_IDs)
            if i >= 1:
                agents_to_add -= n_agents
                agent_IDs = [agent_ID for agent_ID in agent_IDs if '_added'*i in agent_ID]
                agent_IDs = agent_IDs[:agents_to_add]
            else:
                agent_IDs = [agent_ID for agent_ID in agent_IDs if not 'car2go' in agent_ID]
                agent_IDs = agent_IDs[:agents_to_add]

            # copy agent objects
            previous_time = time.process_time()
            added_agents = [self.agents[agent_ID] for agent_ID in agent_IDs]
            new_added_agents = pickle.loads(pickle.dumps(added_agents))
            print('\tINFO: Copying %s agents took %s' % (len(added_agents),
                            self.data_handler.get_time_string(time.process_time() - previous_time )))

            new_center_counter = 0
            for i, agent in enumerate(new_added_agents):
                if 'car2go' in agent.ID:
                    continue
                try:
                    # give new agent_ID to added agent and add it to self.agents
                    new_agent_ID = agent.ID + '_added'
                    self.agents[new_agent_ID] = agent
                    self.agents[new_agent_ID].ID = new_agent_ID
                    self.agents[new_agent_ID].start_date_agent = self.start_date_simulation

                except Exception as e:
                    print('\t\tWARNING: Creating added agents did not work for: %s' % e)

    def _create_activity_heap(self):
        ''' This method creates an activity heap (priority queue) which sorts
            the agents based on the time of their next activity. This activity
            heap is stored in the self.activity_heap attribute.

        Updates:
            activity_heap
        '''

        activity_heap = []
        for agent in self.agents.values():
            heapq.heappush(activity_heap, (agent.time_next_activity, agent))
        self.activity_heap = activity_heap

    def repeat_simulation(self, repeat = 30, measures = ['agent_validation'],
        method = 'relMAE', display_progress = True):
        ''' This method calls the simulate method a number of times.

        Kwargs:
            repeat (int): The number of times the simulate method is to be
                called. Default is 30.
            measures (List[str]): A list of measure names. Possible measures are
                charging_station_validation, agent_validation,
                time_per_simulation. #NOTE: update this list
            method (str): Method of validation in case validation is true.
                Options are 'MAE' or 'relMAE'. Default is 'relMAE'.
            display_progress (bool): If True, the progress of each simulation
                run will be shown.

        Updates:
            sensors
        '''

        if self.agent_initialization in ['create_and_store', 'create']:
            print('ERROR: Cannot run simulation when storing or creating agents.')
            sys.exit()

        if 'time_per_simulation' in measures:
            self.sensors['time_per_simulation'] = []

        for i in range(repeat):
            if self.info_printer:
                print('INFO: Simulation repeat %d of %d' % (i + 1, repeat))
            begin = time.process_time()

            self._simulate(display_progress = display_progress)

            if 'time_per_simulation' in measures:
                self.sensors['time_per_simulation'].append(
                    time.process_time() - begin)

            if self.info_printer:
                if display_progress:
                    print()
                print('\tINFO: Complete simulation run (%d of %d) took %s' %
                    (i + 1, repeat,
                    self.data_handler.get_time_string(time.process_time() - begin)))

        if 'agent_validation' in measures or 'agent_validation_with_IDs' in measures:
            if self.info_printer:
                before = time.process_time()
                print('INFO: Validating agents')
            if 'agent_validation' in measures and 'agent_validation_with_IDs' in measures:
                self.sensors['agent_validation_with_IDs'] = self._validate_agents(method = method, with_IDs = True)
                self.sensors['agent_validation'] =  {val_type: [val for ID, val in self.sensors['agent_validation_with_IDs'][val_type]] for val_type in self.sensors['agent_validation_with_IDs']}
            elif 'agent_validation' in measures:
                self.sensors['agent_validation'] = self._validate_agents(method = method)
            else:
                self.sensors['agent_validation_with_IDs'] = self._validate_agents(method = method, with_IDs = True)
            if self.info_printer:
                print('\tINFO: Validating agents took %s' %
                    self.data_handler.get_time_string(time.process_time() - before))

        if 'charging_station_validation' in measures or 'charging_station_validation_with_location_keys' in measures:
            if self.info_printer:
                before = time.process_time()
                print('INFO: Validating charging stations')
            if 'charging_station_validation' in measures and 'charging_station_validation_with_location_keys' in measures:
                self.sensors['charging_station_validation_with_location_keys'] = self.charging_station_validation(method = method, with_location_keys = True)
                self.sensors['charging_station_validation'] = {val_type: [val for ID, val in self.sensors['charging_station_validation_with_location_keys'][val_type]] for val_type in self.sensors['charging_station_validation_with_location_keys']}
            elif 'agent_validation' in measures:
                self.sensors['charging_station_validation'] = self.charging_station_validation(method = method)
            else:
                self.sensors['charging_station_validation_with_location_keys'] = self.charging_station_validation(method = method, with_location_keys = True)

            if self.info_printer:
                print('\tINFO: Validating charging stations took %s' %
                    self.data_handler.get_time_string(time.process_time() - before))

        if 'simulated_sessions' in measures:
            self.sensors['simulated_sessions'] = self.get_simulated_sessions()

        if 'number_of_charging_stations_in_simulation' in measures:
            all_css = set()
            for agent in self.agents.values():
                for css in agent.centers_css.values():
                    for cs in css['habit']:
                        all_css.add(cs)
            self.sensors['number_of_charging_stations_in_simulation'] = [len(all_css)]

        if 'number_of_unique_users_with_location_keys' in measures:
            all_css = []
            counter = {}
            for agent in self.agents.values():
                for css in agent.centers_css.values():
                    all_css += css['habit']
                    for cs in css['habit']:
                        if cs in counter:
                            counter[cs] += 1
                        else:
                            counter[cs] = 1
            self.sensors['number_of_unique_users_with_location_keys'] = list(counter.items())

        if 'number_of_agents_per_charging_station' in measures:
            all_css = []
            counter = {}
            for agent in self.agents.values():
                for css in agent.centers_css.values():
                    all_css += css['habit']
                    for cs in css['habit']:
                        if cs in counter:
                            counter[cs] += 1
                        else:
                            counter[cs] = 1
            self.sensors['number_of_agents_per_charging_station'] = list(counter.values())

        if 'selection_process_attempts' in measures:
            self.sensors['selection_process_attempts'] = []
            
            for agent in self.agents.values():
                all_percentages = agent.sensors['selection_process_attempts']
                # for run_nr in range(len(agent.sensors['selection_process_attempts'])):
                # all_percentages.append(agent.sensors['selection_process_attempts'][run_nr])
                self.sensors['selection_process_attempts'].append(all_percentages)
                all_percentages = agent.sensors['failed_process_attempts']
                
        if 'failed_connection_attempt' in measures:
            self.sensors['failed_connection_attempt'] = []
            for agent in self.agents.values():
                self.sensors['failed_connection_attempt'].append(agent.sensors['failed_connection_attempt'])
                
        if 'failed_sessions' in measures:
            self.sensors['failed_sessions'] = []
            for agent in self.agents.values():
                self.sensors['failed_sessions'].append(agent.sensors['list_failed_charge_session'])

                
    def reset(self):
        ''' This method resets the agents and (re)creates the activity heap
            according to the new time of next activity of the agents.

        Updates:
            agents
            activity_heap
        '''

        for agent in self.agents.values():
            agent.set_initial_state()
        self._create_activity_heap()

    def _simulate(self, display_progress = True):
        ''' This method contains the simulation loop. The agent with the
            smallest time of next activity is allowed to execute their activity,
            which generates a new time of next activity for this agent.
            Then the next agent is gotten by once again getting the agent with
            the smallest time of next activity. The simulation stops based on
            the specified stop condition.

        Kwargs:
            display_progress (bool): Set to true to see the progress of the
                simulation.

        Updates:
            current_time
            sensors['nr_activities_executed_per_agent']
            simulation_total_duration
            simulation_progress
        '''

        self.progress_simulation = 0.0

        if self.stop_condition == 'time':
            self.simulation_total_duration = self.stop_condition_parameters['max_time'] - \
                self.start_date_simulation

        self.current_time = self.start_date_simulation
        self.sensors['nr_activities_executed_per_agent'].append({agent: 0 for agent in self.agents})
        self.sensors['nr_activities_executed_per_cp'].append({cp: 0 for cp in self.environment.css_info})

        if self.previously_simulated:
            self.reset()

        self.previously_simulated = True
        (self.current_time, active_agent) = heapq.heappop(self.activity_heap)

        while not self._stop(display_progress):
            time_next_activity, succesfully_executed_activity = \
                self._execute_activity(active_agent)
            #print(active_agent.ID, time_next_activity)

            if succesfully_executed_activity:
                self.sensors['nr_activities_executed_per_agent'][-1][active_agent.ID] += 1
                self.sensors['nr_activities_executed_per_cp'][-1][active_agent.active_cs] += 1
            item = (time_next_activity, active_agent)
            (self.current_time, active_agent) = heapq.heappushpop(self.activity_heap,
                 item)

        for agent in self.agents:
            self.agents[agent].save_simulation_data()
        self.environment.reset()

    def _stop(self, display = True):
        ''' This method tells the simulation whether is should stop, based
            on the selected stop condition and its parameters.

            Possible stop conditions are 'time' and
            'nr_activities_executed_per_agent':
            - 'time' means the simulation stops after the current time of the
            simulation passes the max time (which should be given in the
            input file).
            - 'nr_activities_executed_per_agent' means the simulation stops
            after a specified number of activities per agent have been executed.
            This max number of activities should be given in the input file.

        Kwargs:
            display (bool): Set to true to see the progress of the simulation.

        Returns:
            (bool): True indicates that the simulation should stop.
        '''

        if self.stop_condition == 'time':
            if display and self.info_printer:
                simulation_progress = (self.current_time - self.start_date_simulation)
                if self.progress_simulation != round(float(simulation_progress /
                    self.simulation_total_duration), 1):
                    sys.stdout.write('\r' + '\tINFO: Progress of simulation: %.1f'
                        % (float(simulation_progress / self.simulation_total_duration)))
                    sys.stdout.flush()
                    self.progress_simulation = round(float(simulation_progress /
                        self.simulation_total_duration), 1)
            return self.current_time >= self.stop_condition_parameters['max_time']

        if self.stop_condition == 'nr_activities_executed_per_agent':
            if display:
                simulation_progress = min(self.sensors[nr_activities_executed_per_agent][-1].values())
                if self.progress_simulation != round(float(simulation_progress /
                    self.stop_condition_parameters['min_nr_activities_executed_per_agent']), 1):
                    sys.stdout.write('\r\tINFO: Progress of simulation: %.1f' %
                        float(simulation_progress /
                        self.stop_condition_parameters['min_nr_activities_executed_per_agent']))
                    sys.stdout.flush()
                    self.progress_simulation = round(float(simulation_progress /
                        self.stop_condition_parameters['min_nr_activities_executed_per_agent']), 1)

            return min(self.sensors['nr_activities_executed_per_agent'][-1].values()) >= \
                self.stop_condition_parameters['min_nr_activities_executed_per_agent']

        print('ERROR: Stop condition (%s) is not valid.' % self.stop_condition)
        sys.exit()

    def _execute_activity(self, agent):
        ''' This method executes the current activity of the agents and determines
            the time of the next activity of the agent (and possibly the next
            center of the agent).

        Args:
            agent (Agent): The agent whose activity will be executed.

        Updates:
            agent.time_next_activity
            agent.active_center
            agent.active_cs
            environment.css_occupation

        Returns:
            agent.time_next_activity (DateTime): The time of the next activity
                of the agent.
            succesfully_executed_activity (bool): Indicates if the agent has
                succesfully executed their next activity.
        '''


        if agent.is_connected:
            # t = datetime.datetime.now()
            ''' Current activity of agent is disconnecting '''
            self.environment.disconnect_agent(agent.ID, agent.active_cs)
            agent.is_connected = False
            # print('%s: disconnectin1 took %s' % (agent.ID[:5], datetime.datetime.now() - t))
            # t = datetime.datetime.now()

            ''' Next activity of agent is connecting '''
            agent.get_next_connection_time_and_place()
            succesfully_executed_activity = True
            # print('%s: disconnectin2 took %s' % (agent.ID[:5], datetime.datetime.now() - t))

            # print('%s: disconnecting took %s' % (agent.ID[:5], datetime.datetime.now() - t))
        else:
            ''' Current activity of agent is connecting '''
            if agent.is_possible_to_connect(self.current_time):
                # t  = datetime.datetime.now()
                
                # if agent should be deleted, as it is not in use anymore, take its time next activity to be end simulation
                if agent.time_next_activity > agent.end_date_agent:
                    agent.time_next_activity = self.stop_condition_parameters['max_time']
                    succesfully_executed_activity = False
                    return agent.time_next_activity, succesfully_executed_activity

                agent.get_next_disconnection_time()
                agent.select_cs(self.current_time)
                # print('update in _execute_activity')
                #print(agent.ID, "update_history")
                agent.update_history(self.current_time)

                agent.is_connected = True
                self.environment.connect_agent(agent.ID, agent.active_cs)

                ''' Next activity of agent is disconnecting '''

                succesfully_executed_activity = True
                agent.sensors['selection_process_attempts'][agent.sensors['run_counter']].append(False)

                # print('%s:    connecting took %s' % (agent.ID[:5], datetime.datetime.now() - t))

            else:
                #agent.history = agent.history[:-1]
                succesfully_executed_activity = False
                agent.sensors['selection_process_attempts'][agent.sensors['run_counter']].append(True)
                
                # if selection_process is strategy model, we are going to simulate the selection process, 
                # even though there are no available cps. This will enable us to measure cascades and 
                # failed cs and where this happens.
                if agent.selection_process == 'strategy_model':
                    agent.select_cs(self.current_time)
                elif agent.selection_process == 'habit_distance':
                    agent.select_cs(self.current_time)
                    
                occupied = set()
                for cs in agent.centers_css[agent.active_center]['distance']:
                    if self.environment.css_info[cs]['placement_date'] < agent.time_next_activity and \
                    self.environment.css_info[cs]['amount_of_sockets'] > 0:
                        # print('3) adding %s' % str(tuple([cs, agent.time_next_activity, tuple(self.environment.who_occupies(cs)), -1])))
                        occupied.add((cs, agent.time_next_activity, tuple(self.environment.who_occupies(cs)), -1))
                        if len(self.environment.who_occupies(cs)) != len(set(self.environment.who_occupies(cs))):
                            print('ERROR: agent twice in sockets of same CP (%s).' % cs)
                            
                #agent.sensors['failed_charge_sessions'][agent.sensors['run_counter']] += 1
                agent.sensors['occupied_counter'][agent.sensors['run_counter']].append(occupied)
                
                
                # if agent should be deleted, as it is not in use anymore, take its time next activity to be end simulation
                if self.current_time < agent.end_date_agent:
                    agent.get_next_connection_time_and_place(failed_session = True)
                else:
                    agent.time_next_activity = self.stop_condition_parameters['max_time']
                
                #agent.time_next_activity = self.current_time + datetime.timedelta(
                #    minutes = agent.time_retry_center)
                
#                 agent._check_next_connection_time(self.current_time, backwards = False,
#                     next_center = agent.active_center)

        return agent.time_next_activity, succesfully_executed_activity

    def _validate_agents(self, method = 'relMAE', with_IDs = False):
        ''' This method returns the errors for every agent on the training
            and the test data.

        Kwargs:
            method (str): Method of validation. 'MAE' (Mean Absolute Error) or
                'relMAE' (relative Mean Absolute Error). Default 'relMAE'.
            with_IDs (boolean): Indicates whether the errors should be stored
                as float, or as a tuple containing the agent ID and the error.

        Returns:
            all_errors (Dict[str, Any]): As keys we have 'training' and 'test'
                and each key contains a list of values of the means of the errors
                of each agent (calculated by the specified measure method).
        '''

        all_errors = {'training': [], 'test': []}

        for agent in self.agents.values():
            error = agent.validate(method = method, with_IDs = with_IDs)
            all_errors['training'].append(error['training'])
            all_errors['test'].append(error['test'])
        return all_errors

    def charging_station_validation(self, method = 'relMAE', with_location_keys = False,
        visualize_overall = False, visualize_individual = False):
        ''' This method performs the validation of the charging stations.
            Optionally this validation can be visualized. All charging stations
            that are in a center of an agent (either in the training data or
            in the simulated data) are considered. For each of those charging
            stations we consider all the sessions at that charging station by
            the agents that are in the simulation. The activity patterns of the
            charging stations based on those sessions are used for validation.

        Kwargs:
            method (str): Method of validation. 'MAE' (Mean Absolute Error) or
                'relMAE' (relative Mean Absolute Error). Default 'relMAE'.
            with_location_keys (bool): If true, store each validation error
                together with its corresponding location key.
            visualize_overall (bool): If put on True, a histogram is created
                showing the validation values for all charging stations.
                Default is False.
            visualize_individual (bool): If put on True, per charging station
                two activity plots are created, one for training and one for
                test, each of those plots contains real (training/test) data
                and simulated data.

        Returns:
            all_errors (Dict[str, List]): A dictionary containing all the
                errors of the charging station for training and test data. The
                keys of the dictionary are 'training' and 'test' and each of
                those contains a list with error values for the indicated data.
        '''

        all_errors = {'training': [], 'test': []}

        simulated_sessions_of_all_agents_per_run = []
        all_agents = list(self.agents.values())
        for run in range(len(all_agents[0].all_simulated_sessions)):
            simulated_sessions_of_all_agents_per_run.append(pandas.concat(
                [agent.all_simulated_sessions[0] for agent in all_agents]))

        all_css = []
        counter = {}
        for agent in all_agents:
            for css in agent.centers_css.values():
                all_css += css['habit']
                for cs in css['habit']:
                    if cs in counter:
                        counter[cs] += 1
                    else:
                        counter[cs] = 1
        all_css = set(all_css)

        filtered_training_sessions = self.data_handler.training_data[
            self.data_handler.training_data['ID'].isin(list(self.agents.keys()))]
        filtered_test_sessions = self.data_handler.test_data[
            self.data_handler.test_data['ID'].isin(list(self.agents.keys()))]

        for i, cs in enumerate(all_css):
            if self.info_printer:
                if i % 500 == 0:
                    print('INFO: charging station validation validating %d of %d' %
                        (i, len(all_css)))
            all_agents = list(self.agents.values())
            mean_simulated_pattern = self.data_handler.get_activity_pattern(
                simulated_sessions_of_all_agents_per_run[0][simulated_sessions_of_all_agents_per_run[0]['location_key'] == cs])
            for simulated_sessions_of_all_agents in simulated_sessions_of_all_agents_per_run[1:]:
                mean_simulated_pattern += self.data_handler.get_activity_pattern(
                    simulated_sessions_of_all_agents[simulated_sessions_of_all_agents_per_run[0]['location_key'] == cs])
            if numpy.sum(mean_simulated_pattern) != 0:
                mean_simulated_pattern /= numpy.sum(mean_simulated_pattern)

            sessions_training = filtered_training_sessions[filtered_training_sessions['location_key'] == cs]
            sessions_test = filtered_test_sessions[filtered_test_sessions['location_key'] == cs]
            figs = []
            if len(sessions_training) > 0 and len(sessions_test) > 0:
                unique_training = len(sessions_training.ID.unique())
                pattern_training = self.data_handler.get_activity_pattern(sessions_training)
                pattern_training /= numpy.sum(pattern_training)
                error_value = self.data_handler.get_error(pattern_training, mean_simulated_pattern,
                    method = method)
                if with_location_keys:
                    all_errors['training'].append((cs, error_value))
                else:
                    all_errors['training'].append(error_value)

                if visualize_individual:
                    figs.append(self._get_barplot(pattern_training, mean_simulated_pattern,
                        "training (%s): %.2f (# unique training/simulated: %d/%d)" %
                        (cs[:6], error_value, unique_training, counter[cs])))

                unique_test = len(sessions_test.ID.unique())
                pattern_test = self.data_handler.get_activity_pattern(sessions_test)
                pattern_test /= numpy.sum(pattern_test)
                error_value = self.data_handler.get_error(pattern_test,
                    mean_simulated_pattern, method = method)

                if with_location_keys:
                    all_errors['test'].append((cs, error_value))
                else:
                    all_errors['test'].append(error_value)

                if visualize_individual:
                    figs.append(self._get_barplot(pattern_test, mean_simulated_pattern,
                        "test (%s): %.2f (# unique test/simulated: %d/%d)" %
                        (cs[:6], error_value, unique_test, counter[cs])))
                    IPython.display.display(ipywidgets.HBox(figs))

        if visualize_overall:
            scale_x = bqplot.LinearScale()
            scale_y = bqplot.LinearScale()
            ax_x = bqplot.Axis(label='counts', scale=scale_x, tick_format='0.2f')
            ax_y = bqplot.Axis(label='relMAE errors', scale=scale_y,
                orientation='vertical', grid_lines='solid')

            training_len = len(all_errors['training'])
            test_len = len(all_errors['test'])
            hist_training = bqplot.Hist(sample = all_errors['training'],
                scales={'sample': scale_x, 'count': scale_y}, colors = ['blue'],
                opacities = [0.5] * training_len, labels = ['Training'],
                display_legend = True)
            hist_test = bqplot.Hist(sample = all_errors['test'],
                scales={'sample': scale_x, 'count': scale_y}, colors = ['green'],
                opacities = [0.5] * test_len, labels = ['Test'],
                display_legend = True)
            hist_training.bins = 30
            hist_test.bins = 30

            fig = bqplot.Figure(axes=[ax_x, ax_y], marks=[hist_training, hist_test])
            IPython.display.display(fig)

        return all_errors

    def get_simulated_sessions(self):

        simulated_sessions_of_all_agents_per_run = []
        all_agents = list(self.agents.values())
        for run in range(len(all_agents[0].all_simulated_sessions)):
            simulated_sessions_of_all_agents_per_run.append(pandas.concat(
                [agent.all_simulated_sessions[0] for agent in all_agents]))
        return simulated_sessions_of_all_agents_per_run


    def _get_barplot(self, dist_real, dist_simulated, title):
        ''' This method creates a bar plot of the given distribution.

        Args:
            dist (DataFrame): Contains the distribution.
            title (str): The name of the distribution, which will be plotted
                as a label in the figure. Note that if the title is equal to
                'Disconnection duration dist', the figure will be adjusted
                (different margins, size, title location and font size).

        Returns:
            fig_activity (bqplot.figure.Figure): The resulting bar plot figure.
        '''

        sc_x1, sc_y1 = bqplot.OrdinalScale(), bqplot.LinearScale()
        c_sc = bqplot.OrdinalColorScale(colors = [bqplot.CATEGORY10[2]])

        sc_y1.min, sc_y1.max = 0, float(max(dist_real)) * 1.1

        bar_x = bqplot.Axis(scale = sc_x1, orientation = 'horizontal',
                grid_lines = 'none')
        bar_y = bqplot.Axis(scale = sc_y1, orientation = 'vertical',
                tick_format = '0.0f', grid_lines = 'none')

        bar_chart_real = bqplot.Bars(x = range(len(dist_real.index)),
            y = dist_real, color_mode = 'element', colors = ['black'],
            scales = {'x': sc_x1, 'y': sc_y1}, opacities = [0.5] * len(dist_real),
            stroke = 'black')

        bar_chart_simulated = bqplot.Bars(x = range(len(dist_real.index)),
            y = dist_simulated, color_mode = 'element', color = [1] * len(dist_simulated),
            scales = {'x': sc_x1, 'y': sc_y1, 'color': c_sc},
            opacities = [0.5] * len(dist_simulated), stroke = 'black')

        title_label = bqplot.Label(x = [0.1], y = [0.4],
            font_size = 15, font_weight = 'bolder', colors = ['black'],
            text = [title], enable_move = True)

        fig_activity = bqplot.Figure(marks=[bar_chart_real, bar_chart_simulated,
            title_label], min_width = 200, min_height = 150, axes=[bar_x, bar_y],
            fig_margin = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0})

        return fig_activity

 