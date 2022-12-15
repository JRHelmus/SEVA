'''
agent.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Updated by:
    Jerome Mies
    Jurjen Helmus
    
update contains
    - failed connection attempts 

Last updated August 2020 
'''
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy
import sys
import datetime
import pickle
import os
import time
import operator
import shapely
import bqplot
import ipywidgets
import ipyleaflet
import IPython
import random
import math
import number_of_choices_per_parameter
#sys.path.insert(0, 'nn/')
#from NN_utilities import load_model_sim
#from pprint import pprint
#from NN_SP_preprocess import preprocess_single_session

class Agent():
    ''' The Agent class deals with an agent using its three processes of
        disconnecting, connecting and selecting charging stations.

    Args:
        args (Dict[str, Any]): Arguments which the agent needs. This contains:
            'data_handler' (DataHandler): Instance of the DataHandler class;
            'environment' (Environment): Instance of the Environment class;
            'parameters' (Dict[str, Any]): The parameters of the agent that
                were given in the input file;
            'start_date_simulation' (DateTime): The start date of the
                simulation;
            'warmup_period' (TimeDelta): The duration of the warmup period;
            'agent_initialization' (str): This attribute determines whether
                to load agents from memory or to create new ones and store
                them to memory. Options are 'create', 'create_and_use',
                'create_and_store' and 'load_and_use'.
        overwrite_parameters (Dict[str, Any]): Parameter values to use instead
            of the values specified in the input file. Often used for
            experiments. Parameters that can be overwritten can be found in the
            readme.
        info_printer (bool): Parameter to decide whether to print information
            about run times.
        agent_ID (str): The ID of the agent.
        filepath_agent_database (str): Path to the agent database.

    Kwargs:
        measures (List[str]): A list of values to measure (usually for
            experiments). Possible measures are number_of_centers,
            number_of_charging_stations and walking_preparedness.
        sim_sensors (Dict[str, List]): A dictionary containing measures as keys
            and a list of their values of the run as value at that key.

    Attributes:
        ID (str): Unique identifier of the agent.
        data_handler (DataHandler): Instance of DataHandler object.
        environment (Environment): Instance of Environment object.
        training_sessions (DataFrame): The training sessions of the agent.
        test_sessions (DataFrame): The test sessions of the agent.
        simulated_sessions (DataFrame): The simulated sessions of the agent.
        is_connected (bool): Contains the state of the agent.
        centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
            (lon, lat) as keys and each center being a dictionary with the
            keys 'habit' and 'distance'. The value of the habit key is a set
            of charging stations (location keys), while the value of the
            distance key is a dictionary with the charging stations
            (location keys) as keys and their distance to the center
            (in meters) as value.
        original_centers_css (Dict[Tuple[float, float], Set]): Centers (lon, lat)
            as keys and each value being a set of location keys of the charging
            stations in that center.
        centers_info (Dict[Tuple[float, float], Dict[str, Any]]): Centers
            (lon, lat) as keys with their corresponding values a dictionary
            with info about the center:
            - nr_of_sessions_in_center: amount of sessions within center,
            - center_nr: unique number of the center (for visualization),
        disconnection_duration_dists (Dict[Tuple[float, float]: List[float]]): A
            dictionary of disconnection duration distributions with centers as
            keys and the distributions as values. The distributions are
            lists of probabilities in each bin.
        connection_duration_dists (Dict[Tuple[float, float]: List[float]]): A
            dictionary of connection duration distributions with centers as
            keys and the distributions as values. The distributions are
            lists of probabilities in each bin.
        arrival_dists (Dict[Tuple[float, float], List[float]]): A dictionary of
            arrival distributions with the centers as keys. The distributions
            are lists of probabilities in each bin.
        activity_patterns_training (Dict[Tuple[float, float], List[float]]): A
            dictionary of activity patterns based on the training data with the
            centers as keys. The activity patterns are lists of probabilities
            of activity in each bin.
        activity_patterns_test (Dict[Tuple[float, float], List[float]]): A
            dictionary of activity patterns based on the test data with the
            centers as keys. The activity patterns are lists of probabilities
            of activity in each bin.
        preferences (Dict[Tuple[float, float], Dict[str, int]]): The tuples
            (lon, lat) of the center as keys and as value a dictionary with
            the preference for each charging station as value and the
            charging station (location key) as key.
        selection_process (str): 'habit_distance' or 'choice_model'.
        selection_process_parameters (Dict[str, float] or
            Dict[str, Dict[str, float]]): If selection_process is
            'habit_distance', the key is 'habit_probability' and the value
            the fraction of time the agent bases its
            selection choice on its habitual preference (habit).
            If the selection process is "choice_model" the dictionary
            contains the a dictionary with keys "Amsterdam", "The Hague",
            "Rotterdam" and "Utrecht". This dictionary should contain the
            keys "intercept", "distance", "charging_speed", "charging_fee" and
            "parking_fee" with the logit model coefficient values as values.
        time_next_activity (DateTime): Time at which the next activity takes
            place. The next activity can be either a disconnection or a
            connection.
        time_retry_center (int): Number of minutes to wait until retrying to
            connect to a charging station in the active center when all
            charging stations are currently occupied.
        minimum_radius (float): Default maximum distance between
            a center to a charging station in meters.
        walking_preparedness (float): Maximum distance between a center to
            a charging station in meters multiplied by 1.1. If this value is
            lower than the minimum_radius, it default to the
            minimum_radius
        maximum_distance (float): Maximum distance between a center to
            a charging station in meters.
        active_center (Tuple[float, float]): Center (lon, lat) at which agent
            is connected or plans to connect to.
        active_cs (str): The location key of the charging station at which the
            agent is connected.
        history (List[Dict[str, Any]]): Each element of the history contains
            a dictionary with information about the decision process for an
            activity taken by the agent. The dictionary contains
            the keys time, connected, next_center, disconnection_duration,
            corrected_disconnection_duration, arrival_probs, time_next_activity,
            centers.
        start_date_simulation (DateTime): The start date of the simulation.
        warmup_period (TimeDelta): The duration of the warmup period.
        sensors (Dict[str, Any]): A dictionary containing observing information
            about the agent. The keys and values are:
        disconnection_duration_mistake_counters (List[int]): The number of
            times a wrong disconnection duration was chosen in a run. This
            duration is wrong when the agent tries to connect but all
            centers have a probability of zero of starting to connect at
            that time.
        connection_duration_mistake_counters (List[int]): The number of
            times a wrong connection duration was chosen in a run. This
            duration is wrong when the agent has never disconnected at this
            time (and therefore its disconnection duration distribution is
            an empty distribution at this time).
        selection_process_attempts (List[int]): The number of times per run
            a charging station was selected which was already in use.
        total_disconnections (List[int]): The number of times the agent
            disconnected per run.
        total_connections (List[int]): The number of times the agent
            connected per run.
        total_selections (List[int]): The number of times the agent selected
            a charging station per run.
        run_counter (int): The curent run number.
        all_simulated_sessions (List[DataFrame]): A list containing the
            simulated sessions (in form of a DataFrame) of all simulation runs
            of the agent.
        activity_patterns_training (List[float]): The activity patterns of
            the training set of the agent.
        activity_patterns_test (List[float]): The activity patterns of the
            test set of the agent.
        simulated_activity_patterns (List[float]): The activity
            patterns simulated for the agent.
        errors (Dict[str, Dict[Tuple[float, float], Dict[str, float]]]]): A
            dictionary with the keys 'MEA' and 'relMAE'. Under these keys there
            is a dictionary with centers (lon, lat) as keys and each of those
            has a dictionary as value with as keys 'training' and 'test' and
            as values the mean error of this data, center and error method.
        car_type (str): The car type of the agent. Possible options are 'PHEV'
            or 'FEV'.
        transform_parameters (Dict[str, float]): Contains information about
            which fractions of the (phev) population should be transformed to
            either low battery fev or high battery fev agents. The keys are:
                - prob_no_transform;
                - prob_to_low_fev;
                - prob_to_high_fev.
        cs_zone_information (Dict[Tuple[float, float], Dict[str, str]])
    '''

    def __init__(self, args, info_printer, agent_ID, overwrite_parameters,
        filepath_agent_database,
        # cluster_model_1, cluster_model_2, cluster_model_3, cluster_model_4, cluster_model_5,
        measures = [], sim_sensors = {}):

        self.info_printer = info_printer
        self.ID = agent_ID
        self.simulation_parameters = args['simulation_parameters']
        #self.previous_session = pandas.DataFrame(columns=current_columns)
        
        
        # NOTE: feels like there should be a check_args() here
        # NOTE 2: feels like a lot of the inits can be put here (environment, etc)
        # NOTE one more: thought, with the 'load_and_use', should we check if the bin size fits the loaded agents?
        self.data_handler = args['data_handler']
        if not self._load_and_check_parameters(args['parameters'],
            overwrite_parameters, args['agent_initialization']):
            sys.exit()
        self.cs_zone_information = {}



        if 'time_per_initialization' in measures:
            process_time_start = time.process_time()

        if args['agent_initialization'] == 'create_and_use':
            self.environment = args['environment']
            self.warmup_period = args['warmup_period']
            self.start_date_simulation = args['start_date_simulation']
            if not self._load_and_check_parameters(args['parameters'],
                overwrite_parameters,
                args['agent_initialization']):
                sys.exit()
            if 'car2go' in self.ID:
                self._load_sessions_car2go()
                self._load_centers_car2go(args['parameters'])
                self._load_distributions_car2go()
            else:
                self._load_sessions()
                self._load_centers(args['parameters'])
                self._load_distributions()
            self.check_center_for_city()
            self._load_additional_attributes()
            self._reset_results()
            transform_to = self._should_transform()
            self._transform_agent(transform_to)
            self.set_initial_state()
        elif args['agent_initialization'] == 'create_and_store':
#             print("Agent started to be created")
            self.environment = args['environment']
            self.warmup_period = args['warmup_period']
            self.start_date_simulation = args['start_date_simulation']
            if not self._load_and_check_parameters(args['parameters'],
                overwrite_parameters,
                args['agent_initialization']):
                sys.exit()
            if 'car2go' in self.ID:
                self._load_sessions_car2go()
                self._load_centers_car2go(args['parameters'])
                self._load_distributions_car2go()
            else:
                self._load_sessions()
                self._load_centers(args['parameters'])
                self._load_distributions()
            self.check_center_for_city()
            self._load_additional_attributes()

            my_agent_data = self._get_agent_data()
            if self.simulation_parameters['agent_creation_method'] not in ['given', 'previous']:
#                 print(self.ID)
                if 'car2go' in self.ID:
#                     print(self.ID)
#                     print(my_agent_data)
                    if not os.path.isfile(filepath_agent_database + str(self.ID)[:10] + '.pkl'):
                        with open(filepath_agent_database + str(self.ID)[:10] + '.pkl',
                            'wb') as agent_file:
                            pickle.dump(my_agent_data, agent_file)
                else:
                    if not os.path.isfile(filepath_agent_database + str(self.ID) + '.pkl'):
                        with open(filepath_agent_database + str(self.ID) + '.pkl',
                            'wb') as agent_file:
                            pickle.dump(my_agent_data, agent_file)
        elif args['agent_initialization'] == 'create':
            self.data_handler = args['data_handler']
            self.environment = args['environment']
            self.warmup_period = args['warmup_period']
            self.start_date_simulation = args['start_date_simulation']
            if not self._load_and_check_parameters(args['parameters'],
                overwrite_parameters,
                args['agent_initialization']):
                sys.exit()
            if 'car2go' in self.ID:
                self._load_sessions_car2go()
                self._load_centers_car2go(args['parameters'])
                self._load_distributions_car2go()
            else:
                self._load_sessions()
                self._load_centers(args['parameters'])
                self._load_distributions()
            self._load_additional_attributes()
        elif args['agent_initialization'] == 'load_and_use':
           
            self.environment = args['environment']
            self.warmup_period = args['warmup_period']
            self.start_date_simulation = args['start_date_simulation']
            if not self._load_and_check_parameters(args['parameters'],
                overwrite_parameters,
                args['agent_initialization']):
                sys.exit()
                
            # when loading data from pickle file, Car2go data is not always loaded and thus needs to come from
            # folder with all individual agents, hence we have a different filepath_agent_database here
            if not self.simulation_parameters['IDs_from_agent_database']:
                filepath_agent_database_car2go = "data/agent_database/all_agents/" 
                if 'car2go' in self.ID:
                    self._load_sessions_car2go()
                    with open(filepath_agent_database_car2go + str(self.ID)[:10] + '.pkl', 'rb') \
                        as agent_file:
                        my_agent_data = pickle.load(agent_file)
                    self.start_date_agent = self.start_date_simulation
                    self._set_agent_data(my_agent_data)
                    self.check_center_for_city()
                elif 'added' in self.ID:
                    pass
                else:
                    raise Exception('WARNING: Load_and_use but not from agent_database but still this agent appears ' + \
                                    'here: (%s). \n This should not happen.' % self.ID)
                        
            if self.simulation_parameters['IDs_from_agent_database']:
                if self._agent_in_database(filepath_agent_database):
                    self._load_sessions()
                    agent_file = filepath_agent_database + str(self.ID) + '.pkl'
                    my_agent_data = pandas.read_pickle(agent_file)
                    self._set_agent_data(my_agent_data)
                    self._check_skip()
                    self.check_center_for_city()
                

                elif 'car2go' in self.ID:
                    #last check if distances to cps are larger than the normal walking distance,
                    # if this is not the case, we need to redefine the centers_css of the car2go agent:
                    max_distance = 0
                    for center in self.centers_css:

                        for cs in self.centers_css[center]['distance']:
                             if self.centers_css[center]['distance'][cs] > max_distance:
                                    max_distance = self.centers_css[center]['distance'][cs]
                    if max_distance < 501:
                        self._load_centers_car2go(args['parameters'])
                    
                elif 'car2go' not in self.ID:
                    raise Exception('WARNING: Agent (%s) does not exist in agent database.' %
                        self.ID)
            self.start_date_agent = self.simulation_parameters['start_date_simulation']
            self.end_date_agent = self.simulation_parameters['stop_condition_parameters']['max_time']
            self._reset_results()
            transform_to = self._should_transform()
            self._transform_agent(transform_to)
            self.set_initial_state()
            
        else:
            raise Exception('ERROR: Wrong input for agent_initialization.')

        if 'number_of_centers' in measures:
            sim_sensors['number_of_centers'].append(len(self.original_centers_css.keys()))
        if 'number_of_centers_with_IDs' in measures:
            sim_sensors['number_of_centers_with_IDs'].append((self.ID, len(self.original_centers_css.keys())))

        if 'average_number_of_charging_stations' in measures:
            average_nr_of_css = numpy.mean([len(css)
                for center, css in self.original_centers_css.items()])
            sim_sensors['average_number_of_charging_stations'].append(average_nr_of_css)
        if 'average_number_of_charging_stations_with_IDs' in measures:
            average_nr_of_css = numpy.mean([len(css)
                for center, css in self.original_centers_css.items()])
            sim_sensors['average_number_of_charging_stations_with_IDs'].append((self.ID, average_nr_of_css))

        if 'walking_preparedness' in measures:
            sim_sensors['walking_preparedness'].append(self.walking_preparedness)
        if 'maximum_distance' in measures:
            sim_sensors['maximum_distance'].append(self.maximum_distance)
        if 'time_per_initialization' in measures:
            sim_sensors['time_per_initialization'].append(time.process_time() - process_time_start)

        if 'number_of_training_sessions_with_IDs' in measures:
            sim_sensors['number_of_training_sessions_with_IDs'].append((self.ID, len(self.training_sessions)))

    def _agent_in_database(self, filepath_agent_database):
        ''' This method checks if an agent is already in the agent database.

        Args:
            filepath_agent_database (str): Path to the agent database.

        Returns:
            (bool): True if the agent exists in the agent database.
        '''
        if 'car2go' in self.ID:
            return os.path.isfile(filepath_agent_database + str(self.ID)[:10] + '.pkl')
        else:
            return os.path.isfile(filepath_agent_database + str(self.ID) + '.pkl')

    def _load_sessions_car2go(self):
        ''' This method loads the sessions of the agent and stores them as
            attributed. 
            

        Updates:
            training_sessions
            test_sessions
        '''
        self.training_sessions = self.data_handler.training_data.loc[
            self.data_handler.training_data.ID == self.ID]
        self.test_sessions = self.data_handler.test_data.loc[
            self.data_handler.test_data.ID == self.ID]
        self._get_disconnection_duration()

    def _load_sessions(self):
        ''' This method loads the sessions of the agent and stores them as
            attributed. Raises an exception when the agent has either too few
            training sessions or too few test sessions.

        Updates:
            training_sessions
            test_sessions
        '''

        agent_training_sessions = self.data_handler.training_data.loc[
            self.data_handler.training_data.ID == self.ID]
        self.test_sessions = self.data_handler.test_data.loc[
            self.data_handler.test_data.ID == self.ID]
        self.training_sessions = agent_training_sessions
        
#         try:
#             self.training_sessions = self.data_handler.check_gap_agent_sessions(
#                 agent_training_sessions)
#         except:
#             # NOTE: gap sessions checking fails when [NaT] comes out.
#             raise Exception('WARNING: Failed to check gap agent sessions for %s.' %self.ID)
        self.training_sessions = self.training_sessions.sort_values(
            by = 'start_connection')
        if len(self.training_sessions) < self.data_handler.minimum_nr_sessions_center:
            raise Exception('WARNING: Too few sessions (%d) for agent ' %
                len(self.training_sessions) + '(%s) in the training data.' %
                self.ID)

        if len(self.test_sessions) < self.data_handler.minimum_nr_sessions_center:
            raise Exception('WARNING: Too few sessions (%d) for ' %
                len(self.test_sessions) + 'agent (%s) in the test data.' %
                self.ID)
        self._get_disconnection_duration()

    def _load_and_check_parameters(self, parameters, overwrite_parameters,
        agent_initialization):
        ''' This method checks whether the parameters needed for the agent
            contain correct values.

        Args:
            parameters (Dict[str, Any]): The parameters from the input file.
            overwrite_parameters (Dict[str, Any]): The overwrite parameters.
            agent_initialization (str): This attribute determines whether
                to load agents from memory or to create new ones and store
                them to memory. Options are 'create', 'create_and_use',
                'create_and_store' and 'load_and_use'.


        Updates:
            selection_process
            selection_process_parameters
            time_retry_center
            minimum_radius

        Returns:
            (bool): Returns True if all parameters are valid.
        '''

        if 'selection_process' in overwrite_parameters:
            self.selection_process = overwrite_parameters['selection_process']
        else:
            self.selection_process = parameters['selection_process']

        if 'selection_process_parameters' in overwrite_parameters:
            self.selection_process_parameters = overwrite_parameters['selection_process_parameters']
        else:
            self.selection_process_parameters = parameters['selection_process_parameters']

        if self.selection_process == 'habit_distance':
            if not isinstance(self.selection_process_parameters, dict):
                print('ERROR: selection_process_parameters (%s) is not a dictionary.' %
                    self.selection_process_parameters)
                return False
            if 'habit_probability' not in self.selection_process_parameters:
                print('ERROR: habit_probability not in selection_process_parameters ' +
                    'when habit_distance is chosen as selection process.')
                return False
            if not isinstance(self.selection_process_parameters['habit_probability'], float):
                print('ERROR: habit_probability (%s) is not a float.' %
                    str(self.selection_process_parameters['habit_probability']))
                return False
            if self.selection_process_parameters['habit_probability'] < 0 or \
                self.selection_process_parameters['habit_probability'] > 1:
                print('ERROR: habit_probability (%.2f) is not between 0 and 1.' %
                    self.selection_process_parameters['habit_probability'])
                return False
        elif self.selection_process == 'choice_model':
            if not isinstance(self.selection_process_parameters, dict):
                print('ERROR: selection_process_parameters (%s) is not a dictionary.' %
                    self.selection_process_parameters)
                return False
            for city in ['Amsterdam', 'The Hague', 'Rotterdam', 'Utrecht']:
                if city not in self.selection_process_parameters:
                    print('ERROR: %s not in selection_process_parameters ' % city +
                        'when choice_model is chosen as selection process.')
                    return False
                if not isinstance(self.selection_process_parameters[city], dict):
                    print('ERROR: selection_process_parameters[%s] (%s) is not a dictionary.' %
                        (city, self.selection_process_parameters))
                    return False

                for key in ['intercept', 'distance', 'charging_speed', 'charging_fee', 'parking_fee']:
                    if key not in self.selection_process_parameters[city]:
                        print('ERROR: %s not in selection_process_parameters ' % key +
                            'when choice_model is chosen as selection process.')
                        return False
                    if not isinstance(self.selection_process_parameters[city][key], float) and \
                        not isinstance(self.selection_process_parameters[city][key], int):
                        print('ERROR: %s is not a float or int.' %
                            self.selection_process_parameters[city][key] +
                            'Its type is %s' % type(self.selection_process_parameters[city][key]))
                        return False
        elif self.selection_process == 'strategy_model':
            if not isinstance(self.selection_process_parameters, dict):
                print('ERROR: selection_process_parameters (%s) is not a dictionary.' %
                    self.selection_process_parameters)
                return False
            if 'age_compensation' not in self.selection_process_parameters:
                print('ERROR: age_compensation not in selection_process_parameters ' +
                    'when strategy_model is chosen as selection process.')
                return False
            if not isinstance(self.selection_process_parameters['age_compensation'], float):
                print('ERROR: age_compensation (%s) is not a float.' %
                    str(self.selection_process_parameters['age_compensation']))
                return False
            if self.selection_process_parameters['age_compensation'] < 0 or \
                self.selection_process_parameters['age_compensation'] > 1:
                print('ERROR: age_compensation (%.2f) is not between 0 and 1.' %
                    self.selection_process_parameters['age_compensation'])
                return False
        elif self.selection_process == 'habit_nn':
            pass

        else:
            print('ERROR: self.selection_process is not habit_distance nor choice_model.')
            return False

        if 'time_retry_center' in overwrite_parameters:
            self.time_retry_center = overwrite_parameters['time_retry_center']
        else:
            self.time_retry_center = parameters['time_retry_center']
        if not isinstance(self.time_retry_center, int):
            print('ERROR: time_retry_center (%s) is not an int.' %
                str(self.time_retry_center))
            return False
        if self.time_retry_center < 0:
            print('ERROR: time_retry_center (%d) is negative.' %
                self.time_retry_center)
            return False
        if self.time_retry_center < self.data_handler.bin_size_dist:
            self.time_retry_center = self.data_handler.bin_size_dist

        if 'minimum_radius' in overwrite_parameters:
            self.minimum_radius = overwrite_parameters['minimum_radius']
        else:
            self.minimum_radius = parameters['minimum_radius']
        if not isinstance(self.minimum_radius, int):
            print('ERROR: minimum_radius (%s) is not an int.' %
                str(self.minimum_radius))
            return False
        if self.minimum_radius < 0:
            print('ERROR: minimum_radius (%d) is negative.' %
                self.minimum_radius)
            return False
        
        try:
            self.city_to_simulate
        except AttributeError:
            if 'city' in self.simulation_parameters:
                self.city_to_simulate = self.simulation_parameters['city']

            else:
                print("WARNING: No specific city to simulate selected, will simulate all cities")
                self.city_to_simulate = "all"
                
            if self.city_to_simulate == ["Amsterdam", "The Hague", "Rotterdam", "Utrecht"]:
                self.city_to_simulate == "all"
                
            if not isinstance(self.city_to_simulate, str):
                print('ERROR: city (%s) is not a string.' %
                    str(self.city_to_simulate))
                return False
            if self.city_to_simulate not in ["Amsterdam", "The Hague", "Rotterdam", "Utrecht", "all"]:
                print("ERROR: Name of city unknown, not any of these: Amsterdam, The Hague, Rotterdam, Utrecht, all.")
                return False
           
                
            
        if 'transform_parameters' in overwrite_parameters:
            transform_parameters = overwrite_parameters['transform_parameters']
        elif 'transform_parameters' in parameters:
            transform_parameters = parameters['transform_parameters']
        else:
            print('ERROR: No transform_parameters in parameters.')
            return False
        if 'prob_no_transform' not in transform_parameters:
            print('ERROR: No prob_no_transform defined in transform_parameters.')
            return False
        if not isinstance(transform_parameters['prob_no_transform'], float):
            print('ERROR: prob_no_transform in transform_parameters is not a float.')
            return False
        if 'prob_to_low_fev' not in transform_parameters and \
                transform_parameters['prob_no_transform'] != 1.0:
            print('ERROR: No prob_to_low_fev defined in transform_parameters.')
            return False
        if 'prob_to_high_fev' not in transform_parameters and \
                transform_parameters['prob_no_transform'] != 1.0:
            print('ERROR: No prob_to_high_fev defined in transform_parameters.')
            return False
        if not isinstance(transform_parameters['prob_to_low_fev'], float):
            print('ERROR: prob_to_low_fev in transform_parameters is not a float.')
            return False
        if not isinstance(transform_parameters['prob_to_high_fev'], float):
            print('ERROR: prob_to_high_fev in transform_parameters is not a float.')
            return False
        summed_probs = (transform_parameters['prob_no_transform'] +
            transform_parameters['prob_to_low_fev'] +
            transform_parameters['prob_to_high_fev'])
        if abs(1.0 - summed_probs) > 0.001:
            print('ERROR: The probabilities in transform parameters do not sum ' +
                'to 1.0. Namely prob_no_transform (%.2f) + ' %
                transform_parameters['prob_no_transform'] +
                'prob_to_low_fev (%.2f) + ' %
                transform_parameters['prob_to_low_fev'] +
                'prob_to_high_fev (%.2f) is %.2f and does not equal 1.0.' %
                (transform_parameters['prob_to_high_fev'], summed_probs))
            return False
        if agent_initialization not in (['load_and_use', 'create_and_use']) and \
            transform_parameters['prob_no_transform'] < 1.0:
            print('ERROR: Agent initalization method (%s) is not supported in ' %
                agent_initialization + 'combination with transforming agents.' +
                'If prob_no_transform (%.2f) is below 1.0, agent_initalization ' %
                transform_parameters['prob_no_transform'] + 'should be either ' +
                'load_and_use or create_and_use.')
            return False
        self.transform_parameters = transform_parameters

        if 'skip_high_fev_agents' in overwrite_parameters:
            skip_high_fev_agents = overwrite_parameters['skip_high_fev_agents']
        elif 'skip_high_fev_agents' in parameters:
            skip_high_fev_agents = parameters['skip_high_fev_agents']
        else:
            print('ERROR: No skip_high_fev_agents in parameters.')
            return False
        if not isinstance(skip_high_fev_agents, bool):
            print('ERROR: skip_high_fev_agents is not a boolean.')
            return False
        if 'skip_low_fev_agents' in overwrite_parameters:
            skip_low_fev_agents = overwrite_parameters['skip_low_fev_agents']
        elif 'skip_high_fev_agents' in parameters:
            skip_low_fev_agents = parameters['skip_low_fev_agents']
        else:
            print('ERROR: No skip_low_fev_agents in parameters.')
            return False
        if not isinstance(skip_low_fev_agents, bool):
            print('ERROR: skip_low_fev_agents is not a boolean.')
            return False
        if 'skip_phev_agents' in overwrite_parameters:
            skip_phev_agents = overwrite_parameters['skip_phev_agents']
        elif 'skip_phev_agents' in parameters:
            skip_phev_agents = parameters['skip_phev_agents']
        else:
            print('ERROR: No skip_phev_agents in parameters.')
            return False
        if not isinstance(skip_phev_agents, bool):
            print('ERROR: skip_phev_agents is not a boolean.')
            return False
        if 'skip_unknown_agents' in overwrite_parameters:
            skip_unknown_agents = overwrite_parameters['skip_unknown_agents']
        elif 'skip_unknown_agents' in parameters:
            skip_unknown_agents = parameters['skip_unknown_agents']
        else:
            print('ERROR: No skip_unknown_agents in parameters.')
            return False
        if not isinstance(skip_unknown_agents, bool):
            print('ERROR: skip_unknown_agents is not a boolean.')
            return False
        if skip_phev_agents and self.transform_parameters['prob_no_transform'] != 1.0:
            print('ERROR: skip_phev_agents is True while prob_no_transform is 1.0.')
            return False
        if (skip_low_fev_agents or skip_high_fev_agents) and agent_initialization != 'load_and_use':
            print('ERROR: skip_low_fev_agents and skip_high_fev_agents is only supported for load_and_use agent initialization.')
            return False
        self.skip_phev_agents = skip_phev_agents
        self.skip_low_fev_agents = skip_low_fev_agents
        self.skip_high_fev_agents = skip_high_fev_agents
        self.skip_unknown_agents = skip_unknown_agents

        return True

    def _load_centers_car2go(self, parameters):
        ''' This method loads the centers of the agent and updates the
            relevant attributes. Raises an exception when the agent has
            zero centers or when these centers are not used frequently in the
            test data.

        Args:
            parameters (Dict[str, Any]): The parameters of the agent that were
                given in the input file.

        Updates:
            centers_css
            centers_info
            original_centers_css
            walking_preparedness
        '''

        if 'ams' in self.ID:
            city = 'Amsterdam'
            center_loc = (4.896253, 52.372356)
        elif 'den' in self.ID:
            city = 'Den Haag'
            center_loc = (4.292700, 52.074588)
        elif 'rot' in self.ID:
            city = 'Rotterdam'
            center_loc = (4.483730, 51.929376)
        elif 'utr' in self.ID:
            city = 'Utrecht'
            center_loc = (5.120255, 52.093844)
        else:
            print('WARNING: unknown city, car2go in Amsterdam.')
            city = 'Amsterdam'
            center_loc = (4.896253, 52.372356)

        self.centers_css = {center_loc: {'habit': [], 'distance': []}}
        self.original_centers_css = {center_loc: {'habit': []}}

        self.walking_preparedness = 10000
        self.maximum_distance = 10000

        for center in self.centers_css:
            self.centers_css[center]['distance'] = \
                self.environment.get_nearby_css(self.centers_css, center,
                self.data_handler.training_data, self.walking_preparedness,
                city = city)
            self.centers_css[center]['habit'] = tuple(self.centers_css[center]['distance'].keys())

            self.original_centers_css[center]['habit'] = self.centers_css[center]['habit']

        self.centers_info = {}
        self.centers_info[center_loc] = {'nr_of_sessions_in_center': len(self.training_sessions), 'center_nr': 0}

    def _load_centers(self, parameters):
        ''' This method loads the centers of the agent and updates the
            relevant attributes. Raises an exception when the agent has
            zero centers or when these centers are not used frequently in the
            test data.

        Args:
            parameters (Dict[str, Any]): The parameters of the agent that were
                given in the input file.

        Updates:
            centers_css
            centers_info
            original_centers_css
            walking_preparedness
        '''
        if self.city_to_simulate == "all":
            self.centers_css, self.centers_info = self.data_handler.get_centers(
                self.training_sessions)
        else:
            self.centers_css, self.centers_info = self.data_handler.get_centers(
            self.training_sessions.loc[self.training_sessions['city'] == self.city_to_simulate])
            
        self.check_center_for_city()

        if len(self.centers_css.keys()) == 0:
            raise Exception('WARNING: Agent (%s) does not have any centers.' %
                self.ID)
        self.original_centers_css = {center:
            set(self.centers_css[center]['habit']) for center in self.centers_css}

        if not self._enough_sessions_in_test():
            raise Exception('WARNING: Too few sessions for agent ' +
                '(%s) in test data for centers.' %
                self.ID)

        self.walking_preparedness = self._get_walking_preparedness()

        for center in self.centers_css:
            self.centers_css[center]['distance'] = \
                self.environment.get_nearby_css(self.centers_css, center,
                self.data_handler.training_data, self.walking_preparedness, city = self.centers_info[center]['city'])
            
        

    def _enough_sessions_in_test(self):
        ''' This method checks if there are enough sessions in the test data
            for each of the agent's centers.

        Returns:
            (bool): True if the centers have enough sessions in the test data
                and False otherwise.
        '''

        for center in self.centers_css:
            sessions_in_center = 0
            for cs in self.centers_css[center]['habit']:
                sessions_in_center += len(self.test_sessions.loc[
                    self.test_sessions['location_key'] == cs])
            if sessions_in_center < self.data_handler.minimum_nr_sessions_center:
                return False
        return True

    def _get_walking_preparedness(self):
        ''' This method determines the walking preparedness of the agent. We
            define the walking preparedness as the maximum distance the agent
            has anywhere from a center to a charging station in that center
            multiplied with 1.1. If this results in something less than the
            default walking preparedness, then we set the agent's walking
            preparedness to this default value.

        Updates:
            maximum_distance (float)

        Returns:
            walking_preparedness (float): The walking preparedness for the agent.
        '''
        try:
            self.maximum_distance = max([self.environment.get_distance(cs, center)
                for center in self.centers_css
                for cs in self.centers_css[center]['habit']])
        except (ValueError, KeyError):
            self.maximum_distance = self.minimum_radius
            return self.minimum_radius

        if self.maximum_distance < self.minimum_radius:
            return self.minimum_radius
        else:
            return self.maximum_distance * 1.1
        
    def _get_disconnection_duration(self):
        ''' This method will update the disconnection duration based on the times between connections.
        
        Updates:
            training_sessions
        '''
        self.training_sessions = self.training_sessions.sort_values(by='start_connection', inplace = False)
        self.training_sessions.start_connection = self.training_sessions.start_connection.shift(-1)
        if len(self.training_sessions) == 0:
            self.training_sessions['disconnection_duration'] = []
        else:
            self.training_sessions['disconnection_duration'] = self.training_sessions.apply(lambda row:
                row.start_connection - row.end_connection, axis = 1)
        
        
    def _load_distributions_car2go(self):
        ''' This method loads the distributions of the agent based on its
            training data and centers.

        Updates:
            activity_patterns_training
            activity_patterns_test
            overall_activity_pattern
            disconnection_duration_dists
            connection_duration_dists
            arrival_dists
        '''
        path_to_file = 'data/car2go_behavior/car2go_behavior_dists.pkl'
        with open(path_to_file, 'rb') as data_file:
            [activity_pattern_norm, \
                arrival_dist_norm, connection_duration_dists, \
                disconnection_durations_for_car2go] = pandas.read_pickle(data_file, compression = None)


        self.activity_patterns_training = {center: activity_pattern_norm for center in self.centers_css}
        self.activity_patterns_test = {center: activity_pattern_norm for center in self.centers_css}

        for center in self.centers_css:
            self.activity_patterns_training[center] /= numpy.sum(
                self.activity_patterns_training[center])
            self.activity_patterns_test[center] /= numpy.sum(
                self.activity_patterns_test[center])

        # first_appearance = self.training_sessions['start_connection'].iloc[0].date()
        # last_appearance = self.training_sessions['start_connection'].iloc[-1].date()
        # number_of_days_active = (last_appearance - first_appearance).days
        # self.overall_activity_pattern = self.data_handler.get_activity_pattern(
        #     self.training_sessions) / number_of_days_active
        self.overall_activity_pattern = activity_pattern_norm

        self.disconnection_duration_dists = disconnection_durations_for_car2go

        self.connection_duration_dists = {center: connection_duration_dists for center in self.centers_css}

        self.arrival_dists = {center: arrival_dist_norm for center in self.centers_css}

    def _load_distributions(self):
        ''' This method loads the distributions of the agent based on its
            training data and centers.

        Updates:
            activity_patterns_training
            activity_patterns_test
            overall_activity_pattern
            disconnection_duration_dists
            connection_duration_dists
            arrival_dists
        '''

        self.activity_patterns_training = self.data_handler.get_activity_patterns_centers(
            self.training_sessions, self.centers_css)
        # print('got activity_patterns_training %s' % datetime.datetime.now())
        self.activity_patterns_test = \
            self.data_handler.get_activity_patterns_centers(self.test_sessions,
            self.centers_css)
        # print('got activity_patterns_test %s' % datetime.datetime.now())

        for center in self.centers_css:
            self.activity_patterns_training[center] /= numpy.sum(
                self.activity_patterns_training[center])
            # print('got activity_patterns_training for center %s' % datetime.datetime.now())

            self.activity_patterns_test[center] /= numpy.sum(
                self.activity_patterns_test[center])
            # print('got activity_patterns_test for center %s' % datetime.datetime.now())

        first_appearance = self.training_sessions['start_connection'].iloc[0].date()
        last_appearance = self.training_sessions['start_connection'].iloc[-1].date()
        number_of_days_active = (last_appearance - first_appearance).days
        self.overall_activity_pattern = self.data_handler.get_activity_pattern(
            self.training_sessions) / number_of_days_active

        # print('norm activity_patterns_training %s' % datetime.datetime.now())

        self.disconnection_duration_dists = self.data_handler.get_disconnection_duration_dists(
            self.training_sessions)
        # print('got disconnection_duration_dists %s' % datetime.datetime.now())

        self.connection_duration_dists = self.data_handler.get_connection_duration_dists(
            self.training_sessions, self.centers_css)
        # print('got connection_duration_dists %s' % datetime.datetime.now())

        self.arrival_dists = self.data_handler.get_arrival_dists(
            self.training_sessions,self.centers_css)
        # print('got arrival_dists at %s' % datetime.datetime.now())

    def _load_additional_attributes(self):
        ''' This method updates additional attributes of the agent. This
            includes updating the car type and battery category of the agent.

        Updates:
            car_type
            battery_category
        '''
        self.car_type = "Unknown"
        if 'car2go' in self.ID:
            self.car_type = 'FEV'
            self.battery_category = 'fev_low'
            return
        battery_size = numpy.percentile(self.training_sessions.kWh, 98)

        
        if self.ID in self.data_handler.car_type_data.index:
            self.car_type = self.data_handler.car_type_data.get_value(self.ID, 'TYPE')
        if self.car_type == "PHEV":
            self.battery_category = "phev"
        elif self.car_type == "Unknown":
            self.battery_category = "unknown"
        elif self.car_type == "FEV":
            if battery_size <= 33:
                self.battery_category = "fev_low"
            else:
                self.battery_category = "fev_high"

    def _reset_results(self):
        ''' This method sets the agent attributes containing info about the
            simulated sessions and sensors of the run.

        Updates:
            all_simulated_sessions
            sensors
        '''

        self.errors = {'MAE': {}, 'relMAE': {}}
        self.all_simulated_sessions = []
        self.sensors = {'disconnection_duration_mistake_counters': [],
            'connection_duration_mistake_counters': [],
            'selection_process_attempts': [], 'total_disconnections': [],
            'total_connections': [], 'total_selections': [],
            'occupied_counter': [],
            'run_counter': -1,
            'failed_charge_sessions':[],
            'cascaded_charge_sessions':[],
            're_attempts_sessions':[],
            're_attempts_poles':[],
            'cascaded_poles':[],
            'inconvenience':[],
            'failed_connection_attempt':[],
            'list_failed_charge_session':[]
                       }
        
        for center in self.centers_info:
            self.centers_info[center]['sensors'] = {'failed_charge_sessions': [],
                                           'cascaded_charge_sessions':[]}

    def _get_agent_data(self):
        ''' This method gets the data of the attributes of the agents that are
            stored in the agent database and returns those in a dictionary.

        Returns:
            Dict[str, Any]: A dictionary with the data of the agents. The keys
                in this dictionary are:
                    - centers_css
                    - centers_info
                    - original_centers_css
                    - maximum_distance
                    - activity_patterns_training
                    - activity_patterns_test
                    - overall_activity_pattern
                    - disconnection_duration_dists
                    - connection_duration_dists
                    - arrival_dists
        '''

        my_agent_data = {'centers_css': self.centers_css,
            'centers_info': self.centers_info,
            'original_centers_css': self.original_centers_css,
            'maximum_distance': self.maximum_distance,
            'walking_preparedness': self.walking_preparedness,
            'activity_patterns_training': self.activity_patterns_training,
            'activity_patterns_test': self.activity_patterns_test,
            'overall_activity_pattern': self.overall_activity_pattern,
            'disconnection_duration_dists': self.disconnection_duration_dists,
            'connection_duration_dists': self.connection_duration_dists,
            'arrival_dists': self.arrival_dists}
        if hasattr(self, 'car_type'):
            my_agent_data['car_type'] = self.car_type
        if hasattr(self, 'battery_category'):
            my_agent_data['battery_category'] = self.battery_category
        return my_agent_data

    def _set_agent_data(self, my_agent_data):
        ''' This method sets the attributes of the agent using the given agent
            data. Note that the car_type attribute is only loaded when the
            transform_parameters indicate that at least a fraction of the
            population should be transformed.

        Args:
            my_agent_data (Dict[str, Any]): The agent data as loaded from the
                agent database. The keys of this dictionary indicate the
                attribute of the agent and the values under these keys contain
                the data that should be stored for the agent attribute.

        Updates:
            centers_css
            centers_info
            original_centers_css
            maximum_distance
            activity_patterns_training
            activity_patterns_test
            overall_activity_pattern
            disconnection_duration_dists
            connection_duration_dists
            arrival_dists
            car_type
        '''

        self.centers_css = my_agent_data['centers_css']
        self.centers_info = my_agent_data['centers_info']
        self.original_centers_css = my_agent_data['original_centers_css']
        if 'maximum_distance' in my_agent_data:
            self.maximum_distance = my_agent_data['maximum_distance']
        self.walking_preparedness = my_agent_data['walking_preparedness']

        self.activity_patterns_training = my_agent_data['activity_patterns_training']
        self.activity_patterns_test = my_agent_data['activity_patterns_test']
        self.overall_activity_pattern = my_agent_data['overall_activity_pattern']
        self.disconnection_duration_dists = my_agent_data['disconnection_duration_dists']
        self.connection_duration_dists = my_agent_data['connection_duration_dists']
        self.arrival_dists = my_agent_data['arrival_dists']
        try:
            self.car_type = my_agent_data['car_type']
            self.battery_category = my_agent_data['battery_category']
        except KeyError:
            error = ('WARNING: The agent data of agent (' + self.ID +
                ') does not contain the required keys (car_type, battery_category).')
            raise Exception(error)
            
#         for center in self.centers_css:
#             for cs in self.centers_css[center]['distance']:
#                 cities_in_center.add(self.environment.css_info[cs]['city'])
            
    def check_center_for_city(self):
        ''' This method checks in which city the centers of the agent lie.
            The city will be added in the centers_info. 

        Updates:
            centers_info
        '''
        upper_limit_lat_Rot = 52.0
        lower_limit_lon_Rot = 4.3

        lower_limit_lat_Ams = 52.25

        upper_limit_lat_Utr = 52.25
        upper_limit_lon_Utr = 5.2
        lower_limit_lon_Utr = 4.95
        lower_limit_lat_Utr = 52.04

        lower_limit_lat_DH = 52.0
        upper_limit_lon_DH = 4.45

        for center in self.centers_info:
            
            if center[1] > lower_limit_lat_Ams:
                city = "Amsterdam"
                
            elif center[1] > lower_limit_lat_DH and center[0] < upper_limit_lon_DH:
                city = "Den Haag"

            elif center[1] < upper_limit_lat_Rot and center[0] > lower_limit_lon_Rot:
                city = "Rotterdam"

            elif center[1] > lower_limit_lat_Utr and center[1] < upper_limit_lat_Utr and \
                center[0] < upper_limit_lon_Utr and center[0] > lower_limit_lon_Utr:
                city = "Utrecht"

            else:
                city = "not in G4"


            self.centers_info[center]['city'] = city

    def _check_skip(self):
        ''' This method raises an exception whenever the agent should be skipped.
        '''
        if not hasattr(self, 'battery_category') and (self.skip_phev_agents or \
            self.skip_low_fev_agents or self.skip_high_fev_agents or self.skip_unknown_agents):
            raise Exception('ERROR: Cannot skip agent type without a known battery_category')
        if self.skip_phev_agents:
            if self.battery_category == 'phev':
                raise Exception('WARNING: Skipping phev agent.')
        if self.skip_low_fev_agents:
            if self.battery_category == 'fev_low':
                raise Exception('WARNING: Skipping low fev agent.')
        if self.skip_high_fev_agents:
            if self.battery_category == 'fev_high':
                raise Exception('WARNING: Skipping high fev agent.')
        if self.skip_unknown_agents:
            if self.battery_category == 'unknown':
                raise Exception('WARNING: Skipping unknown agent.')

    def _should_transform(self):
        ''' This method determines whether the agent should be transformed or
            not. If the agent has car_type PHEV it will be transformed with
            probability (1 - prob_no_transform). If the agent is to be
            transformed it will be transformed to low battery FEV or high
            battery FEV according to prob_to_low_fev and prob_to_high_fev.
            If the agent has another car_type than PHEV it will not be
            transformed.

        Returns:
            transform_to (str): Possible options are 'none' if the agent should
                not be transformed, 'low_fev' for transforming to a low battery
                FEV and 'high_fev' for transforming to a high battery FEV.
        '''

        transform_to = 'none'
        if self.transform_parameters['prob_no_transform'] < 1.0 and \
            self.car_type == 'PHEV':
            p = random.random()
            if p > self.transform_parameters['prob_no_transform']:
                if p > self.transform_parameters['prob_no_transform'] + self.transform_parameters['prob_to_low_fev']:
                    transform_to = 'high_fev'
                else:
                    transform_to = 'low_fev'
        return transform_to

    def _transform_agent(self, transform_to):
        ''' This method transforms the agent from PHEV to low or high battery
            FEV. It updates the transformed distributions in place.

        Args:
            transform_to (str): The type to transform to. Possible options are
                'none' if the agent should not be transformed, 'low_fev' for
                transforming to a low battery FEV and 'high_fev' for
                transforming to a high battery FEV.

        Updates:
            disconnection_duration_dists
            connection_duration_dists
        '''

        def _transform_distributions(original_dists, transforming_dist,
            size_new_dists):
            ''' This method transforms distributions according to the given
                transformation distribution, the mean target distributions and
                a transformation factor.

            Args:
                original_dists (List(Series)): The original distributions.
                transforming_dist (List[float]): The distribution that contains
                    information regarding the mean transformation.
                size_new_dists (int): The size (number of bins) that should be
                    transformed.

            Returns:
                (List(Series)): The transformed distributions.
            '''

            def _split_distributions(values_dist, splitting_dists):
                ''' This method splits the values of a distribution into
                    multiple distributions such that they are in the same
                    ratio as in the splitting_dists distributions.

                Args:
                    values_dist (List[float]): The distribution to split.
                    splitting_dists (List[Series]): The distributions that
                        indicate the ratios in which to split the distribution.

                Returns:
                    resulting_dists (List[Series]): A list of Series containing
                        the splitted distributions.
                '''

                def get_sum_bin(dists, bin_nr):
                    ''' Sums all the values in the indicated bin for a list of
                        distributions.

                    Args:
                        dists (List[float]): The distributions.
                        bin_nr (int): The bin number to sum.

                    Returns:
                        (float): The summed values in the bin.
                    '''

                    result = 0.0
                    for dist in dists:
                        if len(dist) <= bin_nr or isinstance(dist, pandas.DataFrame):
                            continue
                        result += float(list(dist)[bin_nr])
                    return result

                full_index = [self.data_handler.offset]
                while (len(full_index) < len(values_dist)):
                    full_index.append(full_index[-1] + datetime.timedelta(
                        minutes = self.data_handler.bin_size_dist))

                resulting_dists = [pandas.Series(data = [0.0]*len(values_dist),
                    index = full_index) for i in range(len(splitting_dists))]

                for b in range(len(values_dist)):
                    sum_bin = get_sum_bin(splitting_dists, b)
                    for d in range(len(splitting_dists)):
                        if isinstance(splitting_dists[d], pandas.DataFrame):
                            continue
                        if sum_bin == 0:
                            new_value = values_dist[b] / len(splitting_dists) #equal distribution over dists
                        else:
                            if len(splitting_dists[d].index) <= b:
                                continue
                            else:
                                # print(full_index)
                                # print(splitting_dists[b].index)
                                # print(len(splitting_dists), len(values_dist), len(full_index))
                                # try:
                                new_value = (splitting_dists[d][full_index[b]] / sum_bin * values_dist[b])
                                # except Exception as e:
                                #     print('is this splitting dist just too short?')
                                #     print('We want to access the %dth index, with value %s' %(b, full_index[b]))
                                #     print(splitting_dists[d])
                                #     raise e
                                # print(type(resulting_dists[d]))
                                resulting_dists[d].set_value(full_index[b], new_value)
                                # print('should be: %.4f' %new_value)
                                # print('now is:')
                                # print(resulting_dists[d].get_value(full_index[b]))
                    # print('sum bin %s' %d)
                    # print(sum(resulting_dists[d]))
                # print(sum(_sum_distributions(resulting_dists)))
                return resulting_dists

            def _sum_distributions(distributions):
                ''' This method sums the distributions and returns a single
                    distribution where each bin has a value of the sum of that bin
                    in each of the input distributions.

                Args:
                    distributions (List(Series)): A list of pandas Series containing
                        the distributions to sum.

                Returns:
                    total_distribution (Series): A single pandas Series containing
                        the sum of the input distributions. Elements in the
                        Series are int.
                '''

                total_distribution = pandas.Series()
                for dist in distributions:
                    if isinstance(dist, pandas.DataFrame):
                        continue
                    if total_distribution.empty:
                        total_distribution = dist
                    else:
                        total_distribution = total_distribution.radd(dist,
                            level=None, fill_value=0.0, axis=0)
                return total_distribution.astype(float)

            summed_original = _sum_distributions(original_dists)
            sum_ = float(sum(summed_original[:size_new_dists]))
            summed_original = summed_original[:size_new_dists] / sum_
            summed_original = list(summed_original)
            if len(summed_original) < size_new_dists:
                size_new_dists = len(summed_original)
            transformed_summed_dist = [max(0.0, summed_original[i] * transforming_dist[i])
                for i in range(size_new_dists)]

            return _split_distributions(transformed_summed_dist, original_dists)

        directory = 'data/simulation_pkls/'
        if transform_to == 'none':
            return
        elif transform_to == 'low_fev':
            file_detail = 'low'
        elif transform_to == 'high_fev':
            file_detail = 'high'
        else:
            raise Exception('ERROR: transform_to (%s) in transform_agent is invalid' % transform_to)

        file_binsize = '_bin' + str(self.data_handler.bin_size_dist)
        with open(directory + 'transformation_disconnection_phev_to_' + file_detail + file_binsize + '.pkl', 'rb') as f:
            disconnection_transform_dist = pickle.load(f)
        with open(directory + 'transformation_connection_phev_to_' + file_detail + file_binsize + '.pkl', 'rb') as f:
            connection_transform_dist = pickle.load(f)

        dc_size = int(60 * 24 / self.data_handler.bin_size_dist * 40)
        con_size = int(60 * 24 / self.data_handler.bin_size_dist * 5)

        for center in self.connection_duration_dists:
            self.connection_duration_dists[center] = _transform_distributions(
                self.connection_duration_dists[center],
                connection_transform_dist, con_size)
        self.disconnection_duration_dists = _transform_distributions(
            self.disconnection_duration_dists, disconnection_transform_dist, dc_size)
        if transform_to == 'high_fev':
            self.battery_category = 'fev_high'
        else:
            self.battery_category = 'fev_low'
        self.car_type = 'FEV'

    def set_initial_state(self):
        ''' This method (re)sets the agent's initial state.

        Updates:
            preferences
            sensors
            simulated_sessions
            history
            is_connected
            active_center
            active_cs
            time_next_activity
        '''

        self.preferences = {center: self.data_handler.get_preferences(
            self.training_sessions, self.centers_css[center]['habit'])
            for center in self.centers_css.keys()}

        if self.selection_process == 'strategy_model':
            if 'car2go' in self.ID:
                self.active_center = list(self.centers_css.keys())[0]
            else:
                centers = list(self.centers_css.keys())
                center_probs = [self.centers_info[center]['nr_of_sessions_in_center']
                    for center in centers]
                center_probs_norm = center_probs / numpy.sum(center_probs)

                self.active_center = centers[numpy.random.choice(range(len(centers)),
                p = center_probs_norm)]
                self.initialise_strategies()
                self.initialise_memory(current_time = self.start_date_simulation)            
        

        index_dist = self.data_handler.index_dist(self.start_date_simulation)
        connection_prob = self.overall_activity_pattern[index_dist]

        self.history = []
        self.simulated_sessions = {}
        self.sensors['run_counter'] += 1
        self.sensors['disconnection_duration_mistake_counters'].append(0)
        self.sensors['connection_duration_mistake_counters'].append(0)
        self.sensors['selection_process_attempts'].append([])
        self.sensors['total_disconnections'].append(0)
        self.sensors['total_connections'].append(0)
        self.sensors['total_selections'].append(0)
        self.sensors['occupied_counter'].append([])
        self.sensors['failed_charge_sessions'].append(0)
        self.sensors['cascaded_charge_sessions'].append(0)
        self.sensors['re_attempts_sessions'].append(0)
        self.sensors['re_attempts_poles'].append([])
        self.sensors['cascaded_poles'].append([])
        self.sensors['inconvenience'].append([])
        self.sensors['failed_connection_attempt'].append([])  
        self.sensors['list_failed_charge_session'].append([])
        
        for center in self.centers_info:
            self.centers_info[center]['sensors']['failed_charge_sessions'].append(0)
            self.centers_info[center]['sensors']['cascaded_charge_sessions'].append(0)

        self.is_connected = numpy.random.uniform() < connection_prob
        
        #self.start_date_agent = self.start_date_simulation
        
#         if self.start_date_agent == self.start_date_simulation:
#             pass
#         else:
#             self.time_next_activity = self.start_date_agent
#             self.active_cs = None
#             self.is_connected = False
            
        # for added centers just take the first center, since this is not based on the activities anymore.
        if 'added' in self.ID:
            self.active_center = list(self.centers_css.keys())[0]
            self.is_connected = False
            self.time_next_activity = self.start_date_agent
            self.active_cs = None
            
        else:
            if self.is_connected:
                index_dist = self.data_handler.index_dist(self.start_date_simulation)
                activity_patterns_items = self.activity_patterns_training.items()
                activity_probs = [dist[index_dist]
                    for _, dist in activity_patterns_items]
                if numpy.sum(activity_probs) == 0:
                    activity_probs_norm = [1/len(activity_probs)] * len(activity_probs)
                else:
                    activity_probs_norm = activity_probs / numpy.sum(activity_probs)
                centers = [center for center, _ in activity_patterns_items]

                self.active_center = centers[numpy.random.choice(range(len(centers)),
                        p = activity_probs_norm)]
                sampled_time = self.data_handler.sample(self.arrival_dists[self.active_center])


                activity_start_time = self.start_date_simulation + sampled_time - \
                    pandas.to_timedelta('1 day')

    #             print(self.centers_css, self.active_center, self.ID)
                ### Sometimes the active center is not in centers_css for some reason.
                if self.active_center not in self.centers_css:
                    self.active_center = list(self.centers_css.keys())[0]

                if not self.is_possible_to_connect(activity_start_time):
                    self.is_connected = False
                    self.sensors['selection_process_attempts'][self.sensors['run_counter']].append(True)
                    self.time_next_activity = activity_start_time + datetime.timedelta(
                        minutes = self.time_retry_center)
                    self._check_next_connection_time(activity_start_time, backwards = False,
                        next_center = self.active_center)
                    #print('WARNING: not possible to connect %s' % self.ID)
                    occupied = set()
                    for cs in self.centers_css[self.active_center]['distance']:
                        if cs in self.environment.css_info:
                            if self.environment.css_info[cs]['placement_date'] < activity_start_time and \
                            self.environment.css_info[cs]['amount_of_sockets'] > 0:
                                # print('2) adding %s' % str(tuple([cs, self.time_next_activity, tuple(self.environment.who_occupies(cs)), -1])))
                                occupied.add((cs, self.time_next_activity, tuple(self.environment.who_occupies(cs)), -1))
                                if len(self.environment.who_occupies(cs)) != len(set(self.environment.who_occupies(cs))):
                                    print('ERROR: agent twice in sockets')
                    self.sensors['occupied_counter'][self.sensors['run_counter']].append(occupied)
                else:
                    time_index = self.data_handler.index_dist(activity_start_time)
                    connection_duration = self.data_handler.sample(
                        self.connection_duration_dists[self.active_center][time_index])
                    self.time_next_activity = activity_start_time + connection_duration
                    self._check_next_disconnection_time(activity_start_time)
                    self.select_cs(activity_start_time)
                    self.update_history(activity_start_time)
                    self.environment.connect_agent(self.ID, self.active_cs)

            else:
                if 'car2go' in self.ID:
                    self.active_center = list(self.centers_css.keys())[0]
                    sampled_time = self.data_handler.sample(self.arrival_dists[self.active_center])

    #             elif 'added' in self.ID:
    #                 self.active_center = list(self.centers_css.keys())[0]
    #                 ## maybe this needs to be changed. Since added car now starts at start_date_simulation, but this should of 
    #                 ## course ba at start_date_agent
    #                 sampled_time = 0
    #                 print(self.ID)

                else:
                    self.active_cs = None
                    centers = list(self.centers_css.keys())
                    center_probs = [self.centers_info[center]['nr_of_sessions_in_center']
                        for center in centers]
                    center_probs_norm = center_probs / numpy.sum(center_probs)

                    self.active_center = centers[numpy.random.choice(range(len(centers)),
                    p = center_probs_norm)]
    #                 print(self.ID)

                    ### don't know why, but this work-around works, but should dive into arrival dists.
                    if self.active_center in self.arrival_dists:
                        sampled_time = self.data_handler.sample(self.arrival_dists[self.active_center])
                    else:
                        sampled_time = datetime.timedelta(minutes = 20)


                self.time_next_activity = pandas.to_datetime('%d%d%d' %
                    (self.start_date_simulation.year,
                    self.start_date_simulation.month, self.start_date_simulation.day),
                    format = '%Y%m%d') + sampled_time
            
        if self.selection_process == 'strategy_model':
            if not 'car2go' in self.ID:
                self.initialise_memory(current_time = self.start_date_simulation)
                self.initialise_strategies()

    def is_possible_to_connect(self, current_time):
        #NOTE: comments
        cs_in_environment = [cs for cs in self.centers_css[self.active_center]['distance'] \
                             if cs in self.environment.css_info]
        return len([cs for cs in cs_in_environment if not self.environment.is_occupied(cs) and \
            self.environment.css_info[cs]['placement_date'] < current_time and \
            self.environment.css_info[cs]['amount_of_sockets'] > 0])

    def select_cs(self, current_time):
        ''' This method selects which charging station the agent will use. The
            center of this location has already been specified (active_center).

        Args:
            time (DateTime): The current time of the simulation.

        Updates:
            active_cs
            time_next_activity (in case no charging station could be selected)
        '''
        self.sensors['total_selections'][self.sensors['run_counter']] += 1
        if 'car2go' in self.ID:
            #print('select CS for unhabitual user')
            self._select_using_uniform_distribution(current_time)
            
            #here the system goed wrong!
            
            
        elif self.selection_process == 'habit_distance':
            use_distance = True
            if numpy.random.uniform() <= self.selection_process_parameters['habit_probability']:
                self._select_using_habit(current_time)
                if self.active_cs:
                    use_distance = False

            if use_distance:
                self._select_using_distance(current_time)

            self.sensors['selection_process_attempts'][self.sensors['run_counter']].append(False)
            if self.active_cs:
                self._update_preferences()
        elif self.selection_process == 'choice_model':
            # time = datetime.datetime.now()
            self._select_using_choice_model(current_time)
            # print('choice model selection process took %s' % (datetime.datetime.now() - time))

        elif self.selection_process == 'strategy_model':
            # time = datetime.datetime.now()
            if 'car2go' in self.ID:
                self._select_using_uniform_distribution(current_time)
            else:
                self._select_using_strategy_model(current_time)


                
                
    
    def _select_using_strategy_model(self, current_time):
        cities_in_center = set()
        for cs in self.centers_css[self.active_center]['distance']:
            
            cities_in_center.add(self.environment.css_info[cs]['city'])
#         if len(cities_in_center) > 1:
#             print('WARNING: multiple cities in one center (%s)' % (cities_in_center))
#         city_of_center = cities_in_center.pop()
#         if city_of_center == 'Den Haag':
#             city_of_center = 'The Hague'

#         if not hasattr(self, 'home_zone_group'):
#             self._get_home_zone_group_and_permit_indicator()

        start_connection = current_time
        end_connection = self.time_next_activity

        if len(self.centers_css[self.active_center]['distance']) == 1:
            self.active_cs = list(self.centers_css[self.active_center]['distance'].keys())[0]

        possible_cs = []
        for cs in self.centers_css[self.active_center]['distance']:
            if cs in self.environment.css_info:
                if self.environment.css_info[cs]['placement_date'] < current_time and \
                self.environment.css_info[cs]['amount_of_sockets'] > 0:
                    possible_cs.append(cs)
        
        # chosen_pole = self.select_cs_based_on_strategy(current_strategy, possible_cs)
        current_strategy = self.choose_strategy()
        
        
        # if there are any cs in range, pick an available CP based on strategy
        # When this CP is full do again, until no CPs possible or available CP is selected.
        cascade_length_car = 0
        re_attempt_list = []
        cascade_list = []
        inconvenience = 0
        cascade = False
        re_attempt = False
        previous_pole = -1
        if len(possible_cs) > 0:
            pole_selection = False
            while pole_selection == False and len(possible_cs) > 0:
                chosen_pole = self.select_cs_based_on_strategy(current_strategy, possible_cs)     
                if self.environment.is_occupied(chosen_pole):
                    self.update_scores('occupied', pole = chosen_pole)
                    re_attempt = True
                    
                    
                    if self.environment.is_cascaded(chosen_pole, self.time_next_activity, check = True):
                        cascade = True
                        
                    # if first a CP is selected which is full there is a cascade
                    if cascade == True:
                        self.centers_info[self.active_center]['sensors']['cascaded_charge_sessions'][self.sensors['run_counter']] += 1
                        self.sensors['cascaded_charge_sessions'][self.sensors['run_counter']] += 1
                        self.environment.is_cascaded(chosen_pole, self.time_next_activity, check = False, cascade = True)
                        cascade_length_car += 1
                        
                    if re_attempt and not cascade:
                        re_attempt_list.append(chosen_pole)
                        self.sensors['re_attempts_sessions'][self.sensors['run_counter']] += 1
                    elif cascade:
                        cascade_list.append(chosen_pole)
                        
                    
                    if previous_pole != -1:
                        loc_previous_pole = (self.environment.css_info[previous_pole]['longitude'], \
                                            self.environment.css_info[previous_pole]['latitude'])
                        loc_chosen_pole = (self.environment.css_info[chosen_pole]['longitude'], \
                                            self.environment.css_info[chosen_pole]['latitude'])
                        inconvenience += self.environment.get_distance_general(loc_previous_pole, loc_chosen_pole)
 
                    
                    previous_pole = chosen_pole
                    possible_cs.remove(chosen_pole)
                else:
                    # select the CP
                    
                    if re_attempt:
                        self.environment.is_cascaded(chosen_pole, self.time_next_activity, check = False, cascade = False)
                    
                    inconvenience += self.centers_css[self.active_center]['distance'][chosen_pole] 
                    self.sensors['inconvenience'][self.sensors['run_counter']].append( \
                        inconvenience)
                    
                    self.active_cs = chosen_pole
                    pole_selection = True
                    self.update_scores('success')
            # if no possibilities, try again in a while
        if len(possible_cs) == 0:
            
            self.centers_info[self.active_center]['sensors']['failed_charge_sessions'][self.sensors['run_counter']] += 1
            self.sensors['failed_charge_sessions'][self.sensors['run_counter']] += 1
            self.active_cs = None
            
            inconvenience += 2*self.walking_preparedness
            self.sensors['inconvenience'][self.sensors['run_counter']].append( \
                        inconvenience)
            
        if re_attempt:
            self.sensors['re_attempts_poles'][self.sensors['run_counter']].append(re_attempt_list)
        if cascade:
            self.sensors['cascaded_poles'][self.sensors['run_counter']].append(cascade_list)
            # need to define self.age_compensation in the beginning
               
        #else:
            
            #if cascade == True:
            #    cascade_time_CP[pole] = max(cascade_time_CP[pole],next_charge[car]+chargetime)
            #    cascade_length = cascade_CP[pole] + cascade_length_car
            #elif cascade == False and all(cascade_time_CP[poles_in_same_place] < time): 
            #    cascade_CP[pole] = 0
            #    cascade_length = 0 
            #else: 
            #    cascade_length = 0
                
            #cascaded_sessions_list.append(cascade_length)
                    
    def initialise_memory(self, current_time, initial_memory = 20, age_compensation = 0.98):
        #initialise_memory
        self.memory = dict()
        for cs in self.centers_css[self.active_center]['distance']:
            # if initilization is with all cps added, after which, every time we check for new addition of cps.
            # this statement should be changed
            if cs in self.environment.css_info:
                if self.environment.css_info[cs]['placement_date'] < current_time:
                    self.memory[cs] = []

        self.initial_memory = initial_memory

        # initialise memory based on existing data
        for center in self.centers_css:
            
            used_cps_in_center = self.centers_css[center]['habit']
        
            # initialize empty memory 
            self.centers_css[center]['memory'] = {}
            
            last_training_sessions = self.training_sessions[self.training_sessions['location_key'].isin(used_cps_in_center)][-self.initial_memory:]
            
            # for all cs that have been used in the center count the times it has been charged at each pole to start memory
            for cs in self.centers_css[center]['habit']:
                if age_compensation == 1:
                    charge_sessions_at_cs = sum(last_training_sessions['location_key'] == cs)
                    self.centers_css[center]['memory'][cs] =  charge_sessions_at_cs

                    # if we make use of age_compensation, we first initialize all cs at zero
                else:
                    self.centers_css[center]['memory'][cs] = 0
            else:
                for index, cs in enumerate(reversed(list(last_training_sessions['location_key']))):
                    self.centers_css[center]['memory'][cs] += age_compensation**index
    
    def initialise_strategies(self):
        # name_strategies, e.g. research on what strategies
        self.strategies = {'max_score': [0], "prob_based": [0], "distance": [0]}

    def choose_strategy(self):
        # I want distance strategy:
        return "distance"
        if len(self.strategies) == 1:
            return numpy.random.choice(list(self.strategies.keys()))
        elif 'prob_based' in list(self.strategies.keys()):
            return 'prob_based'



    def select_cs_based_on_strategy(self, current_strategy = "prob_based", possible_cs = []):
        # make sure that we only select the cps that are not full for selecting a new cp.
        
        if current_strategy == "distance":
            if possible_cs != []:
                possible_poles = {}
                for pole in possible_cs:
                    possible_poles[pole] = self.centers_css[self.active_center]['distance'][pole]
                    
                return min(possible_poles, key=possible_poles.get)
            else:
                # print('No possible poles, to charge')
                return None
        else:
            
            if possible_cs != []:
                possible_poles = {}
                for pole in possible_cs:
                    if pole in self.centers_css[self.active_center]['memory']:
                        possible_poles[pole] = self.centers_css[self.active_center]['memory'][pole]

            if current_strategy == "max_score":
                if possible_poles != {}:
                    return max(possible_poles, key=possible_poles.get)
                else:
                    return numpy.random.choice(possible_cs)
                
            elif current_strategy == "prob_based":
                if possible_poles != {}:
                    norm = {}
                    cs_list = {}
                    if all(value > 0 for value in list(possible_poles.values())):
                        for key in possible_poles:
                            norm[key] = self.norm_prob(possible_poles, key)
                    else:
                        min_value = min(possible_poles.values())

                        for key in possible_poles:
                            cs_list[key] = possible_poles[key] - min_value

                        for key in cs_list:
                            norm[key] = self.norm_prob(cs_list, key)

                    if sum(list(norm.values())) == 1: 
                        return numpy.random.choice(list(norm.keys()), p = list(norm.values()))
                    else:
                        return numpy.random.choice(list(norm.keys()))
                else:
                    return numpy.random.choice(possible_cs)
        if possible_poles == {}:
            print('No possible poles, to charge')
            return None
            
            #norm = {}
            #cs_list = {}
            #if all(value > 0 for value in list(self.centers_css[self.active_center]['memory'].values())):
            #    for key in self.centers_css[self.active_center]['memory']:
            #        norm[key] = self.norm_prob(self.centers_css[self.active_center]['memory'], key)
            #else:
            #    min_value = min(self.centers_css[self.active_center]['memory'].values())
            #    
            #    for key in self.centers_css[self.active_center]['memory']:
            #        cs_list[key] = self.centers_css[self.active_center]['memory'][key] - min_value
            #    for key in cs_list:
            #        norm[key] = self.norm_prob(cs_list, key)
                
                
            #return numpy.random.choice(list(norm.keys()), p = list(norm.values()))

    def norm_prob(self, cs_list, key):
        if sum(cs_list.values()) != 0:
            return cs_list[key] / sum(cs_list.values())
        else: 
            return cs_list[key]
        
    def update_scores(self, success = "success", age_compensation = 0.98, pole = []):
        if pole == []:
            pole = self.active_cs
        # update memory of cps
        for cs in self.centers_css[self.active_center]['memory']:
            self.centers_css[self.active_center]['memory'][cs] *= age_compensation

        if pole in self.centers_css[self.active_center]['memory'].keys():
            if success == "success":
                self.centers_css[self.active_center]['memory'][pole] += 1
            else:
                self.centers_css[self.active_center]['memory'][pole] -= 1
        else:
            if success == "success":
                self.centers_css[self.active_center]['memory'][pole] = 1
            if success == "occupied":
                self.centers_css[self.active_center]['memory'][pole] = -1

        # update strategies of choosing a cp


    def _select_using_uniform_distribution(self, current_time):
        '''  Updated version  ''' 
        
        possible_cs = []
        for cs in self.centers_css[self.active_center]['distance']:
            if cs in self.environment.css_info:
                if self.environment.css_info[cs]['placement_date'] < current_time and \
                self.environment.css_info[cs]['amount_of_sockets'] > 0:
                    possible_cs.append(cs)
                                        
        free_css = [cs for cs in possible_cs if not self.environment.is_occupied(cs)]
        if not free_css:
            print('WARNING: No css are free, when in select_using_uniform_distribution. ' +
                 'This should never happen.')
            self.active_cs = None
            self.sensors['list_failed_charge_session'].append([self.ID, current_time])
            return

        self.active_cs = numpy.random.choice(free_css)
        
        
        
        

    def _select_using_choice_model(self, current_time):

        cities_in_center = set()
        for cs in self.centers_css[self.active_center]['distance']:
            cities_in_center.add(self.environment.css_info[cs]['city'])
        if len(cities_in_center) > 1:
            print('WARNING: multiple cities in one center (%s)' % (cities_in_center))
        city_of_center = cities_in_center.pop()
        if city_of_center == 'Den Haag':
            city_of_center = 'The Hague'

        if not hasattr(self, 'home_zone_group'):
            self._get_home_zone_group_and_permit_indicator()

        if not hasattr(self, 'card_provider'):
            self._get_card_provider()

        if not hasattr(self, 'all_charging_speeds'):
            self.all_charging_speeds = number_of_choices_per_parameter.get_all_charging_speeds()

        if not hasattr(self, 'all_prices'):
            self.all_prices = number_of_choices_per_parameter.get_all_prices()

        if not hasattr(self, 'connection_duration_converter'):
            self._get_connection_duration_converter()

        start_connection = current_time
        end_connection = self.time_next_activity

        if len(self.centers_css[self.active_center]['distance']) == 1:
            self.active_cs = list(self.centers_css[self.active_center]['distance'].keys())[0]

        explanatory_vars = {'cs': [], 'distance': [], 'charging_speed': [], 'charging_fee': [], 'parking_fee': []}

        for cs in self.centers_css[self.active_center]['distance']:
            if self.environment.css_info[cs]['placement_date'] < current_time:
                explanatory_vars['cs'].append(cs)
                # time1 = datetime.datetime.now()
                explanatory_vars['distance'].append(self.centers_css[self.active_center]['distance'][cs])
                # print('distance took %s' % (datetime.datetime.now() - time1))
                # time1 = datetime.datetime.now()
                explanatory_vars['charging_speed'].append(self._get_charging_speed(cs))
                # print('charging_speed took %s' % (datetime.datetime.now() - time1))
                # time1 = datetime.datetime.now()
                explanatory_vars['charging_fee'].append(self._get_charging_fee(cs, end_connection - start_connection))
                # print('charging_fee took %s' % (datetime.datetime.now() - time1))
                # time1 = datetime.datetime.now()
                if self.permit_indicator == 0:
                    explanatory_vars['parking_fee'].append(0)
                else:
                    explanatory_vars['parking_fee'].append(self._get_parking_fee(cs,
                        city_of_center, start_connection, end_connection))
                # print('parking_fee took %s' % (datetime.datetime.now() - time1))

        for var in explanatory_vars:
            if var != 'cs':
                if len(set(explanatory_vars[var])) > 1:
                    explanatory_vars[var] = [value / max(explanatory_vars[var])
                        for value in explanatory_vars[var]]
                else:
                    explanatory_vars[var] = [0 for value in explanatory_vars[var]]

        chosen = False
        occupied = set()
        occupied_data = set()
        choice_count = 0
        while not chosen:
            # if len(occupied) > 0:
            #     print('is possible to connect = %s' % self.is_possible_to_connect(current_time))
            #     print('all options = %s' % explanatory_vars['cs'])
            #     print('occupied = %s' % occupied)
            probs_per_cs = []
            choices = []
            for i in range(len(explanatory_vars['cs'])):
                if explanatory_vars['cs'][i] not in occupied:
                    choices.append(explanatory_vars['cs'][i])
                    a = self.selection_process_parameters[city_of_center]['intercept'] + \
                        self.selection_process_parameters[city_of_center]['distance'] * explanatory_vars['distance'][i] + \
                        self.selection_process_parameters[city_of_center]['charging_speed'] * explanatory_vars['charging_speed'][i] + \
                        self.selection_process_parameters[city_of_center]['charging_fee'] * explanatory_vars['charging_fee'][i] + \
                        self.selection_process_parameters[city_of_center]['parking_fee'] * explanatory_vars['parking_fee'][i]
                    probs_per_cs.append(numpy.exp(a) / (1 + numpy.exp(a)))
            probs_per_cs = probs_per_cs / numpy.sum(probs_per_cs)

            choice = numpy.random.choice(choices, p = probs_per_cs)
            choice_count += 1
            # if len(occupied) > 0:
            #     print('choice = %s' % choice)
            if not self.environment.is_occupied(choice):
                chosen = True
                # if len(occupied) > 0:
                #     print('not occupied, thus chosen')
            else:
                # print('1) adding %s' % str(tuple([choice, self.time_next_activity, tuple(self.environment.who_occupies(choice)), choice_count])))
                occupied_data.add((choice, self.time_next_activity, tuple(self.environment.who_occupies(choice)), choice_count))
                occupied.add(choice)
                # if len(occupied) > 0:
                #     print('occupied, try again: %s with probs %s' % (choices, probs_per_cs))

        self.active_cs = choice
        # if len(occupied) > 0:
        #     print('chosen = %s' % self.active_cs)
        #     print('occupied_counter = %d' % len(occupied))
        self.sensors['occupied_counter'][self.sensors['run_counter']].append(occupied_data)

        self._update_preferences()

 
    def _get_connection_duration_converter(self):
        training_sessions = self.training_sessions.copy(deep = True)
        training_sessions['connection_duration'] = training_sessions['end_connection'] - training_sessions['start_connection']
        training_sessions['connection_duration'] = training_sessions['connection_duration'].apply(lambda row: row / numpy.timedelta64(1, 'D'))
        training_sessions['Connection Duration (in Days)'] = training_sessions['connection_duration']
        training_sessions['kWh Charged'] = training_sessions['kWh']
        training_sessions['connection_duration'] = training_sessions['end_connection'] - \
            training_sessions['start_connection'] + self.data_handler.offset

        df =  training_sessions.set_index('connection_duration', drop = False, inplace = False)
        df = df.groupby(pandas.TimeGrouper(freq = '%dmin' % self.data_handler.bin_size_dist)).mean()
        df = df.fillna(method = 'pad')
        df.reset_index(level=0, inplace=True)
        self.connection_duration_converter = list(df['kWh'])

  
    def _select_using_habit(self, current_time):
        ''' Select a charging station in the active center using the preferences
            of the agent. active_cs is set to None if the selection is
            unsuccessful.

        Updates:
            active_cs
        '''
 
      
        possible_cs = list(self.preferences[self.active_center].keys())
        possible_cs = [cs for cs in possible_cs
                if self.environment.css_info[cs]['placement_date'] < current_time and self.environment.css_info[cs]['amount_of_sockets'] > 0]

        
        previous_CP =-1
        while len(possible_cs) > 0:
            
            
            
            prefs = [self.preferences[self.active_center][cs]
                for cs in possible_cs]
            normed_prefs = prefs / numpy.sum(prefs)
            cs = numpy.random.choice(possible_cs, p = normed_prefs)

            if self.environment.is_occupied(cs):
                
                self.sensors['failed_connection_attempt'].append([self.ID, current_time, cs, previous_CP])
                previous_CP=cs
                possible_cs.remove(cs)
            else:
                self.active_cs = cs
                self.sensors['list_failed_charge_session'].append([self.ID, current_time])
                
                return
           
        self.active_cs = None
        self.sensors['list_failed_charge_session'].append([self.ID, current_time])


        
    def _select_using_distance(self, current_time):
        ''' Select a charging station in the active center using the distances
            of the charging stations to this active center. active_cs is set
            to None if the selection is unsuccessful.

        Updates:
            active_cs
        '''

        css = self.centers_css[self.active_center]['distance']
#         if hasattr(self, 'time_next_activity'):
#             current_time = self.time_next_activity
#         else:
#             current_time = self.start_date_simulation
        
        possible_cs = [cs for cs in css
                if self.environment.css_info[cs]['placement_date'] < current_time and self.environment.css_info[cs]['amount_of_sockets'] > 0]
        
        while len(possible_cs) > 0:
            distances = [self.centers_css[self.active_center]['distance'][cs]
                for cs in possible_cs ]
            reversed_distances = [1.0 / dist if dist > 0 else 1 for dist in distances]
            norm_distances = reversed_distances / numpy.sum(reversed_distances)
            cs = numpy.random.choice(possible_cs, p = norm_distances)
        
            if self.environment.is_occupied(cs):
                self.sensors['failed_connection_attempt'].append([self.ID, current_time, cs])

                possible_cs.remove(cs)
            else:
                self.active_cs = cs
                return
                
                
        self.active_cs = None
        self.sensors['list_failed_charge_session'].append([self.ID, current_time])
            

        
        

    def _check_next_connection_time(self, previous_time_next_activity,
        backwards = True, next_center = None):
        ''' This method checks whether the next connection time is valid,
            meaning the agent can connect at this time. The agent can connect at
            a certain time when it has done so before (the probability of
            connecting at a center at the time next activity is not zero for all
            centers). Thus at least one center has a non-zero probability of
            starting to a connection at the time next activity. If the next
            connection time is not valid, it is changed so that it is valid.

        Args:
            previous_time_next_activity (DateTime): The time of the last activity.

        Kwargs:
            backwards (bool): If True, also look backwards to get a valid time
                next activity.
            next_center (Tuple[float, float]): Find the correct time next activity
                given the next center has already been chosen.

        Updates:
            time_next_activity
        '''

        minimal_disconnection_duration = pandas.to_timedelta('%dmin' %
            self.data_handler.bin_size_dist)
        index_dist = self.data_handler.index_dist(self.time_next_activity)
        arrival_dists = self.arrival_dists.items()
        dist_length = len(list(arrival_dists)[0][1])

        if next_center:
            arrival_dists_of_possible_center = [(center, dist) for (center,
                dist) in arrival_dists if center == next_center]
        else:
            arrival_dists_of_possible_center = arrival_dists

        for increase in range(dist_length):
            if numpy.sum([dist[(index_dist + increase) % len(dist)] for _,
                dist in arrival_dists_of_possible_center]):
                break

            if backwards:
                if numpy.sum([dist[(index_dist - increase) % len(dist)]
                    for _, dist in arrival_dists_of_possible_center]):
                    if self.time_next_activity + pandas.to_timedelta('%dmin'
                        % (self.data_handler.bin_size_dist * -1 * increase)) \
                        - previous_time_next_activity > \
                        minimal_disconnection_duration:
                        increase *= -1
                        break

        if not numpy.sum([dist[(index_dist + increase) % len(dist)] for _,
            dist in arrival_dists_of_possible_center]):
            print('WARNING: came out of for loop in ' +
                'check_next_connection_time with value %d for agent %s, but still not ' %
                (increase, self.ID) + 'correct disconnection duration.')
        if increase != 0:
            if self.start_date_simulation + self.warmup_period > \
                self.time_next_activity and not backwards:
                self.sensors['disconnection_duration_mistake_counters'][
                    self.sensors['run_counter']] += 1
            time_diff = (self.data_handler.bin_size_dist * increase)
            self.time_next_activity += pandas.to_timedelta('%dmin' % time_diff)

    def _update_preferences(self):
        ''' This method updates the preferences of the agent as well as adding
            the active charging station to the centers_css[center]['habit'] if
            this charging station is not yet there.

        Updates:
            preferences
            centers_css
        '''

        if self.active_cs in self.preferences[self.active_center]:
            self.preferences[self.active_center][self.active_cs] += 1
        else:
            self.preferences[self.active_center][self.active_cs] = 1
            self.centers_css[self.active_center]['habit'] += (self.active_cs,)

    def _check_next_disconnection_time(self, previous_time_next_activity):
        ''' This method checks whether the next disconnection time is valid,
            meaning the agent can disconnect at this time. The agent can
            disconnect at a certain time when it has done so before (the
            probability of disconnecting from a center at the time next activity
            is not zero for all centers). Thus at least one center has a
            non-zero probability of ending a connection at the time next
            activity. If the next disconnection time is not valid, it is changed
            so that it is valid.

        Args:
            previous_time_next_activity (DateTime): the time of the last activity.

        Updates:
            time_next_activity
        '''

        minimal_connection_duration = pandas.to_timedelta('%dmin' %
            self.data_handler.bin_size_dist)
        index = self.data_handler.index_dist(self.time_next_activity)
        increase = 0
        for increase in range(len(self.disconnection_duration_dists)):
            if numpy.sum(self.disconnection_duration_dists[(index - increase) %
                len(self.disconnection_duration_dists)]).all():
                if self.time_next_activity + pandas.to_timedelta('%dmin' %
                    (self.data_handler.bin_size_dist * -1 * increase)) - \
                    previous_time_next_activity > minimal_connection_duration:
                    increase *= -1
                    break
            if numpy.sum(self.disconnection_duration_dists[(index + increase) %
                len(self.disconnection_duration_dists)]).all():
                break
        if not numpy.sum(self.disconnection_duration_dists[(index + increase) %
            len(self.disconnection_duration_dists)]).all():
            print('WARNING: Came out of for-loop in check_next_disconnection_time ' +
                'with value %d but still not correct connection duration.' %
                increase)
            return

        if increase != 0:
            if self.start_date_simulation + self.warmup_period > \
                self.time_next_activity:
                self.sensors['connection_duration_mistake_counters'][
                    self.sensors['run_counter']] += 1
            time_diff = (self.data_handler.bin_size_dist * increase)
            self.time_next_activity += pandas.to_timedelta('%dmin' % time_diff)

    def get_next_connection_time_and_place(self, failed_session = False):
        ''' This method calculates the next connection time and place (center)
            of the agent. This information is stored in the attributes
            time_next_activity and active_center. Furthermore the decision
            process of this calculation is appended to the history of the agent.

        Updates:
            time_next_activity
            active_center
            history
        '''
        
        time = self.time_next_activity
        
        if failed_session:
            time_index = self.data_handler.index_dist(self.time_next_activity)
            connection_duration = self.data_handler.sample(
                self.connection_duration_dists[self.active_center][time_index])
            
            time_index = self.data_handler.index_dist(time)
            disconnection_duration = self.data_handler.sample(self.disconnection_duration_dists[time_index])
            
            self.time_next_activity = time + connection_duration + disconnection_duration
        
        else:
            self.sensors['total_disconnections'][self.sensors['run_counter']] += 1

            # t = datetime.datetime.now()
            index = self.data_handler.index_dist(time)
            disconnection_duration = \
                self.data_handler.sample(self.disconnection_duration_dists[index])
            # print('%s: 1a took %s' % (self.ID[:5], datetime.datetime.now() - t))

            # t = datetime.datetime.now()

            self.time_next_activity = time + disconnection_duration
            self._check_next_connection_time(time)
            # print('%s: 1b took %s' % (self.ID[:5], datetime.datetime.now() - t))
            # t = datetime.datetime.now()

        index_dist = self.data_handler.index_dist(self.time_next_activity)
        arrival_dists_items = self.arrival_dists.items()
        arrival_probs = [dist[index_dist] for _, dist in arrival_dists_items]
        centers = [center for center, _ in arrival_dists_items]
        # print('%s: 2 took %s' % (self.ID[:5], datetime.datetime.now() - t))

        # t = datetime.datetime.now()

        if numpy.sum(arrival_probs) == 0:
#             print('ERROR: check_next_connection_time failed. Assigning ' +
#                 'random active_center.')
            self.sensors['disconnection_duration_mistake_counters'][
                self.sensors['run_counter']] += 1
            arrival_probs_norm = arrival_probs
            self.active_center = centers[numpy.random.choice(range(len(centers)))]
        else:
            arrival_probs_norm = arrival_probs / numpy.sum(arrival_probs)
            self.active_center = centers[numpy.random.choice(range(len(centers)),
                p = arrival_probs_norm)]



#         if not failed_session:
#             self.history.append({'time': time, 'connected': False,
#                 'next_center': self.active_center, 
#                 'disconnection_duration': disconnection_duration,
#                 'corrected_disconnection_duration': self.time_next_activity - time,
#                 'arrival_probs': arrival_probs_norm,
#                 'time_next_activity': self.time_next_activity, 'centers': centers})


    def get_next_disconnection_time(self):
        ''' This method calculates the next disconnection time of the agent and
            stores this in the attribute time_next_activity. Furthermore the
            decision process of this calculation is appended to the history of
            the agent.

        Updates:
            time_next_activity
            history
        '''

        time = self.time_next_activity
        time_index = self.data_handler.index_dist(self.time_next_activity)
        connection_duration = self.data_handler.sample(
            self.connection_duration_dists[self.active_center][time_index])

        self.time_next_activity = time + connection_duration

        self.sensors['total_connections'][self.sensors['run_counter']] += 1

        self._check_next_disconnection_time(time)

    def update_history(self, start_time):
                    
        connection_duration = self.time_next_activity - start_time
        self.history.append({'time': start_time, 'connected': True,
            'connection_duration': connection_duration,
            'corrected_connection_duration': connection_duration,
            'time_next_activity': self.time_next_activity,
            'active_center': self.active_center, 'active_cs': self.active_cs})
        # print('self.active_cs = %s' % self.active_cs)

        if start_time > (self.start_date_simulation + self.warmup_period):
            dictionary_row = {'location_key': self.active_cs, 'ID': self.ID,
                'start_connection': start_time,
                'end_connection': self.time_next_activity,
                'kWh': self._get_kWh_session(connection_duration) 
                              #>>> this may be the only thing todo!
                             }
            dictionary_row.update(self.environment.css_info[self.active_cs])

            nr_of_simulated_sessions = len(self.simulated_sessions)
            self.simulated_sessions[nr_of_simulated_sessions + 1] = dictionary_row
        # print('done with history')
        
        

    def _get_kWh_session(self, connection_duration):
        '''

        #NOTE: fill this
        '''
        connection_duration_hours = connection_duration.total_seconds() / 60 / 60    
        if 'car2go' in self.ID: 
            kWh = min(connection_duration_hours*random.sample([3.7,11],1)[0],110)
        else:
        
            selected_sessions = self.training_sessions[['connection_time_hours','disconnection_duration','kWh']]


            bins=-1
            nr_records=0
            #first create sample set based on connection duration

            while nr_records<1:
                bins+=1

                records_filtered_for_sample = selected_sessions.query('connection_time_hours<' + \
                                                                     str(connection_duration_hours+1+bins) + \
                                'and connection_time_hours>'+str(connection_duration_hours-1-bins))

                nr_records = records_filtered_for_sample['connection_time_hours'].count()            

            #from this set the most likeli disconnection duration may be sampled
            kWh = numpy.random.choice(records_filtered_for_sample['kWh']) 
    
               
        return kWh

    
    
    
    def save_simulation_data(self):
        ''' This method saves the simulation data for the last simulation run
            of the agent.

        Updates:
            all_simulated_sessions
        '''

        self.all_simulated_sessions.append(pandas.DataFrame.from_dict(
            self.simulated_sessions, orient = 'index'))

    def validate(self, method = 'relMAE', with_IDs = False):
        ''' This method returns the error of the centers of the agent.

        Kwargs:
            method (str): Method of validation. 'MAE' for Mean Absolute Error
                or 'relMAE' for relative Mean Absolute Error. Default 'relMAE'.
            with_IDs (boolean): Indicates whether the errors should be stored
                as float, or as a tuple containing the agent ID and the error.

        Returns:
            (Dict[str, float]): Contains the errors of 'training' and 'test',
                which are the keys of the dictionary. The errors are the values
                at those keys and indicate the mean of the specified error
                measure over the centers of the agent.
        '''

        if not hasattr(self, 'simulated_activity_patterns'):
            self._create_activity_patterns()

        if method != 'MAE' and method != 'relMAE':
            print('WARNING: Kwarg \'method\' (%s) in '% method +
                'agent.validate() was specified incorrectly. It should be ' +
                'MAE or relMAE. Now using the default (relMAE).')
            method = 'relMAE'

        if method == 'MAE':
            if not hasattr(self, 'MAEs'):
                self._errors_centers(method = 'MAE')

            sum_MAE_test = 0
            sum_MAE_training = 0
            for center in self.centers_css:
                sum_MAE_training += self.errors['MAEs'][center]['training']
                sum_MAE_test += self.errors['MAEs'][center]['test']
            if with_IDs:
                return {'training': (self.ID, sum_MAE_training / len(self.centers_css)),
                    'test': (self.ID, sum_MAE_test / len(self.centers_css))}
            else:
                return {'training': sum_MAE_training / len(self.centers_css),
                    'test': sum_MAE_test / len(self.centers_css)}

        if method == 'relMAE':
            if len(self.errors['relMAE']) == 0:
                self._errors_centers(method = 'relMAE')

            sum_relMAEs_test = 0
            sum_relMAEs_training= 0
            for center in self.centers_css:
                sum_relMAEs_training += self.errors['relMAE'][center]['training']
                sum_relMAEs_test += self.errors['relMAE'][center]['test']
            if with_IDs:
                return {'training': (self.ID, sum_relMAEs_training / len(self.centers_css)),
                    'test': (self.ID, sum_relMAEs_test / len(self.centers_css))}
            else:
                return {'training': sum_relMAEs_training / len(self.centers_css),
                    'test': sum_relMAEs_test / len(self.centers_css)}

    def _create_activity_patterns(self):
        ''' This method creates the activity patterns of the agent for the
            training data, the test data and the simulated data.

        Updates:
            simulated_activity_patterns
            activity_patterns_training
            activity_patterns_test
        '''

        activity_patterns_per_run = \
            [self.data_handler.get_activity_patterns_centers(sessions_one_run,
            self.centers_css) for sessions_one_run in self.all_simulated_sessions]
        self.simulated_activity_patterns = \
            self.data_handler.convert_activity_pattern_data(activity_patterns_per_run)

        mean_simulated_activity_patterns = {}
        for center in self.centers_css:
            self.activity_patterns_training[center] /= numpy.sum(
                self.activity_patterns_training[center])
            self.activity_patterns_test[center] /= numpy.sum(
                self.activity_patterns_test[center])

            mean_activity_pattern = self.simulated_activity_patterns[center][0]
            for activity_pattern in self.simulated_activity_patterns[center][1:]:
                mean_activity_pattern += activity_pattern
            mean_activity_pattern /= numpy.sum(mean_activity_pattern)
            mean_simulated_activity_patterns[center] = mean_activity_pattern

        self.simulated_activity_patterns = mean_simulated_activity_patterns

    def _errors_centers(self, method = 'relMAE'):
        ''' This method calculates the error of each bin comparing the simulated
            sessions and the training set as well as the simulated sessions and
            the test set. For each center the errors of the bins are averaged.
            The result is stored in the errors attribute. Two types of error
            calculation are supported: the Mean Abolute Error (MEA) and the
            relative MAE.

        Kwargs:
            method (str): Method of validation. 'MAE' for Mean Absolute Error
                or 'relMAE' for relative Mean Absolute Error. Default 'relMAE'.

        Updates:
            errors
        '''

        for center in self.centers_css:
            self.errors[method][center] = {'training': self.data_handler.get_error(
                self.activity_patterns_training[center],
                self.simulated_activity_patterns[center], method = method),
                'test': self.data_handler.get_error(
                self.activity_patterns_test[center],
                self.simulated_activity_patterns[center], method = method)}

    def visualize(self):
        ''' This method visualizes the agent by creating and displaying an
            ipywidget containing this visualization.
        '''

        if len(self.all_simulated_sessions[-1]) == 0:
            print('ERROR: Agent (%s) has no simulated sessions and ' % self.ID +
                'therefore will not be visualized.')
            return

        if not hasattr(self, 'simulated_activity_patterns'):
            self._create_activity_patterns()

        center_info, cs_info = self._get_visualization_data()

        x_min = min(min(cs_info['lons']), min(center_info['lons']))
        x_max = max(max(cs_info['lons']), max(center_info['lons']))
        y_min = min(min(cs_info['lats']), min(center_info['lats']))
        y_max = max(max(cs_info['lats']), max(center_info['lats']))
        x_diff = (x_max - x_min) / 10.
        y_diff = (y_max - y_min) / 10.
        x_sc = bqplot.LinearScale(min = x_min - max(x_diff, 0.001),
            max = x_max + max(x_diff, 0.001))
        y_sc = bqplot.LinearScale(min = y_min - max(y_diff, 0.001),
            max = y_max + max(y_diff, 0.001))

        ax_x = bqplot.Axis(label='Longitude', tick_format='0.2f', scale=x_sc,
            grid_lines='solid')
        ax_y = bqplot.Axis(label='Latitude', tick_format='0.2f', scale=y_sc,
            orientation='vertical', side='left', grid_lines='solid')

        c_sc_cs = bqplot.OrdinalColorScale(colors = bqplot.CATEGORY10)
        c_sc_center = bqplot.OrdinalColorScale(colors = bqplot.CATEGORY10)

        size_sc_center = bqplot.LinearScale(min = min(center_info['sizes_simulated']),
            max = max(center_info['sizes_simulated']))
        size_sc_cs = bqplot.LinearScale(min = float(min(cs_info['sizes_simulated'])),
            max = float(max(cs_info['sizes_simulated'])))

        sc_opacity = bqplot.LinearScale()

        tt_css = bqplot.Tooltip(fields = ['name', 'size', 'rotation', 'skew'],
            labels = ['CS Name', 'Mean nr of simulated sessions',
            'Percentage chosen (real)', 'Percentage chosen (simulated)'],
            formats = ['', '.1f', '.2f', '.2f'])

        agent_label = ipywidgets.HTML(value = '<font size=\'3\'>%s</font><br><br>' %
            self.ID)

        if len(self.errors['MAE']) == 0:
            self._errors_centers(method = 'MAE')
        if len(self.errors['relMAE']) == 0:
            self._errors_centers(method = 'relMAE')

        center = center_info['location'][0]

        activity_pattern_training = self._get_barplot(
            self.activity_patterns_training[center],
            self.simulated_activity_patterns[center],
            datetime.datetime(self.data_handler.offset.year,
            self.data_handler.offset.month, self.data_handler.offset.day, 23, 59),
            'Training')
        activity_pattern_test = self._get_barplot(self.activity_patterns_test[center],
            self.simulated_activity_patterns[center],
            datetime.datetime(self.data_handler.offset.year,
            self.data_handler.offset.month, self.data_handler.offset.day, 23, 59),
            'Test')

        validation = ipywidgets.HTML('')
        legend = ipywidgets.HTML('Loading...')
        tt_centers = ipywidgets.VBox([ipywidgets.HBox([legend]),
            activity_pattern_training, activity_pattern_test, validation])
        unselected_opacity = 0.3

        cs_scatter = bqplot.Scatter(x = cs_info['lons'], y = cs_info['lats'],
            color = cs_info['center_nrs'], names = cs_info['names'],
            extra = cs_info['sizes'], size = cs_info['sizes_simulated'],
            rotation = cs_info['percentage_chosen_real'],
            skew = cs_info['percentage_chosen_simulation'], display_names = False,
            scales = {'x': x_sc, 'y': y_sc, 'color': c_sc_cs, 'size': size_sc_cs,
            'opacity': sc_opacity}, default_size = 1000, tooltip = tt_css,
            tooltip_style = {'opacity': 1}, animate = True, stroke = 'Black',
            marker = 'circle')

        center_scatter = bqplot.Scatter(x = center_info['lons'],
            y = center_info['lats'], color = center_info['center_nrs'],
            location = center_info['location'], size = center_info['sizes'],
            display_names = False, scales = {'x': x_sc, 'y': y_sc,
            'color': c_sc_center, 'size': size_sc_center, 'opacity': sc_opacity},
            default_size = 300, tooltip = tt_centers, tooltip_style = {'opacity': 1},
            animate = True, stroke = 'Black', marker = 'cross')

        center_labels = bqplot.Label(x = [x for (x, _) in center_info['location']],
             y = [y + max(y_diff, 0.00015) for (_, y) in center_info['location']],
             font_size = 30, font_weight='bolder',
             colors = bqplot.CATEGORY10, enable_move = True,
             text = center_info['center_nrs'], scales = {'x': x_sc, 'y': y_sc})

        def hover_center_dists(change):
            if change.new is not None:
                center = center_info['location'][change.new]

                barchart_dist = [(activity_pattern_training,
                    self.activity_patterns_training[center]),
                    (activity_pattern_test, self.activity_patterns_test[center])]

                for barchart, dist in barchart_dist:
                    barchart.marks[0].y = dist
                    barchart.marks[1].y = self.simulated_activity_patterns[center]

                    barchart.marks[0].scales['y'].max = float(max(dist)) * 1.1
                    barchart.marks[1].scales['y'].max = float(max(dist)) * 1.1

                    barchart.marks[1].scales['color'].colors = [bqplot.CATEGORY10[change.new]]
                    barchart.marks[1].color = [change.new] * len(barchart.marks[1].color)
                    barchart.marks[1].stroke = bqplot.CATEGORY10[change.new]

                    barchart.marks[2].y = [barchart.scale_y.max / 2]

                center_index = center_info['location'].index(center)
                legend_color = bqplot.CATEGORY10[change.new][1:]
                legend_center_size = center_info['sizes'][center_index]

                legend.value = '<font style=\'background-color:#eeeeee;\' ' + \
                    'size = 3>Legend: real, <font color=\'#%s\'>' % legend_color + \
                    'simulated</font></font>'

                validation.value = '<table size = 3 style=\'background-color:' + \
                    '#eeeeee; width: 100%; text-align: center;\'> <tr> <td>' + \
                    '</td><td>Training</td> <td>Test</td> <td>Simulation</td></tr>' + \
                    '<tr> <td>number of sessions</td><td>%d</td>' % \
                    center_info['sizes'][center_index] + \
                    '<td>%d</td> <td>%.2f +- %.2f</td></tr>' % \
                    (center_info['sizes_test'][center_index],
                    center_info['sizes_simulated'][center_index],
                    center_info['sizes_simulated_std'][center_index]) + \
                    '<tr> <td>relMAE</td><td>%.3f</td>' % \
                    self.errors['relMAE'][center]['training'] + \
                    '<td>%.3f</td> <td>-</td></tr>' % \
                    self.errors['relMAE'][center]['test'] + \
                    '<tr> <td>sessions/month</td>' + \
                    '<td>%.2f</td> <td>%.2f</td> <td>%.2f +- %.2f</td></tr></table>' % \
                    (center_info['training_sessions_p_month'][center_index],
                    center_info['test_sessions_p_month'][center_index],
                    center_info['simulated_sessions_p_month'][center_index][0],
                    center_info['simulated_sessions_p_month'][center_index][1] * \
                    1.96 / numpy.sqrt(len(self.all_simulated_sessions)))

        center_scatter.observe(hover_center_dists, 'hovered_point')

        start_time_secs = self.history[0]['time'].to_datetime64().astype(int)
        end_time_secs = self.history[-1]['time'].to_datetime64().astype(int)
        time_stepsize = int(pandas.to_timedelta('%dmin' %
            self.data_handler.bin_size_dist).to_timedelta64().astype(int))

        fig = bqplot.Figure(marks=[cs_scatter, center_scatter, center_labels],
            axes = [ax_x, ax_y], width = 700, title_style = {'font-size': '25px'},
            title = str(pandas.to_datetime(start_time_secs)),
            fig_margin = {'top': 55, 'bottom': 30, 'left': 50, 'right': 20})

        date_slider = ipywidgets.IntSlider(min = start_time_secs, max = end_time_secs,
            description='DateTime', value = start_time_secs, step = time_stepsize,
            readout = False, width = '50em')

        info = ipywidgets.HTML(value = '')

        def date_changed(change):
            info_str = ['<br><br><br><font size=\'6\'>Information</font><br>'+
                '<font size=\'4\'>']

            if date_slider.value >= self.history[-1]['time'].to_datetime64().astype(int):
                point_in_history = -1
                info_str.append('<b>Connected:</b><br>%s<br><b>No more activities</b>.' %
                    self.history[point_in_history]['connected'])
            else:
                for i in range(len(self.history)):
                    if date_slider.value >= self.history[i]['time'].to_datetime64().astype(int) and \
                       date_slider.value < self.history[i + 1]['time'].to_datetime64().astype(int):
                        point_in_history = i

                info_str.append('<b>Connected:</b><br>%s<br><b>Time next activity:</b><br>%s.' %
                    (self.history[point_in_history]['connected'],
                     self.history[point_in_history + 1]['time']))

            info_history = self.history[point_in_history]

            if not info_history['connected']:
                if self.history[point_in_history]['corrected_disconnection_duration'] != \
                   self.history[point_in_history]['disconnection_duration']:
                    info_str.append('<br>(disconnection duration was wrong)<br>(corrected: %s)<br>' %
                        self.history[point_in_history]['corrected_disconnection_duration'])

                center_scatter.default_opacities = [unselected_opacity] * len(center_info['location'])
                cs_scatter.default_opacities = [unselected_opacity] * len(cs_info['names'])

                ordered_arrival_probs = []
                for i in range(len(info_history['arrival_probs'])):
                    center_location = center_info['location'][center_info['center_nrs'].index(i)]
                    arrival_prob = info_history['arrival_probs'][info_history['centers'].index(center_location)]
                    ordered_arrival_probs.append(arrival_prob)

                next_center = center_info['center_nrs'][center_info['location'].index(info_history['next_center'])]
                info_str.append(('<br><br><b>Decision time:</b><br>%s<br><b>'
                    'Disconnection duration:</b><br>%s<br><b>'
                    'Ordered arrival probs:</b><br>%s<br><b>'
                    'Next center:</b><br>%d' % (info_history['time'],
                    info_history['disconnection_duration'],
                    ', '.join(['%.3f' % prob for prob in ordered_arrival_probs]),
                     next_center)))
            else:
                opacities = [unselected_opacity] * len(center_info['location'])
                opacities[center_info['location'].index(info_history['active_center'])] = 1
                center_scatter.default_opacities = opacities

                opacities = [unselected_opacity] * len(cs_info['names'])
                opacities[cs_info['names'].index(info_history['active_cs'])] = 1
                cs_scatter.default_opacities = opacities

                if self.history[point_in_history]['corrected_connection_duration'] != \
                   self.history[point_in_history]['connection_duration']:
                    info_str.append('<br>(connection duration was wrong)<br>(corrected: %s)<br>' %
                        self.history[point_in_history]['corrected_connection_duration'])

                active_center = center_info['center_nrs'][center_info['location'].index(info_history['active_center'])]
                info_str.append(('<br><br><b>Decision time:</b><br>%s<br><b>'
                    'Connection duration:</b><br>%s<br><b>Active'
                    ' center:</b><br>%d<br><b>Active cs:</b><br>%s' %
                    (info_history['time'], info_history['connection_duration'],
                    active_center, info_history['active_cs'])))

            info_str.append('</font>')
            info.value = ''.join(info_str)
            info.width = '20em'
            fig.title = 'Time = %s' % str(pandas.to_datetime(date_slider.value))

        date_slider.observe(date_changed, 'value')
        play_button = ipywidgets.Play(min = start_time_secs, max = end_time_secs,
            step = time_stepsize, interval = 500)
        ipywidgets.jslink((play_button, 'value'), (date_slider, 'value'))
        IPython.display.display(ipywidgets.VBox(
            [ipywidgets.HBox([play_button, date_slider]),
            ipywidgets.HBox([fig, info]), agent_label]))

        date_changed(None)
        sys.stdout.flush()
        return fig

    def _get_visualization_data(self):
        ''' This method generates the center and charging station information
            needed for the visualization of the agent..

        Returns:
            center_info (Dict[str, List[Any]]): Information about centers, with as
                keys 'names', 'lons', 'lats', 'center_nrs' and 'sizes'. The
                values are list containing the values of each of the centers.
            cs_info (Dict[str, List[Any]]): Information about charging stations,
                with as keys 'names', 'lons', 'lats', 'center_nrs', 'sizes',
                'percentage_chosen_real' and 'percentage_chosen_simulation'.
                The values are list containing the values of each of the
                charging stations.
        '''

        cs_info = {'names': [], 'lons': [], 'lats': [], 'center_nrs': [],
            'sizes': [], 'sizes_simulated': [], 'percentage_chosen_real': [],
            'percentage_chosen_simulation': []}
        center_info = {'location': [], 'lons': [], 'lats': [], 'center_nrs': [],
            'sizes_test': [], 'sizes_simulated': [], 'sizes': [],
            'training_sessions_p_month': [], 'test_sessions_p_month': [],
            'simulated_sessions_p_month': [], 'sizes_simulated_std': []}

        for center in self.centers_css:
            center_info['location'] += [center]
            center_info['lons'] += [center[0]]
            center_info['lats'] += [center[1]]
            center_nr = self.centers_info[center]['center_nr']
            center_info['center_nrs'] += [center_nr]
            center_info['sizes'] += [self.centers_info[center]['nr_of_sessions_in_center']]

            training_sessions = self.training_sessions.loc[
                self.training_sessions.location_key.isin(self.centers_css[center]['habit'])]
            training_sessions = training_sessions.sort_values(by = 'start_connection')
            first_appearance_training = training_sessions['start_connection'].iloc[0].date()
            last_appearance_training = training_sessions['start_connection'].iloc[-1].date()
            center_info['training_sessions_p_month'] += [len(training_sessions) / \
                (last_appearance_training - first_appearance_training).days * 30]

            test_sessions = self.test_sessions.loc[
                self.test_sessions.location_key.isin(self.centers_css[center]['habit'])]
            test_sessions = test_sessions.sort_values(by = 'start_connection')
            first_appearance_test = test_sessions['start_connection'].iloc[0].date()
            last_appearance_test = test_sessions['start_connection'].iloc[-1].date()
            center_info['test_sessions_p_month'] += [len(test_sessions) /
                (last_appearance_test - first_appearance_test).days * 30]
            center_info['sizes_test'] += [len(test_sessions)]

            all_simulated_sessions_p_month = []
            all_simulated_sizes = []
            for i in range(len(self.all_simulated_sessions)):
                simulated_sessions = self.all_simulated_sessions[i].loc[
                    self.all_simulated_sessions[i].location_key.isin(self.centers_css[center]['habit'])]
                simulated_sessions = simulated_sessions.sort_values(by = 'start_connection')
                first_appearance_simulated = simulated_sessions['start_connection'].iloc[0].date()
                last_appearance_simulated = simulated_sessions['start_connection'].iloc[-1].date()
                all_simulated_sizes.append(len(simulated_sessions))
                all_simulated_sessions_p_month.append(len(simulated_sessions) /
                    (last_appearance_simulated - first_appearance_simulated).days * 30)
            center_info['simulated_sessions_p_month'] += \
                [[numpy.mean(all_simulated_sessions_p_month),
                numpy.std(all_simulated_sessions_p_month)]]
            center_info['sizes_simulated'] += [numpy.mean(all_simulated_sizes)]
            center_info['sizes_simulated_std'] += [numpy.std(all_simulated_sizes)]

            nr_of_test_sessions_in_center = 0
            nr_of_simulated_sessions_in_center = 0
            simulated_session_counts = {}
            for cs in self.centers_css[center]['habit']:
                cs_info['names'] += [cs]
                cs_info['lons'] += [self.environment.css_info[cs]['longitude']]
                cs_info['lats'] += [self.environment.css_info[cs]['latitude']]
                cs_info['center_nrs'] += [center_nr]

                if cs in self.original_centers_css[center]:
                    nr_of_real_sessions_in_cs = \
                        len(self.training_sessions.loc[self.training_sessions.location_key == cs])
                else:
                    nr_of_real_sessions_in_cs = 0

                cs_info['sizes'] += [nr_of_real_sessions_in_cs]
                cs_info['percentage_chosen_real'] += [nr_of_real_sessions_in_cs /
                    self.centers_info[center]['nr_of_sessions_in_center']]

                nr_of_test_sessions_in_center += \
                    len(self.test_sessions.loc[self.test_sessions.location_key == cs])

                all_simulated_sessions_in_cs = [len(
                    self.all_simulated_sessions[i].loc[self.all_simulated_sessions[i].location_key == cs])
                    for i in range(len(self.all_simulated_sessions))]
                nr_of_simulated_sessions_in_cs = numpy.mean(all_simulated_sessions_in_cs)

                nr_of_simulated_sessions_in_center += nr_of_simulated_sessions_in_cs
                simulated_session_counts[cs] = nr_of_simulated_sessions_in_cs
                cs_info['sizes_simulated'] += [nr_of_simulated_sessions_in_cs]

            for cs in self.centers_css[center]['habit']:
                if nr_of_simulated_sessions_in_center == 0:
                    cs_info['percentage_chosen_simulation'] += [0]
                else:
                    cs_info['percentage_chosen_simulation'] += \
                        [simulated_session_counts[cs] /
                        nr_of_simulated_sessions_in_center]

        return center_info, cs_info

    def _get_barplot(self, dist_real, dist_simulated, max_datetime, title):
        ''' This method creates a bar plot of the given distribution.

        Args:
            dist (DataFrame): Contains the distribution.
            max_datetime (DateTime): The maximum value of the x-axis.
            title (str): The name of the distribution, which will be plotted
                as a label in the figure. Note that if the title is equal to
                'Disconnection duration dist', the figure will be adjusted
                (different margins, size, title location and font size).

        Returns:
            fig_activity (bqplot.figure.Figure): The resulting bar plot figure.
        '''

        sc_x1, sc_y1 = bqplot.OrdinalScale(), bqplot.LinearScale()
        c_sc = bqplot.OrdinalColorScale(colors = [bqplot.CATEGORY10[0]])

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
            y = dist_simulated, color_mode = 'element', color = [0] * len(dist_simulated),
            scales = {'x': sc_x1, 'y': sc_y1, 'color': c_sc},
            opacities = [0.5] * len(dist_simulated), stroke = 'black')

        title_label = bqplot.Label(x = [0.1], y = [float(max(dist_real)) / 2],
            font_size = 15, font_weight = 'bolder', colors = ['black'],
            text = [title], enable_move = True)

        fig_activity = bqplot.Figure(marks=[bar_chart_real, bar_chart_simulated,
            title_label], min_width = 300, min_height = 75, axes=[bar_x, bar_y],
            fig_margin = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0})

        return fig_activity

    def validation_selection_process(self, center, plot = False):
        data = {'CP': [], 'Percentage Used': [], 'Data Type': [], 'Number of CPs': []}
        # print(self.ID)

        total_training_sessions = 0
        total_test_sessions = 0
        total_simulated_sessions = [0] * len(self.all_simulated_sessions)
        hubs_data = {'location_key': [], 'longitude': [], 'latitude': []}
        # print('habit = %s' % len(set(self.centers_css[center]['habit'])))
        # print('habit = %s' % str(set(self.centers_css[center]['habit'])))
        for cs in set(self.centers_css[center]['habit']):
            total_training_sessions += len(self.training_sessions.loc[self.training_sessions.location_key == cs])
            total_test_sessions += len(self.test_sessions.loc[self.test_sessions.location_key == cs])
            for i in range(len(self.all_simulated_sessions)):
                if len(self.all_simulated_sessions[i]) != 0:
                   total_simulated_sessions[i] += len(self.all_simulated_sessions[i].loc[self.all_simulated_sessions[i]['location_key'] == cs])
                else:
                    print('WARNING: The length of all_simulated_sessions[i] is zero (not hub).')
            hubs_data['location_key'].append(cs)
            hubs_data['longitude'].append(self.environment.css_info[cs]['longitude'])
            hubs_data['latitude'].append(self.environment.css_info[cs]['latitude'])

        hubs = []
        hubs_df = pandas.DataFrame(hubs_data)
        location_key_converter = hubs_df.groupby(
            ['longitude','latitude'])['location_key'].unique().reset_index()
        location_key_hubs = list(location_key_converter['location_key'])
        # print('location_key_hubs = %s' % len(location_key_hubs[0]))
        # print('location_key_hubs = %s' % location_key_hubs)

        for hub in location_key_hubs:
        #     print('hub = %s' % hub)
            nr_of_real_sessions_in_cs = 0
            for cs in hub:
                nr_of_real_sessions_in_cs += \
                    len(self.training_sessions.loc[self.training_sessions.location_key == cs])
            data['CP'].append(hub[0] if len(hub) == 1 else 'Hub')
            if total_training_sessions != 0:
                data['Percentage Used'].append(nr_of_real_sessions_in_cs / total_training_sessions)
            else:
                data['Percentage Used'].append(0)
            data['Data Type'].append('Training')
            data['Number of CPs'].append(len(location_key_hubs))

            nr_of_real_sessions_in_cs = 0
            for cs in hub:
                nr_of_real_sessions_in_cs += \
                    len(self.test_sessions.loc[self.test_sessions.location_key == cs])
            data['CP'].append(hub[0] if len(hub) == 1 else 'Hub')
            if total_test_sessions != 0:
                data['Percentage Used'].append(nr_of_real_sessions_in_cs / total_test_sessions)
            else:
                data['Percentage Used'].append(0)
            data['Data Type'].append('Test')
            data['Number of CPs'].append(len(location_key_hubs))

            all_simulated_sessions_in_cs = [0] * len(self.all_simulated_sessions)
            for i in range(len(self.all_simulated_sessions)):
                for cs in hub:
                    if len(self.all_simulated_sessions[i]) != 0:
                        all_simulated_sessions_in_cs[i] += \
                            len(self.all_simulated_sessions[i].loc[self.all_simulated_sessions[i]['location_key'] == cs])
                    else:
                        print('WARNING: The length of all_simulated_sessions[i] is zero (hub).')
        #     print('all_simulated_sessions_in_cs = %s' % all_simulated_sessions_in_cs)
        #     print('total_simulated_sessions = %s' % total_simulated_sessions)
            for i in range(len(self.all_simulated_sessions)):
                data['CP'].append(hub[0] if len(hub) == 1 else 'Hub')
                if total_simulated_sessions[i] != 0:
                    data['Percentage Used'].append(all_simulated_sessions_in_cs[i] / total_simulated_sessions[i])
                else:
                    data['Percentage Used'].append(0)
                data['Data Type'].append('Simulated')
                data['Number of CPs'].append(len(location_key_hubs))
            data['CP'].append(hub[0] if len(hub) == 1 else 'Hub')
            data['Percentage Used'].append(numpy.mean([all_simulated_sessions_in_cs[i] / total_simulated_sessions[i] if total_simulated_sessions[i] != 0 else 0 \
                for i in range(len(self.all_simulated_sessions))]))
            data['Data Type'].append('Mean Simulated')
            data['Number of CPs'].append(len(location_key_hubs))

        data = pandas.DataFrame(data)
        training_score, test_score = self.get_score(data)

        # print('plotting')
        if plot:
            fig = plt.figure(figsize=(16, 8))
            # print(pandas.DataFrame(data))
            grid = seaborn.pointplot(x = 'CP', y = 'Percentage Used', data = data,
                              hue = 'Data Type', capsize = 0.2, join = False, dodge = True,
                              hue_order = ['Training', 'Simulated', 'Test'])
            plt.ylim(-0.1, 1.1)
            plt.plot([-1, len(location_key_hubs) + 1 + 1], [0, 0], lw=2, color='grey')
            plt.plot([-1, len(location_key_hubs) + 1 + 1], [1, 1], lw=2, color='grey')

            plt.title('Training Score: %.2f\nTest Score: %.2f' % (training_score, test_score), fontsize = 30)
        return self.ID, center, training_score, test_score, len(location_key_hubs), data

    def get_score(self, data):
        training_score = 0
        test_score = 0
        for cs in set(data['CP']):
            training = numpy.mean(data.loc[(data['CP'] == cs) & (data['Data Type'] == 'Training')]['Percentage Used'])
            test = numpy.mean(data.loc[(data['CP'] == cs) & (data['Data Type'] == 'Test')]['Percentage Used'])
            simulated = numpy.mean(data.loc[(data['CP'] == cs) & (data['Data Type'] == 'Simulated')]['Percentage Used'])
            training_score += abs(training - simulated) / max(1 - training, training)
            test_score += abs(test - simulated) / max(1 - test, test)

        return training_score / len(set(data['CP'])), test_score / len(set(data['CP']))

    def plot_css_on_map(self, center, size = 0.2):
        total_training_sessions = 0
        total_test_sessions = 0
        total_simulated_sessions = [0] * len(self.all_simulated_sessions)
        for cs in self.centers_css[center]['habit']:
            total_training_sessions += len(self.training_sessions.loc[self.training_sessions.location_key == cs])
            total_test_sessions += len(self.test_sessions.loc[self.test_sessions.location_key == cs])
            for i in range(len(self.all_simulated_sessions)):
                total_simulated_sessions[i] += len(self.all_simulated_sessions[i].loc[self.all_simulated_sessions[i].location_key == cs])


        cities_in_center = set()
        for cs in self.centers_css[center]['distance']:
            cities_in_center.add(self.environment.css_info[cs]['city'])
        if len(cities_in_center) > 1:
            print('WARNING: multiple cities in one center (%s)' % (cities_in_center))
        city_of_center = cities_in_center.pop()

        features = []

        feature = {"type": "Feature", "properties": {"type": "cluster",
                                                     "center": center,
                                                     "frequency_of_use": self.centers_info[center]['nr_of_sessions_in_center'],
                                                     "style": {"fillOpacity": 0.8, "smoothFactor": 0, "stroke": True,
                                                               "fillColor": 'blue', "color": "#000000"}},
                "geometry": {"type": "Point", "coordinates": list(center)}}
        features.append(feature)

        for nr, cs in enumerate(self.centers_css[center]['distance']):
            coordinates = [self.environment.css_info[cs]['longitude'],
                           self.environment.css_info[cs]['latitude']]

            if total_training_sessions == 0:
                frequency_of_use = size
            else:
                frequency_of_use = size + len(self.training_sessions.loc[self.training_sessions.location_key == cs]) / total_training_sessions

            feature = {"type": "Feature", "properties": {"type": "charging station",
                                                         "cs": cs,
                                                         "frequency_of_use": frequency_of_use,
                                                         "style": {"fillOpacity": 0.8, "smoothFactor": 0, "stroke": True,
                                                                   "color": "#FF0000", "fillColor": 'grey'}},
                       "geometry": {"type": "Polygon", "coordinates":
                                    [self.polygon_generator(offset=coordinates, size = frequency_of_use, shape = 'square')]}}
            features.append(feature)
        data = {"type": "FeatureCollection", "features": features}
        city_centers = {'Amsterdam': [52.373249, 4.896025], 'Utrecht': [52.093562, 5.117568],
                       'Rotterdam': [51.922091, 4.474679], 'Den Haag': [52.066520, 4.304839]}
        m = ipyleaflet.Map(center=city_centers[city_of_center], zoom=12)

        layer_zone = ipyleaflet.GeoJSON(data=data, hover_style={'fillColor': 'black'},
                                        style = {'color':'black', 'weight': 0.7, 'fillColor':'blue', 'fillOpacity':0.6})

        m.add_layer(layer_zone)
        IPython.display.display(m)

    def polygon_generator(self, offset, size = 1, edge_length_lon = 0.0005, edge_length_lat = 0.0005, shape = 'hexagon'):
        '''
        Generates the coordinates of a hexagon with the center being the offset.

        Args:
            offset: center of the polygon

        Kwargs:
            size: scaling factor for the edge lengths
            edge_length_lon: edge length in longitude direction
            edge_length_lat: edge length in latitude direction
            shape: shape of polygon, e.g. 'hexagon', 'triangle' or 'square'

        Returns:
            An array of coordinates (lon, lat) for the center of the polygon.
        '''

        if shape == 'hexagon':
            stepsize = 60
        elif shape == 'triangle':
            stepsize = 120
        elif shape == 'square':
            stepsize = 90
        else:
            print("Shape undefined in polygon_generator function, now making a hexagon")
            stepsize = 60

        coords = []
        lon, lat = offset
        for angle in range(0, 360, stepsize):
            lon = numpy.cos(numpy.radians(angle)) * edge_length_lon * size + offset[0]
            lat = numpy.sin(numpy.radians(angle)) * edge_length_lat * size + offset[1]
            coords.append([lon, lat])
        return coords

    def __lt__(self, other):
        ''' The comparison method of this class.
            (dummy method needed for heapq bug)

        Args:
            other (Agent): The other agent to compare this agent with.

        Returns:
            (bool): Returns True if this agent is less than the other agent.
        '''
        return True
    
    def __deepcopy__(self, memodict={}):
        copy_object = Agent()
        copy_object.value = self.value
        return copy_object

