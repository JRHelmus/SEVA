'''
experiments.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Last updated on June 2017

Description:
    All implemented experiments can be found in this .py file. Examples of how to
    run every experiment can be found in the main() function. Paralellism is
    supported and the MAX_CPUs can be set to the number of cores to use. Note that
    by default this should be 1 and that implies the code is run sequentially.
    All results of the experiments are stored in matching directories within the
    data/experiment_results/ directory.
'''

import simulation
import data_handler
import environment
import agent
import json
import pandas
import numpy
import datetime
import pickle
#import IPython
import ipywidgets
import bqplot
import multiprocessing
import os
import time
import random
import pandas as pd

MAX_CPUs = 2


def experiment_initialization(parameter, values, measures, experiment_name, number_of_agents = 30):
    ''' This function runs the initialization of the simulation for various
        given values of the specified parameter. The given measures are the
        output of the experiment. The results are stored in the
        data/experiment_results directory. A subdirectory is created (if it does
        not already exist), containing the measure names and parameter name, in
        which the results are stored.

    Args:
        parameter (str): The parameter to vary for the experiment.
        values (List[int or float]): A list of values the parameter can take.
        measures (List[str]): A list of measure names. Possible measures are
            number_of_centers, average_number_of_charging_stations and
            walking_preparedness.

    Kwargs:
        number_of_agents (int): Amount of agents to run the experiment with.
            Default is 30.
    '''
    values = list(values) # list(reversed(values)) old version new version ascending

    print('INFO: Started experiment: Initialization which varies %s ' %
        parameter + 'with %s as measures. Varying values are %s. ' %
        (measures, ['%0.2f' % val for val in values]) + '%d agents.' % \
        number_of_agents)
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    parameters_file = 'data/input_parameters/parameters.json'
    with open(parameters_file) as json_data:
        parameters_json = json.load(json_data)

    parameters = [(parameters_file, parameter, i, len(values), value,
        number_of_agents, measures, experiment_name) for i, value in enumerate(values)]

    p = multiprocessing.Pool(MAX_CPUs)
    p.map(inner_loop_experiment_initialization, parameters)

    print('INFO: Finished experiment: Initialization which varies %s ' %
        parameter + 'with %s as measures. Varying values are %s. ' %
        (measures, ['%0.2f' % val for val in values]) + '%d agents.' % \
        number_of_agents)
    # print('INFO: Experiment took %s' %
    #     get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def inner_loop_experiment_initialization(args):
    ''' This function runs the initialization experiment for a single value
        of a parameter measuring one or more measures. It is also in charge
        of storing the results of the experiment for this value.

    Args:
        args (Tuple[Any]): Contains the arguments needed for the experiment,
            namely in order:
                - (str): The path to the parameters file;
                - (str): The parameter to vary for the experiment;
                - (int): The current experiment run number;
                - (int): The total amount of values for the parameter;
                - (int or float): The value of the parameter in this run;
                - (int): Amount of agents to run the experiment with;
                - (List(str)): A list of measure names. Possible measures are
                    number_of_centers, average_number_of_charging_stations and
                    walking_preparedness.
         
    '''

    parameters_file, parameter, i, max_runs, value, number_of_agents, measures, experiment_name = args
    start_time = time.process_time()

    print('INFO: Experiment run %d of %d (%s = %s, measuring %s)' % (i + 1,
        max_runs, parameter, value, measures))

    sim = simulation.Simulation(parameters_file, measures = measures,
        overwrite_parameters = {parameter: value,
            'number_of_agents': number_of_agents,
            'agent_initialization': 'create',
            'filepath_agent_database': ''})
    
    newpath = 'data/experiment_results/%s'  % (experiment_name)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    for m in measures:
        experiment_dir = 'data/experiment_results/%s/experiment_initialization_measures_%s_varying_%s' % \
            (experiment_name, m, parameter)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        experiment_filename = experiment_dir + \
            '/%0.2f_%s_%s_agents.pkl' % (value, parameter, str(number_of_agents))

        with open(experiment_filename, 'wb') as experiment_file:
            pickle.dump(sim.sensors[m], experiment_file)
        print('INFO: Partial results stored in %s' % experiment_filename)

    print('INFO: Experiment run %d (%s = %s, measuring %s) took %s' % (i + 1,
        parameter, value, measures,
        get_time_string(time.process_time() - start_time)))
    print('INFO: End time: %s' % datetime.datetime.now())

def experiment_simulation(parameter, values, measures, simulation_repeats = -1, experiment_name = 'Unnamed_experiment',
    number_of_agents = 30, method = 'relMAE', hack_parameters_file = None):
    ''' This function runs the simulation for various given values of the
        specified parameter. The given measures are the output of the
        experiment. The results are stored in the data/experiment_results
        directory. A subdirectory is created (if it does not already exist),
        containing the measure names and parameter name, in which the results
        are stored.

    Args:
        parameter (str): The parameter to vary for the experiment.
        values (List[int or float]): A list of values the parameter can take.
        measures (List[str]): A list of measure names. Possible measures are
            charging_station_validation, agent_validation,
            time_per_simulation.

    Kwargs:
        number_of_agents (int): Amount of agents to run the experiment with.
            Default is 30.
        simulation_repeats (int): Amount of repeats for the experiment. Default
            is 30.
    '''
    values = list(values)

    if parameter == 'number_of_agents':
        print('INFO: Started experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%d simulation_repeats.' % \
            simulation_repeats)
    else:
        print('INFO: Started experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%s agents and %d simulation_repeats.' % \
            (number_of_agents, simulation_repeats))
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    if hack_parameters_file != None:
        parameters_file = hack_parameters_file
    else:
        parameters_file = 'data/input_parameters/parameters.json'

    parameters = [(parameters_file, parameter, i, len(values), value, number_of_agents,
        simulation_repeats, measures, experiment_name, method) for i, value in enumerate(values)]

    p = multiprocessing.Pool(MAX_CPUs)
    p.map(inner_loop_experiment_simulation, parameters)

    if parameter == 'number_of_agents':
        print('INFO: Finished experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%d simulation_repeats.' % \
            simulation_repeats)
    else:
        print('INFO: Finished experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%s agents and %d simulation_repeats.' % \
            (number_of_agents, simulation_repeats))
    # print('INFO: Experiment took %s' %
    #     get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def inner_loop_experiment_simulation(args):
    ''' This function runs the simulation experiment for a single value
        of a parameter measuring one or more measures. It is also in charge
        of storing the results of the experiment for this value.

    Args:
        args (Tuple[Any]): Contains the arguments needed for the experiment,
            namely in order:
                - (str): The path to the parameters file;
                - (str): The parameter to vary for the experiment;
                - (int): The current experiment run number;
                - (int): The total amount of values for the parameter;
                - (int or float): The value of the parameter in this run;
                - (int): Amount of repeats for the experiment;
                - (List(str)): A list of measure names. Possible measures are
                    number_of_centers, average_number_of_charging_stations and
                    walking_preparedness;
                - (str): Method of validation, either MAE or relMAE.
    '''

    (parameters_file, parameter, i, max_runs, value, number_of_agents, simulation_repeats, measures, \
     experiment_name, method) = args
    start_time = time.process_time()
    
    

    print('INFO: Experiment run %d of %d (%s = %s, measuring %s)' % (i + 1,
        max_runs, parameter, value, measures))

    if parameter == 'number_of_agents':
        sim = simulation.Simulation(parameters_file, measures = measures,
            overwrite_parameters = {parameter: value}) #NOTE: changed this
    elif parameter == "adding_non_habitual_agents":
         sim = simulation.Simulation(parameters_file, measures = measures,
            overwrite_parameters = {'number_of_agents': number_of_agents,
                                   'non_habitual_agents': {"Amsterdam": int(value),
                                                          "Den Haag": int(value),
                                                          "Rotterdam": int(value),
                                                          "Utrecht": int(value)}}) #NOTE: changed this
    else:
        sim = simulation.Simulation(parameters_file, measures = measures,
            overwrite_parameters = {parameter: value,
            'number_of_agents': number_of_agents}) #NOTE: changed this 
    
    newpath = 'data/experiment_results/%s'  % (experiment_name)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    try:
        if parameter == 'simulation_repeats':
            simulation_repeats = value
        sim.repeat_simulation(repeat = simulation_repeats, measures = measures,
            method = method, display_progress = False)

        for m in measures:
            experiment_dir = 'data/experiment_results/%s/experiment_simulation_measures_%s_varying_%s' % \
                (experiment_name, m, parameter)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            if parameter == 'number_of_agents':
                experiment_filename = experiment_dir + '/%s_%s_%s_simulation_repeats.pkl' % (str(value), parameter, str(simulation_repeats))
            else:
                experiment_filename = experiment_dir + '/%0.2f_%s_%s_agents_%s_simulation_repeats.pkl' % (value, parameter, str(number_of_agents), str(simulation_repeats))

            with open(experiment_filename, 'wb') as experiment_file:
                pickle.dump(sim.sensors[m], experiment_file)
            print('INFO: Partial results stored in %s' % experiment_filename)
    except Exception as e:
        print(e)
        print("WARNING: Thus failed")

    print('INFO: Experiment run %d (%s = %s, measuring %s) took %s' %
        (i + 1, parameter, value, measures,
        get_time_string(time.process_time() - start_time)))
    print('INFO: End time: %s' % datetime.datetime.now())

def experiment_initialization_and_simulation(parameter, values, measures, experiment_name = 'Unnamed_experiment',
    number_of_agents = 30, simulation_repeats = 30, method = 'relMAE', initialization_repeats = 1,ratio_unhabitual=0):
    ''' This function runs the experiment that requires initialization as well
        as simulation. For various given values of the specified parameter the
        given measures will be measured. The results are stored in the
        data/experiment_results directory. A subdirectory is created (if it does
        not already exist), containing the measure names and parameter name, in
        which the results are stored.

        Args:
            parameter (str): The parameter to vary for the experiment.
            values (List[int or float]): A list of values the parameter can take.
            measures (List[str]): A list of measure names. Possible measures are
                charging_station_validation, agent_validation,
                time_per_simulation.

        Kwargs:
            number_of_agents (int) Amount of agents to do the experiment with.
                Default is 30.
            simulation_repeats (int): Amount of repeats for the experiment. Default
                is 30.
    '''

    values = list(values)

    print('INFO: Started experiment: initialization and simulation which varies %s ' %
        parameter + 'with %s as measures. Varying values are %s. ' %
        (measures, [val for val in values]) + '%d agents and %d simulation_repeats with %s unhabitual agents ratio.' % \
        (number_of_agents, simulation_repeats, ratio_unhabitual))
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    parameters_file = 'data/input_parameters/parameters.json'
    
    if initialization_repeats > 1:
        parameters = [(parameters_file, parameter, i, len(values), value,
            number_of_agents, simulation_repeats, measures, experiment_name+"_init_repeat_"+str(repeat), method, ratio_unhabitual)
            for i, value in enumerate(values) for repeat in range(initialization_repeats)]
    else:
        parameters = [(parameters_file, parameter, i, len(values), value,
            number_of_agents, simulation_repeats, measures, experiment_name, method, ratio_unhabitual)
            for i, value in enumerate(values)]
    
    

    p = multiprocessing.Pool(MAX_CPUs)
    p.map(inner_loop_experiment_initialization_and_simulation, parameters)

    print('INFO: Finished experiment: initialization and simulation which varies %s ' %
        parameter + 'with %s as measures. Varying values are %s. ' %
        (measures, [val for val in values]) + '%d agents and %d simulation_repeats with %s unhabitual agents ratio.' % \
        (number_of_agents, simulation_repeats, ratio_unhabitual))
    # print('INFO: Experiment took %s' %
    #     get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def inner_loop_experiment_initialization_and_simulation(args):
    ''' This function runs an experiment that requires initialization and
        simulation. For a single value of a parameter measuring one or more
        measures are measured. This function is also in charge of storing the
        results of the experiment for this value.

    Args:
        args (Tuple[Any]): Contains the arguments needed for the experiment,
            namely in order:
                - (str): The path to the parameters file;
                - (str): The parameter to vary for the experiment;
                - (int): The current experiment run number;
                - (int): The total amount of values for the parameter;
                - (int or float): The value of the parameter in this run;
                - (int): Amount of agents in the experiment;
                - (int): Amount of repeats for the experiment;
                - (List(str)): A list of measure names. Possible measures are
                    number_of_centers, average_number_of_charging_stations and
                    walking_preparedness;
                - (str): Method of validation, either MAE or relMAE.
    '''

    (parameters_file, parameter, i, max_runs, value, number_of_agents, simulation_repeats, measures, \
     experiment_name, method, ratio_unhabitual) = args
    start_time = time.process_time()

    print('INFO: Experiment run %d of %d (%s = %s, measuring %s)' % (i + 1,
        max_runs, parameter, value, measures))

    if "adding_habitual_agents" in experiment_name: # JM: Added for the experiment of adding habitual agents.
        if parameter == "adding_extra_agents":
            extra_agents = int(value)
            print('this is the number of normal agents' + str(number_of_agents))
            print('this is the number of extra_agents agents' + str(extra_agents))
            sim = simulation.Simulation(parameters_file, measures = measures,
                overwrite_parameters = {'number_of_agents': number_of_agents,
                                    'agent_initialization': 'load_and_use',
                                   'add_agents_during_simulation':extra_agents,
                                   "non_habitual_agents": {"Amsterdam": ratio_unhabitual,
                                                              "Den Haag": ratio_unhabitual,
                                                              "Rotterdam": ratio_unhabitual,
                                                              "Utrecht": ratio_unhabitual} })
            
            #for agent_ID in agent_IDs:
            #    sim.agents[agent_ID].start_date_agent = sim.start_date_simulation
            #    sim.agents[agent_ID].end_date_agent = sim.stop_condition_parameters['max_time']
        else:
            extra_agents = 0
            print('this is the number of normal agents' + str(value))
            print('this is the number of extra_agents agents' + str(extra_agents))
            sim = simulation.Simulation(parameters_file, measures = measures,
                   overwrite_parameters = {'number_of_agents': value,
                                        'agent_initialization': 'load_and_use',
                                       'add_agents_during_simulation':extra_agents,
                                       "non_habitual_agents": {"Amsterdam": ratio_unhabitual,
                                                              "Den Haag": ratio_unhabitual,
                                                              "Rotterdam": ratio_unhabitual,
                                                              "Utrecht": ratio_unhabitual} })
        
            #agent_IDs = list(sim.agents.keys())
            #agent_IDs = [agent_ID for agent_ID in agent_IDs if not 'car2go' in agent_ID]
            #random.shuffle(agent_IDs)

            #for j, agent_ID in enumerate(agent_IDs):
            #    if j > value:
            #        sim.agents[agent_ID].start_date_agent = sim.stop_condition_parameters['max_time']

            #    else:
            #        sim.agents[agent_ID].start_date_agent = sim.start_date_simulation
            #        sim.agents[agent_ID].end_date_agent = sim.stop_condition_parameters['max_time']
    else:
        sim = simulation.Simulation(parameters_file, measures = measures,
            overwrite_parameters = {parameter: value,
            'number_of_agents': number_of_agents,
            'agent_initialization': 'create_and_use'}) # JRH: here the create and use is selected as option since the experiment is about the initialization 
                
    print('INFO: Experiment run %d of %d is done with initialization at %s' %
        ((i + 1), max_runs, datetime.datetime.now()))

    newpath = 'data/experiment_results/%s'  % (experiment_name)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    try:
        sim.repeat_simulation(repeat = simulation_repeats, measures = measures, method = method,
            display_progress = False)

        for m in measures:
            experiment_dir = 'data/experiment_results/%s/experiment_initialization_and_simulation_measures_%s_varying_%s' % \
                (experiment_name, m, parameter)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            
            experiment_filename = experiment_dir + '/%s_%s_%s_agents_%s_simulation_repeats_%s_unhabitual_agents.pkl' % (str(value), parameter, str(number_of_agents), str(simulation_repeats), str(ratio_unhabitual))
            with open(experiment_filename, 'wb') as experiment_file:
                pickle.dump(sim.sensors[m], experiment_file)
            print('INFO: Partial results stored in %s' % experiment_filename)
    except Exception as e:
        print(e)
        print("WARNING: Thus failed")

    print('INFO: Experiment run %d (%s = %s, measuring %s) took %s' %
        (i + 1, parameter, value, measures,
        get_time_string(time.process_time() - start_time)))
    print('INFO: End time: %s' % datetime.datetime.now())

def create_agents_selection_process_extension(store_IDs = False):
    ''' This function creates all possible agents and stores them in the agent
        database. Optionally a file all_agent_IDs.pkl is stored with all IDs of
        valid agents.

    Kwargs:
        store_IDs (bool): If True the agent IDs of all agents are stored in a
            pickle file.
    '''

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'filepath_agent_database': 'data/agent_database/all_agents/',
        'info_printer': True, 'IDs_from_memory': False,
        'selection_process': 'choice_model',
                'selection_process_parameters':
                {'Amsterdam': {'intercept': 1.13,
                               'distance': -3.52,
                               'charging_speed': -0.89,
                               'charging_fee': -0.50,
                               'parking_fee': -0.77},
                 'The Hague': {'intercept': 1.64,
                               'distance': -3.98,
                               'charging_speed': -1.11,
                               'charging_fee': -0.88,
                               'parking_fee': -1.72},
                 'Rotterdam': {'intercept': 1.53,
                               'distance': -4.04,
                               'charging_speed': -1.08,
                               'charging_fee': -1.55,
                               'parking_fee': 0},
                 'Utrecht': {'intercept': 1.48,
                             'distance': -3.19,
                             'charging_speed': -0.87,
                             'charging_fee': -3.55,
                             'parking_fee': 0}}
                })
    if store_IDs:
        with open ('data/experiment_results/all_agent_IDs.pkl', 'wb') as agents_file:
            pickle.dump(list(sim.agents.keys()), agents_file)

            
def create_agents(store_IDs = False, filepath_store_pickle_file = "data/agent_database/all_agents.pkl"):
    ''' This function creates all possible agents and stores them in the agent
        database. Optionally a file all_agent_IDs.pkl is stored with all IDs of
        valid agents.

    Kwargs:
        store_IDs (bool): If True the agent IDs of all agents are stored in a
            pickle file.
    '''

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'filepath_agent_database': 'data/agent_database/all_agents/',
        'info_printer': True  })

    previous_time = time.process_time()
    
    if store_IDs:
        with open (filepath_store_pickle_file, 'wb') as agents_file:
            pickle.dump(sim.agents, agents_file, pickle.HIGHEST_PROTOCOL)
        
        new_time = time.process_time() - previous_time
        if new_time < 60:
             print('\tINFO: Stored agents in pickle file in %.2f seconds' % new_time)
        elif new_time < 3600:
            print('\tINFO: Stored agents in pickle file in %.2f minutes' % (new_time/60))
        else:
            print('\tINFO: Stored agents in pickle file in %.2f hours' % (new_time/3600))
                  
                  
def create_agents_battery_extension(store_IDs = False):
    ''' This function creates all possible agents and stores them in the agent
        database. Optionally a file all_agent_IDs.pkl is stored with all IDs of
        valid agents.

    Kwargs:
        store_IDs (bool): If True the agent IDs of all agents are stored in a
            pickle file.
    '''

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'filepath_agent_database': 'data/agent_database/all_non_changing_agents_bin60/',
        'info_printer': True, 'IDs_from_agent_database': False, \
        'start_date_training_data': "01-01-2014", \
        'end_date_training_data': "01-01-2018", \
        'start_date_test_data': "01-01-2014", \
        'end_date_test_data': "01-01-2018", \
        'bin_size_dist': 60})
    if store_IDs:
        with open ('data/experiment_results/all_agent_IDs_bin60.pkl', 'wb') as agents_file:
            pickle.dump(list(sim.agents.keys()), agents_file)



def create_unmerged_general_data_and_environment():
    ''' This function creates all possible agents and stores them in the agent
        database. Optionally a file all_agent_IDs.pkl is stored with all IDs of
        valid agents.

    Kwargs:
        store_IDs (bool): If True the agent IDs of all agents are stored in a
            pickle file.
    '''

    sim = simulation.Simulation("data/input_parameters/parameters_unmerged.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'number_of_agents': 0,
        "IDs_from_agent_database": False,
        'filepath_agent_database': 'data/agent_database/all_agents/',
        'info_printer': True})

def create_general_data_and_environment():
    ''' This function does the general preprocessing '''

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'number_of_agents': 0,
        'filepath_agent_database': 'data/agent_database/all_agents/',
        'info_printer': True,
        "IDs_from_agent_database": False,
        'general_preprocess': True,
        'environment_from_memory': False})


def incidental_agents_experiments(number_of_agents, simulation_repeats, step, min_, max_):
    print('INFO: Started incidental agents experiments with %d agents and %d reps' % \
        (number_of_agents, simulation_repeats))
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    parameters_file = 'data/input_parameters/parameters.json'

    parameters = [(number_of_agents, simulation_repeats, i, value, len(numpy.arange(min_, max_ + step, step))) \
        for i, value in enumerate(list(numpy.arange(min_, max_ + step, step))[::-1])]

    p = multiprocessing.Pool(MAX_CPUs)
    p.map(inner_loop_incidental_agents_experiments, parameters)

    print('INFO: Finished incidental agents experiments with %d agents and %d reps' % \
        (number_of_agents, simulation_repeats))
    print('INFO: Experiment took %s' %
        get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def inner_loop_incidental_agents_experiments(args):
    number_of_agents, repeats, i, percentage_incidental_users, max_runs = args
    start_time = time.process_time()

    print('INFO: Experiment run %d of %d (percentage = %s, agents = %s, reps = %s)' % (i + 1,
        max_runs, percentage_incidental_users, number_of_agents, repeats))

    selection_process_parameters = {'Amsterdam': {'intercept': 1.13,
                   'distance': -3.52,
                   'charging_speed': -0.89,
                   'charging_fee': -0.50,
                   'parking_fee': -0.77},
     'The Hague': {'intercept': 1.64,
                   'distance': -3.98,
                   'charging_speed': -1.11,
                   'charging_fee': -0.88,
                   'parking_fee': -1.72},
     'Rotterdam': {'intercept': 1.53,
                   'distance': -4.04,
                   'charging_speed': -1.08,
                   'charging_fee': -1.55,
                   'parking_fee': 0},
     'Utrecht': {'intercept': 1.48,
                 'distance': -3.19,
                 'charging_speed': -0.87,
                 'charging_fee': -3.55,
                 'parking_fee': 0}}
    try:
        simulation_agents = pd.read_pickle("data/agent_database/agents_temp.pkl")
        simulation_agents = random.sample(simulation_agents, number_of_agents)
    except Exception as e:
        print(e)
        raise e
        print('File not found.')

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters={'filepath_agent_database': 'data/agent_database/all_agents_car2go/',
                              "agent_initialization": "create_and_use",
                              "agent_creation_method": "random",
                              'number_of_agents': number_of_agents,
                              'info_printer': True,
                              'agent_IDs': [],
                              'selection_process': 'choice_model',
                              'rollout_strategy_to_use': 'none',
                              'selection_process_parameters': selection_process_parameters,
                              "non_habitual_agents": {"Amsterdam": percentage_incidental_users,
                                                      "Den Haag": percentage_incidental_users,
                                                      "Rotterdam": percentage_incidental_users,
                                                      "Utrecht": percentage_incidental_users}})

    print('INFO: Experiment run %d of %d (percentage = %s, agents = %s, reps = %s) initialized at %s' % (i + 1,
        max_runs, percentage_incidental_users, number_of_agents, repeats, datetime.datetime.now()))

    try:

        sim.repeat_simulation(repeat = repeats, measures = [])
        # validation_selection_process_data = []

        agent_sensors = []
        for agent in sim.agents.values():
            for center in agent.centers_css:
                # agent_id, center, training_score, test_score, nr_cps, data_percentages = \
                #     agent.validation_selection_process(center, plot = False)
                # validation_selection_process_data.append((agent_id, center, training_score, test_score, nr_cps, data_percentages))
                agent_sensors.append((agent.ID, agent.sensors))

        # number_of_agents = len(sim.agents.values())
        # with open('indicental_users_experiments/validation_selection_process_data_%d_agents_%d_repeats_%.1f.pkl' % (number_of_agents, repeats, percentage_incidental_users), 'wb') as agent_file:
        #     pickle.dump(validation_selection_process_data, agent_file)

        with open('data/indicental_users_experiments_all/agent_sensors_%d_agents_%d_repeats_%.1f.pkl' % (number_of_agents, repeats, percentage_incidental_users), 'wb') as agent_file:
            pickle.dump(agent_sensors, agent_file)
    except Exception as e:
        print(e)
        print('Thus failed.')

    print('INFO: Experiment run %d of %d (percentage = %s, agents = %s, reps = %s) took %s' % (i + 1,
        max_runs, percentage_incidental_users, number_of_agents, repeats,
        get_time_string(time.process_time() - start_time)))
    print('INFO: End time: %s' % datetime.datetime.now())

def rollout_strategies_experiments(number_of_agents, simulation_repeats, rollout_strategies_repeats,
    rollout_strategies, percentage_incidental_users, number_of_CPs_to_add_min,
    number_of_CPs_to_add_max, number_of_CPs_to_add_step):
    print('INFO: Started rollout strategies experiments with %d agents and %d reps' % \
        (number_of_agents, simulation_repeats))
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    parameters_file = 'data/input_parameters/parameters.json'

    parameters = []
    i = 0
    total = len(rollout_strategies) * len(range(number_of_CPs_to_add_min,
        number_of_CPs_to_add_max + number_of_CPs_to_add_step, number_of_CPs_to_add_step)) * rollout_strategies_repeats
    for rollout_strategies_repeat in range(rollout_strategies_repeats):
        for rollout_strategy in rollout_strategies:
            for number_of_CPs_to_add in range(number_of_CPs_to_add_min, number_of_CPs_to_add_max + number_of_CPs_to_add_step, number_of_CPs_to_add_step):
                parameters.append((i, total, number_of_agents, simulation_repeats, rollout_strategy, \
                    number_of_CPs_to_add, rollout_strategies_repeat, percentage_incidental_users))
                i += 1

    print('INFO: %d experiments to be run.' % total)
    p = multiprocessing.Pool(MAX_CPUs)
    p.map(inner_loop_rollout_strategies_experiments, parameters)

    print('INFO: Finished rollout strategies experiments with %d agents and %d reps' % \
        (number_of_agents, simulation_repeats))
    print('INFO: Experiment took %s' %
        get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def inner_loop_rollout_strategies_experiments(args):
    i, total, number_of_agents, simulation_repeats, rollout_strategy, \
        number_of_CPs_to_add, rollout_strategies_repeat, percentage_incidental_users = args
    start_time = time.process_time()

    if os.path.exists('rollout_strategies_experiments/agent_sensors_%s_strategy_%d_cps_added_%d_agents_%d_incidental_users_%d_sim_repeats_%d_rollout_strategies_repeat.pkl' % (rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, simulation_repeats, rollout_strategies_repeat)):
        print('INFO: This experiment was already run (rollout strategy = %s, cps added = %s, agents = %s, non-habitual agents = %s, rollout_strategies_repeat = %s)' %
            (rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, rollout_strategies_repeat))
        return

    print('INFO: Experiment run %d of %d (rollout strategy = %s, cps added = %s, agents = %s, non-habitual agents = %s, rollout_strategies_repeat = %s)' %
        (i, total, rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, rollout_strategies_repeat))


    selection_process_parameters = {'Amsterdam': {'intercept': 1.13,
                   'distance': -3.52,
                   'charging_speed': -0.89,
                   'charging_fee': -0.50,
                   'parking_fee': -0.77},
     'The Hague': {'intercept': 1.64,
                   'distance': -3.98,
                   'charging_speed': -1.11,
                   'charging_fee': -0.88,
                   'parking_fee': -1.72},
     'Rotterdam': {'intercept': 1.53,
                   'distance': -4.04,
                   'charging_speed': -1.08,
                   'charging_fee': -1.55,
                   'parking_fee': 0},
     'Utrecht': {'intercept': 1.48,
                 'distance': -3.19,
                 'charging_speed': -0.87,
                 'charging_fee': -3.55,
                 'parking_fee': 0}}

    try:
        simulation_agents = pd.read_pickle("data/experiment_results/all_agent_IDs2018.pkl")
        number_of_agents = len(list(simulation_agents.values()))
        simulation_agents = random.sample(simulation_agents.keys(), number_of_agents)
    except Exception as e:
        print(e)
        raise e
        print('File not found.')

    sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters={'filepath_agent_database': "data/experiment_results/all_agent_IDs2018.pkl",
                              "agent_initialization": "load_and_use",
        'number_of_agents': number_of_agents,
        'info_printer': True,
        'agent_IDs': simulation_agents,
        'selection_process': 'choice_model',
        'selection_process_parameters': selection_process_parameters,
        "non_habitual_agents": {"Amsterdam": percentage_incidental_users,
                                "Den Haag": percentage_incidental_users,
                                "Rotterdam": percentage_incidental_users,
                                "Utrecht": percentage_incidental_users},
        "number_of_CPs_to_add": {"Amsterdam": number_of_CPs_to_add,
                                  "Den Haag": number_of_CPs_to_add,
                                  "Rotterdam": number_of_CPs_to_add,
                                  "Utrecht": number_of_CPs_to_add},
        "rollout_strategy_to_use": rollout_strategy})

    print('INFO: Experiment run %d of %d (rollout strategy = %s, cps added = %s, agents = %s, non-habitual agents = %s, rollout_strategies_repeat = %s) initialized at %s' %
        (i, total, rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, rollout_strategies_repeat,
            datetime.datetime.now()))

    try:
        sim.repeat_simulation(repeat = simulation_repeats, measures = [])
        # validation_selection_process_data = []
        agent_sensors = []
        for agent in sim.agents.values():
            for center in agent.centers_css:
                # agent_id, center, training_score, test_score, nr_cps, data_percentages = \
                #     agent.validation_selection_process(center, plot = False)
                # validation_selection_process_data.append((agent_id, center, training_score, test_score, nr_cps, data_percentages))
                agent_sensors.append((agent.ID, agent.sensors))

        # number_of_agents = len(sim.agents.values())
        # with open('indicental_users_experiments/validation_selection_process_data_%d_agents_%d_repeats_%.1f.pkl' % (number_of_agents, repeats, percentage_incidental_users), 'wb') as agent_file:
        #     pickle.dump(validation_selection_process_data, agent_file)

        with open('data/rollout_strategies_experiments/agent_sensors_%s_strategy_%d_cps_added_%d_agents_%d_incidental_users_%d_sim_repeats_%d_rollout_strategies_repeat.pkl' % (rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, simulation_repeats, rollout_strategies_repeat), 'wb') as agent_file:
            pickle.dump(agent_sensors, agent_file)
    except Exception as e:
        print(e)
        raise e
        print('Thus failed.')

    print('INFO: Experiment run %d of %d (rollout strategy = %s, cps added = %s, agents = %s, non-habitual agents = %s, rollout_strategies_repeat = %s) took %s' %
        (i, total, rollout_strategy, number_of_CPs_to_add, number_of_agents, percentage_incidental_users, rollout_strategies_repeat,
            get_time_string(time.process_time() - start_time)))

    print('INFO: End time: %s' % datetime.datetime.now())


def get_time_string(seconds):
    ''' This method converts an float input of seconds to a string that
        captures the duration in English.

    Args:
        seconds (float): Amount of seconds.

    Returns:
        (str): String that captures the duration in English.
    '''

    if seconds < 60:
        return '%.2f seconds' % seconds
    if seconds < 3600:
        return '%.2f minutes' % (seconds / 60)
    return '%.2f hours' % (seconds / 3600)



def case_study_experiment(parameter, values, measures, simulation_repeats = -1,
    number_of_agents = 30, method = 'relMAE', hack_parameters_file = None):
    ''' This function runs the simulation for various given values of the
        specified parameter. The given measures are the output of the
        experiment. The results are stored in the data/experiment_results
        directory. A subdirectory is created (if it does not already exist),
        containing the measure names and parameter name, in which the results
        are stored.

    Args:
        parameter (str): The parameter to vary for the experiment.
        values (List[int or float]): A list of values the parameter can take.
        measures (List[str]): A list of measure names. Possible measures are
            charging_station_validation, agent_validation,
            time_per_simulation.

    Kwargs:
        number_of_agents (int): Amount of agents to run the experiment with.
            Default is 30.
        simulation_repeats (int): Amount of repeats for the experiment. Default
            is 30.
    '''
    values = list(values)

    if parameter == 'number_of_agents':
        print('INFO: Started experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%d simulation_repeats.' % \
            simulation_repeats)
    else:
        print('INFO: Started experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%s agents and %d simulation_repeats.' % \
            (number_of_agents, simulation_repeats))
    start_time = time.process_time()
    print('INFO: Start time: %s' % datetime.datetime.now())

    if hack_parameters_file != None:
        parameters_file = hack_parameters_file
    else:
        parameters_file = 'data/input_parameters/parameters.json'

    parameters = [(parameters_file, parameter, i, len(values), value, number_of_agents,
        simulation_repeats, measures, method) for i, value in enumerate(values)]

    p = multiprocessing.Pool(MAX_CPUs)
    p.map(case_study_inner_loop_experiment_simulation, parameters)

    if parameter == 'number_of_agents':
        print('INFO: Finished experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%d simulation_repeats.' % \
            simulation_repeats)
    else:
        print('INFO: Finished experiment: simulation which varies %s ' %
            parameter + 'with %s as measures. Varying values are %s. ' %
            (measures, [val for val in values]) + '%s agents and %d simulation_repeats.' % \
            (number_of_agents, simulation_repeats))
    # print('INFO: Experiment took %s' %
    #     get_time_string(time.process_time() - start_time))
    print('INFO: End time: %s' % datetime.datetime.now())

def case_study_inner_loop_experiment_simulation(args):
    ''' This function runs the simulation experiment for a single value
        of a parameter measuring one or more measures. It is also in charge
        of storing the results of the experiment for this value.

    Args:
        args (Tuple[Any]): Contains the arguments needed for the experiment,
            namely in order:
                - (str): The path to the parameters file;
                - (str): The parameter to vary for the experiment;
                - (int): The current experiment run number;
                - (int): The total amount of values for the parameter;
                - (int or float): The value of the parameter in this run;
                - (int): Amount of repeats for the experiment;
                - (List(str)): A list of measure names. Possible measures are
                    number_of_centers, average_number_of_charging_stations and
                    walking_preparedness;
                - (str): Method of validation, either MAE or relMAE.
    '''

    parameters_file, parameter, i, max_runs, value, number_of_agents, simulation_repeats, measures, method = args
    start_time = time.process_time()

    print('INFO: Experiment run %d of %d (%s = %s, measuring %s)' % (i + 1,
        max_runs, parameter, value, measures))

    # NOTE change the database to correct database of bin size
    # NOTE change the bin_size to correct value

    sim = simulation.Simulation(parameters_file, measures = measures,
        overwrite_parameters = {parameter: value,
        'agent_initialization': 'load_and_use',
        'filepath_agent_database': 'data/agent_database/all_agents_car2go/',
        'number_of_agents': number_of_agents,
        'bin_size_dist': 20,
        'skip_low_fev_agents': True,
        'skip_high_fev_agents': True,
        'skip_unknown_agents': True,
        'start_date_training_data': "01-01-2014",
        'end_date_training_data': "01-01-2018",
        'start_date_test_data': "01-01-2014",
        'end_date_test_data': "01-01-2018"})

    try:
        if parameter == 'simulation_repeats':
            simulation_repeats = value
        sim.repeat_simulation(repeat = simulation_repeats, measures = measures,
            method = method, display_progress = False)

        for m in measures:
            experiment_dir = 'data/experiment_results/experiment_simulation_measures_%s_varying_%s' % \
                (m, parameter)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            if parameter == 'number_of_agents':
                experiment_filename = experiment_dir + '/%s_%s_%s_simulation_repeats.pkl' % (str(value), parameter, str(simulation_repeats))
            elif parameter == 'transform_parameters':
                experiment_filename = experiment_dir + '/%s_%s_%s_agents_%s_simulation_repeats.pkl' % (str(value['prob_no_transform']), parameter, str(number_of_agents), str(simulation_repeats))
            else:
                experiment_filename = experiment_dir + '/%s_%s_%s_agents_%s_simulation_repeats.pkl' % (str(value['prob_no_transform']), parameter, str(number_of_agents), str(simulation_repeats))

            with open(experiment_filename, 'wb') as experiment_file:
                pickle.dump(sim.sensors[m], experiment_file)
            print('INFO: Partial results stored in %s' % experiment_filename)

    except Exception as e:
        print(e)
        print("WARNING: Thus failed")

    print('INFO: Experiment run %d (%s = %s, measuring %s) took %s' %
        (i + 1, parameter, value, measures,
        get_time_string(time.process_time() - start_time)))
    print('INFO: End time: %s' % datetime.datetime.now())

def do_case_study_battery_size():

    parameter = 'transform_parameters'
    probs_no_transform = numpy.arange(0.0, 1.2, 0.20)
    # fracs_to_high = numpy.arange(0.0, 1.2, 0.2)
    values = []
    # for p_no_transform in probs_no_transform:
    #     for f_to_high in fracs_to_high:
    #         prob_to_high = ( 1.0 - p_no_transform ) * f_to_high
    #         prob_to_low = ( 1.0 - p_no_transform ) * (1.0 - f_to_high)
    #         values.append({"prob_no_transform": p_no_transform,
    #             "prob_to_low_fev": prob_to_low,
    #             "prob_to_high_fev": prob_to_high})
    for prob in probs_no_transform:
        values.append({"prob_no_transform": prob,
                    "prob_to_low_fev": 0.0,
                    "prob_to_high_fev": 1 - prob})
    # values.append({"prob_no_transform": 1.0,
    #     "prob_to_low_fev": 0.0,
    #     "prob_to_high_fev": 0.0})

    measures = ['simulated_sessions']
    number_of_agents, simulation_repeats = 30, 10

    case_study_experiment(parameter, values, measures,
        simulation_repeats = simulation_repeats,
        number_of_agents = number_of_agents,
        method = 'relMAE')

    return


def main():
    ''' Validation transformation experiment '''
    # parameter = 'transform_parameters'
    # val1 = {"prob_no_transform": 0.0, "prob_to_low_fev": 1.0, "prob_to_high_fev": 0.0 }
    # val2 = {"prob_no_transform": 0.0, "prob_to_low_fev": 0.0, "prob_to_high_fev": 1.0 }
    # values = [val1, val2]
    # measures = ['simulated_agent_sessions']
    # number_of_agents, simulation_repeats = 10, 2
    #
    # experiment_simulation(parameter, values, measures,
    #     simulation_repeats = simulation_repeats,
    #     number_of_agents = number_of_agents,
    #     method = 'relMAE',
    #     hack_parameters_file = 'data/input_parameters/hack_parameters_file.json')

    ''' Case study battery size '''
    # do_case_study_battery_size()

    ''' General preprocess '''
    
    
    # create_unmerged_general_data_and_environment()
    
    #this line above seperates the users and the environment in the raw data
    
    
    #create_general_data_and_environment()
    
    
    

    ''' Create and store all agents '''
    #create_agents(store_IDs = True)
    # parameters = []
    # parameters.append((20, True, 'data/agent_database/new_all_agents_bin20/'))
    # parameters.append((60, True, 'data/agent_database/new_all_agents_bin60/'))
    # parameters.append((120, True, 'data/agent_database/new_all_agents_bin120/'))
    #
    # p = multiprocessing.Pool(MAX_CPUs)
    # p.map(create_agents_multicore, parameters)

    ''' Basic code '''
    #sim = simulation.Simulation("data/input_parameters/parameters.json")
    #sim.repeat_simulation(repeat = 2)
    #sim = simulation.Simulation("data/input_parameters/parameters.json", overwrite_parameters = {'agent_initialization':'load_and_use', 'filepath_agent_database': 'data/agent_database/all_agents_car2go/', 'number_of_agents': 7252})
    #sim.repeat_simulation(repeat = 2)
    #a = sim.agents[random.sample(sim.agents.keys(), 1)[0]]
    #a.visualize()

    ''' Create Car2Go's '''
    # sim = simulation.Simulation("data/input_parameters/parameters_car2go.json")

    ''' Selection Process Data Collection'''
    # number_of_agents = 2549
    # repeats = 5
    # store_selection_process_data(number_of_agents, repeats, approach = 'unweighted')
    # store_selection_process_data(number_of_agents, repeats, approach = 'weighted')
    # store_selection_process_data(number_of_agents, repeats, approach = 'weighted_and_trimmed')

    ''' Incidental Users '''
    # number_of_agents = 6000
    # repeats = 5
    # step = 0.25
    # min_ = 0
    # max_ = 2
    # incidental_agents_experiments(number_of_agents, repeats, step, min_, max_)

    ''' Rollout Strategies '''
    #number_of_agents = 6000
    #simulation_repeats = 1
    #rollout_strategies_repeats = 1
    #percentage_incidental_users = 1
    #number_of_CPs_to_add_min = 0
    #number_of_CPs_to_add_max = 1500
    #number_of_CPs_to_add_step = 100
    #
    #
    #rollout_strategies = ['most_kwh_charged_per_week',
    #    'most_unique_users_per_week',
    #     'most_failed_connection_attempts',
    #    'random']
    #rollout_strategies_experiments(number_of_agents, simulation_repeats, rollout_strategies_repeats,
    #    rollout_strategies, percentage_incidental_users, number_of_CPs_to_add_min,
    #    number_of_CPs_to_add_max, number_of_CPs_to_add_step)


    ''' Random Rollout Strategies '''
    # number_of_agents = 1000
    # simulation_repeats = 1
    # rollout_strategies_repeats = 2
    # percentage_incidental_users = 1
    # number_of_CPs_to_add_min = 100
    # number_of_CPs_to_add_max = 1000
    # number_of_CPs_to_add_step = 100

    # rollout_strategies = ['random']
    # rollout_strategies_experiments(number_of_agents, simulation_repeats, rollout_strategies_repeats,
    #     rollout_strategies, percentage_incidental_users, number_of_CPs_to_add_min,
    #     number_of_CPs_to_add_max, number_of_CPs_to_add_step)


    ''' Incidental Agents Experiments'''
    ''' Experiments in which incidental users are being added, no CPs added'''
    # File agent_test.pkl needed with agents_ids (to speed up simulation initialization)

    # number_of_agents = 1000
    # simulation_repeats = 1
    # min_ = 0.0
    # max_ = 2
    # step = 0.25

    # incidental_agents_experiments(number_of_agents, simulation_repeats, step, min_, max_)


#     ''' 1.1 Experiment initialization:
#         Effects of clustering parameters on clustering metrics '''
# #     #terminal 3
#     clustering = {}

#     min_value, step_value, max_value = 0.1, 0.2, 3
#     values = numpy.arange(min_value, max_value + step_value, step_value)
#     clustering['clustering_birch_threshold'] = values

#     min_value, step_value, max_value = 1.0, 1.0, 20.0
#     values = numpy.arange(min_value, max_value + step_value, step_value)
#     clustering['clustering_lon_lat_scale'] = values

#     min_value, step_value, max_value = 0.01, 0.02, 0.24
#     values = numpy.arange(min_value, max_value + step_value, step_value)
#     clustering['threshold_fraction_sessions'] = values

#     min_value, step_value, max_value = 5, 5, 50
#     values = range(min_value, max_value + step_value, step_value)
#     clustering['minimum_nr_sessions_center'] = values

#     min_value, step_value, max_value = 5, 5, 50
#     values = range(min_value, max_value + step_value, step_value)
#     clustering['minimum_nr_sessions_cs'] = values

#     measures = ['number_of_centers', 'average_number_of_charging_stations',
#         'walking_preparedness', 'maximum_distance', 'time_per_initialization']
#     number_of_agents = 6000
#     for parameter, values in clustering.items():
#         experiment_initialization(parameter, values, measures, experiment_name = "clustering_metrics",
#         number_of_agents = number_of_agents)

#     ''' 1.2 Experiment initialization and simulation:
#         Effect of max gap sessions agent on validation metrics and time metrics '''
#     parameter = 'max_gap_sessions_agent'
#     min_, step_, max_ = 30, 30, 180
#     values = range(min_, max_ + step_, step_)
#     measures = ['agent_validation', 'charging_station_validation',
#       'time_per_simulation', 'time_per_initialization']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "gap_sessions_validation",
#        number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#        method = 'relMAE')

#     ''' 1.3 Experiments initialization and simulation:
#          Effect of bin size dist on validation metrics and time metrics '''
#     parameter = 'bin_size_dist'
#     min_bin_size, bin_size_step, max_bin_size = 10, 10, 360
#     bin_size_range = range(min_bin_size, max_bin_size + bin_size_step, bin_size_step)
#     values = [bin_size for bin_size in bin_size_range if 1440 % bin_size == 0]
#     measures = ['agent_validation', 'charging_station_validation',
#        'time_per_simulation', 'time_per_initialization']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "bin_size_validation",
#         number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#         method = 'relMAE')

#     ''' 1.4 Experiment simulation:
#         Effect of habit probability on validation metrics and time metrics '''
#     parameter = 'habit_probability' # no longer works
#     min_, step_, max_ = 0.0, 0.1, 1
#     values = numpy.arange(min_, max_ + step_, step_)
#     measures = ['charging_station_validation', 'agent_validation',
#        'time_per_simulation']
#     number_of_agents, simulation_repeats = 6000,  5
    
#     experiment_simulation(parameter, values, measures, experiment_name = "habit_probability_validation",
#         simulation_repeats = simulation_repeats,
#         number_of_agents = number_of_agents,
#         method = 'relMAE')

#     ''' 1.5 Experiment simulation:
#         Effect of time retry center on validation metrics and time metrics '''
#     parameter = 'time_retry_center'
#     min_, step_, max_ = 20, 20, 100
#     values = range(min_, max_ + step_, step_)
#     measures = ['charging_station_validation', 'agent_validation',
#        'time_per_simulation', 'selection_process_attempts']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_simulation(parameter, values, measures, experiment_name = "time_retry_center_validation",
#         simulation_repeats = simulation_repeats,
#         number_of_agents = number_of_agents,
#         method = 'relMAE')

#     ''' 1.6 Experiment initialization and simulation:
#         Effect of default walking preparedness metrics and time metrics '''
#     parameter = 'minimum_radius'
#     min_, step_, max_ = 50, 50, 300
#     values = range(min_, max_ + step_, step_)
#     measures = ['agent_validation', 'charging_station_validation',
#       'time_per_simulation', 'time_per_initialization', 'maximum_distance',
#       'walking_preparedness']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_initialization_and_simulation(parameter, values, measures, 
#        experiment_name = "walking_preparedness_time_metrics_validation",
#        number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#        method = 'relMAE')

#     ''' 1.7 Experiments simulation:
#         Effect of warmup period in days on validation metrics and time metrics '''
#     parameter = 'warmup_period_in_days'
#     min_, step_, max_ = 7, 7, 42
#     values = range(min_, max_ + step_, step_)
#     measures = ['charging_station_validation', 'agent_validation',
#        'time_per_simulation']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_simulation(parameter, values, measures, experiment_name = "warmup_period_validation",
#         simulation_repeats = simulation_repeats,
#         number_of_agents = number_of_agents,
#         method = 'relMAE')

#     ''' 1.8 Experiments initialization and simulation:
#         Effect of distance measure on validation metrics and time metrics '''
#     parameter = 'distance_metric'
#     values = ['as_the_crow_flies', 'walking']
#     measures = ['agent_validation', 'charging_station_validation', 'time_per_simulation',
#        'time_per_initialization', 'maximum_distance', 'walking_preparedness']
#     number_of_agents, simulation_repeats = 6000, 5

#     experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "distance_measure_validation",
#         number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#         method = 'relMAE')

#     ''' 1.9 Experiments initialization and simulation:
#         Effect of weighted centers on validation metrics and
#         time metrics and distance and walking preparedness '''
#     parameter = 'weighted_centers'
#     values = [True, False]
#     measures = ['agent_validation', 'charging_station_validation',
#         'time_per_simulation', 'time_per_initialization', 'maximum_distance',
#         'walking_preparedness']
#     number_of_agents, simulation_repeats = 6000, 5
    
#     experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "weighted_centers_validation",
#         number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#         method = 'relMAE')

#     ''' 1.10 Distribution of walking preparedness '''
#     parameter = 'number_of_agents'
#     values = [6000]
#     measures = ['walking_preparedness', 'maximum_distance']
#     simulation_repeats = 5

#     experiment_simulation(parameter, values, measures, experiment_name = "distribution_walking_preparedness",
#         simulation_repeats = simulation_repeats,
#         method = 'relMAE')


#     ''' 1.11 Effect of simulation repeats on validation metrics and time metrics '''
#     parameter = 'simulation_repeats'
#     min_, step_, max_ = 5, 5, 30
#     values = range(min_, max_ + step_, step_)
#     measures = ['charging_station_validation', 'agent_validation', 'time_per_simulation']
#     number_of_agents = 6000

#     experiment_simulation(parameter, values, measures, experiment_name = "simulation_repeats_validation",
#         number_of_agents = number_of_agents, method = 'relMAE')
    
#     '''NEW 2.2 Experiment adding non_habitual users'''
#     parameter = 'adding_non_habitual_agents'
#     values = list(numpy.arange(0,20001,1000))
#     measures = ['walking_preparedness', 'maximum_distance']
#     simulation_repeats = 5
#     number_of_agents = 6000
    
#     experiment_simulation(parameter, values, measures, experiment_name = "adding_non_habitual_agents",
#         number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#         method = 'relMAE')
    
    '''NEW 2.3 Experiment adding habitual users'''
#     parameter = 'adding_habitual_agents'
#     values =range(0,6000,250)
#     measures = ['failed_connection_attempt', 'simulated_sessions','failed_sessions']
#     simulation_repeats = 1
#     initialization_repeats = 1
#     number_of_agents = 6000
    
#     experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "adding_habitual_agents",
#         number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
#         method = 'relMAE', initialization_repeats = initialization_repeats)
    
    '''NEW 2.5 All stress stests Experiment adding extra habitual users'''
    parameter = 'adding_extra_agents'
    values = list(numpy.arange(0,22000,1000,dtype=int))  #list(numpy.arange(14000,20000,2000,dtype=int))
    measures = ['failed_connection_attempt', 'simulated_sessions']
    simulation_repeats = 1
    initialization_repeats = 1
    number_of_agents = 10000 #6000
    list_ratio_unhabitual = [0.05]

    
    for ratio_unhabitual in list_ratio_unhabitual:
        print('running' + str(number_of_agents) + ' agents ' + str(ratio_unhabitual) + " unhabitual ")
        experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "adding_habitual_agents",
                                                 number_of_agents = number_of_agents, simulation_repeats = simulation_repeats,
                                                 method = 'relMAE', initialization_repeats = initialization_repeats,
                                                 ratio_unhabitual = ratio_unhabitual)
    



        
        
    '''NEW 2. All stress stests Experiment with non-habitual users'''
    
#     parameter = 'adding_extra_agents'
#     values = list(numpy.arange(0,1,2,dtype=int))#list(numpy.arange(500,35000,500,dtype=int))
#     measures = ['failed_connection_attempt', 'simulated_sessions']
#     simulation_repeats = 1
#     initialization_repeats = 1
#     list_ratio_unhabitual = [0.05]

#     #important to mention is that the 1 range is for the UPDATED parameter settings
    
#     for number_of_agents in list(numpy.arange(7500,30000,500,dtype=int)): 
#         for ratio_unhabitual in list_ratio_unhabitual:
#             print('running' + str(number_of_agents) + ' agents ' + str(ratio_unhabitual) + " unhabitual ")
#             experiment_initialization_and_simulation(parameter, values, measures, experiment_name = "adding_habitual_agents",
#             number_of_agents = int(number_of_agents), simulation_repeats = simulation_repeats,
#             method = 'relMAE', initialization_repeats = initialization_repeats, ratio_unhabitual = ratio_unhabitual)
    

       
    
    

if __name__ == '__main__':
    main()
