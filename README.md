# SEVA Base Model 
## Last update: 13/03/18

Synopsis
--
Deployment and management of environmental infrastructures, such as charging infrastructure for Electric Vehicles (EV), is a challenging task. For policy makers, it is particularly difficult to estimate the capacity of current deployed public charging infrastructure for a given EV user population. While data analysis of charging data has shown added value for monitoring EV systems, it is not valid to linearly extrapolate charging infrastructure performance when increasing population size.

We developed a data-driven agent-based model that can explore future scenarios to identify non-trivial dynamics that may be caused by EV user interaction, such as competition or collaboration, and that may affect performance metrics. We validated the model by comparing EV user activity patterns in time and space. The model has been used in several scientific contributinos (see below). 

This github contains a python implementation of an agent-based model simulating the charging behavior of Electric Vehicle (EV) users in the Netherlands. This repository contains the base model and several extensions. 

**Prerequisites**
SEVA is a data-driven agent basede model in which the behavior of agents is derived from charging transaction data. The model requires charging data to be trained. This repository contains a subset of the agents that are present in the simulations of [1]. Due to privacy and sensitivity issues, no charging data was allowed to be shared in this repositoty. 

To run the model on new charging data, please refer to the technical documentation found in [2]. This document contains information on data requiremnts.

**Instruction videos**
The following links contain several instruction videos that guide new users through the installation process and the use of the model.


**Recommended follow up steps:**
1. Set all parameters in the ```data/input_parameters/parameters.json``` to the preferred values (or leave them on the default values).
2. Uncomment ```create_unmerged_general_data_and_environment``` in the ```main()``` of ```experiments.py``` and run ```experiments.py``` to create the raw datafile and the environment file.
3. Uncomment ```create_agents(store_IDs = True)``` in the main() of ```experiments.py``` and run to create your agent database.
4. Copy the code from the basic run example version 2 of the simulation (see below) and run this to check if everything is working.
=======

- Go to the modeling_charging_behavior directory.
- (optional) To not have to fill in username and password type: git config --global credential.helper 'cache --timeout=36000'
-  Type git checkout -b your-new-branch-name v1.2 to create your own branch to work on using the base model version 1.2.-
- Commit and push: git commit -am "your commit message" and git push origin your-new-branch-name.
- Put the file with all charge transactions in the directory data/sessions/ and name it ChargesSessions_raw.pkl.


``` python
import simulation
import data_handler
import environment
import agent
import random

sim = simulation.Simulation("data/input_parameters/parameters.json", 
	overwrite_parameters = {'agent_initialization': 'create_and_store', 
				'number_of_agents': 0, 
				'filepath_agent_database': 'data/agent_database/all_agents/', 
				'info_printer': True, 
				'preprocess_info': {'general_preprocess': True, 'city': 'all', 'merge_cs': false} 
	})
```

OR following the next steps:

1. Turn the ```general_preprocess``` parameter in ```data/input_parameters/parameters.json``` to true.
2. Run ```simulation.Simulation("data/input_parameters/parameters.json")``` to do the general preprocessing (which creates ```data/sessions/ChargeSessions_general.pkl```).
3. Put ```general_preprocess``` on false again (revert step 1).
>>>>>>> vincent

- Optionally you could run the following lines to repeat the simulation, however, you should make sure the number_of_agents should be > 0

	``` python
	sim.repeat_simulation()
	a = sim.agents[random.sample(sim.agents.keys(), 1)[0]]
	a.visualize()
	```

### Agent Database creation
- Run the following code tocreate an agent database file for every agent in the dataset:

``` python
import simulation
import data_handler
import environment
import agent
import random

<<<<<<< HEAD
sim = simulation.Simulation("data/input_parameters/parameters.json", overwrite_parameters = {'agent_initialization': 'create_and_use', 'filepath_agent_database': 'agent_database/all_agents/', 'number_of_agents': 10})
sim.repeat_simulation()
a = sim.agents[random.sample(sim.agents.keys(), 1)[0]]
a.visualize()
=======
sim = simulation.Simulation("data/input_parameters/parameters.json",
        overwrite_parameters = {'agent_initialization': 'create_and_store',
        'filepath_agent_database': 'data/agent_database/all_agents/',
        'number_of_agents': 20000,
        'info_printer': True, 'IDs_from_memory': False, 'environment_from_memory': False, \
        'start_date_training_data': "01-01-2014", \
        'end_date_training_data': "01-01-2018", \
        'start_date_test_data': "01-01-2014", \
        'end_date_test_data': "01-01-2018", \
        'bin_size_dist': 20})
if store_IDs:
   with open ('data/experiment_results/all_agent_IDs_all.pkl', 'wb') as agents_file:
   pickle.dump(list(sim.agents.keys()), agents_file)
>>>>>>> vincent
```

OR following the next steps:
 
- Set all parameters in the ```data/input_parameters/parameters.json``` to the preferred values (or leave them on the default values).
- Uncomment ```create_agents(store_IDs = True)``` in the main() of ```experiments.py``` and check if all other experiments are commented.
- Run ```experiments.py``` (this creates your agent database containing agents that satisfy the parameter values that were set).

### Test and Example simulation code

``` python
import simulation
import data_handler
import environment
import agent
import random

sim = simulation.Simulation("data/input_parameters/parameters.json", overwrite_parameters = {'agent_initialization': 'load_and_use', 'filepath_agent_database': 'agent_database/all_agents/', 'number_of_agents': 10})
sim.repeat_simulation()
a = sim.agents[random.sample(sim.agents.keys(), 1)[0]]
a.visualize()
```
Note that for first time usage the ```general_preprocess``` boolean in ```data/input_parameters/parameters.json``` should be on true (and running this might take a while, but it only has to be done once).


### A note on agent creation, storing and loading
In order to speed up the process of running the simulation, we provide the option of storing agents and loading them from memory instead of creating them every time. A directory to save the agents are provided under ```data/agent_database/all_agents```. However note that the validness of agents depends on the input parameters. Thus if any input parameter that influences the validness of the agent is adjusted, it is recommended to create a new agent database. The same applies for the ```data/experiment_results/all_agent_IDs.pkl``` file which is created using the ```create_agents(store_IDs = True)``` function of ```experiments.py```.

## File dependencies
* ```RFID_DETAILS.csv```(csv file): This [file](https://gitlab.computationalscience.nl/ido-laad/modeling_charging_behavior/blob/vincent/data/extra/RFID_DETAILS.csv) is generated in RStudio and consist of UseType data per (RF)ID. This is needed to distinquish habitual users from non-habitual users.
* ```CHIEF_DWH.R``` (R file): This [file](https://gitlab.computationalscience.nl/ido-laad/modeling_charging_behavior/blob/vincent/support_files/CHIEF_DWH.R) consist of methods to access the data stored in the database. Needed in RStudio
* ```EXPORT_DATA.R``` (R file): This [file](https://gitlab.computationalscience.nl/ido-laad/modeling_charging_behavior/blob/vincent/support_files/EXPORT_DATA.R) consist of methods to export the data by means of a csv file per needed dataset, needed to get the data in the right format
* ```pre-processing.py``` (R file): This [file](https://gitlab.computationalscience.nl/ido-laad/modeling_charging_behavior/blob/vincent/support_files/pre-processing.py) converts the downloaded csv file with chargesessions and into a ChargeSession_raw.pkl file
* ```full_agent_list.pkl```(pkl): This [file](https://gitlab.computationalscience.nl/ido-laad/modeling_charging_behavior/blob/vincent/support_files/full_agent_list.pkl) consist of a list of  all validated (RF)IDs for Amsterdam, Utrecht, Den Haag, Rotterdam. Validated for training 2014-2017 and test 2017  
* ``` ``` ():





## Parameter description
#### Instructions
The parameters.json file contains all the variable parameters in the simulation. In the next section we summarize all the variables it should contain, their influence on the simulation and their possible values. All parameters indicated with a * can possibly be given to the ```overwrite_parameters``` kwarg of simulation.

#### Description of the parameters
* ```info_printer```* (boolean): Parameter that allows for extra info to be printed. If put on true extra print statements are visible. Recommended value is true.

##### data_handler
* ```path_to_parkingzone_dict``` (str): Path to the directory of the datafile (.pkl) that contains the dictionary linking charging stations to parking zones.
* ```geojson_amsterdam``` (str): Path to the geojson file containing the parking zones of Amsterdam. Or null if this data is unavailable.
* ```geojson_den_haag``` (str): Path to the geojson file containing the parking zones of Den Haag. Or null if this data is unavailable.
* ```geojson_rotterdam``` (str): Path to the geojson file containing the parking zones of Rotterdam. Or null if this data is unavailable.
* ```geojson_utrecht``` (str): Path to the geojson file containing the parking zones of Utrecht. Or null if this data is unavailable.
* ```path_to_data``` (str): Path to the directory of the datafile containing the  data including the first part of the filename. That is, if the path to the file is "data/sessions_raw.pkl", then this path_to_data would be "data/sessions".
* ```start_date_training_data``` (str): Start date ("dd-mm-yyyy") of which we want to use training data. Data before this date will be filtered out.
* ```end_date_training_data``` (str): End date ("dd-mm-yyyy") of which we want to use training data. Data after this date will be filtered out.
* ```start_date_test_data``` (str): Start date ("dd-mm-yyyy") of which we want to use test data. Data before this date will be filtered out.
* ```end_date_test_data``` (str): End date ("dd-mm-yyyy") of which we want to use test data. Data after this date will be filtered out.
* ```max_gap_sessions_agent```* (int): Amount of days the maximum allowed gap between the sessions of an agent is. If a gap is found, all sessions before this gap will not be considered. Recommended value is 90.
* ```clustering_lon_lat_scale```* (float): Scaling parameter for the clustering algorithm. A higher value will cause the distance to have more importance. Recommended value is 8.0.
* ```clustering_lon_shift```* (float): Shifting the longitude with this value such that the longitude and latitude are both in the same range of values. Recommended value is 47.4.
* ```clustering_birch_threshold```* (float): Threshold parameter of the sklearn Birch algorithm. Recommended value is 1.5.
* ```minimum_nr_sessions_cs```* (int): The minimum number of sessions a charging station needs to have in order for it to be considered in the clustering algorithm. Recommended value is 10.
* ```minimum_nr_sessions_center```* (int): The minimum number of sessions a center needs to have in order for it to be considered a center. Recommended value is 20.
* ```bin_size_dist```* (int): Parameter that determines the bin size in minutes that is used in the simulation. Recommended value is 20.
* ```threshold_fraction_sessions```* (float): The minimum fraction of the agent's total sessions each center should have. Recommended value is 0.08.
* ```weighted_centers```* (bool): A boolean determining whether the location of the centers is weighted with the number of charge transactions at each charging station or not. Recommended value is true.
* ```preprocess_info``` (Dict): Information needed for preprocessing containing the following keys and values:
  * ```general_preprocess``` (bool): Determines whether to do the full general preprocessing of the data or not. This should be done the first time using the simulation, but otherwise is not needed anymore unless the data changes.
  * ```city``` (str): Offers the possibility to filter out one city from the dataset. If value "all" no cities are filtered. If value is not "all" it should match a value in the "city" column of the general dataset. Recommended value is "all".
  * ```merge_cs``` (boolean): If true all charging stations that share a longitude and latitude value are merged under a single location key. Recommended value is true.

##### agent
* ```selection_process```* (float): The selection process that the simulation should use. Options are "habit_distance" and "choice_model". With the option "habit_distance" the selection process makes agents select their charging station based on either habit or distance. With the option "choice_model" the charging stations are selected based on the logistic regression model which results in a probability for each charging station.
* ```selection_process_parameters``` (Dict): The parameters fitting the selection process should be supplied here. With selection process "habit_distance" the dictionary should contain the key "habit_probability" with a float value. This is the probability with which the agents select their charging station based on habit. With 1 - habit_probability the agents will select their charging station based on distance. With the selection process "choice_model" the dictionary should contain the a dictionary with keys "Amsterdam", "The Hague", "Rotterdam" and "Utrecht". This dictionary should contain the keys "intercept", "distance", "charging_speed", "charging_fee" and "parking_fee" with the logit model coefficient values as values.
* ```time_retry_center```* (int): The number of minutes an agent waits until it tries to find a charging station again if it encountered all possible charging stations to be occupied. Recommended value is 20.
* ```minimum_radius```* (int): The radius in which an agent will consider charging stations if it selects a charging station based on distance has an under limit of this minimum_radius. Recommended value is 150.
* ```transform_parameters```* (Dict[str, float]): Contains information about which fractions of the (phev) population should be transformed to either low battery fev or high battery fev agents. The keys are```frac_no_transform```, ```frac_to_low_fev``` and ```frac_to_high_fev```.
* ```skip_high_fev_agents```* (bool): Determines whether or not to use agents with a high fev battery car type. Default is False.
* ```skip_low_fev_agents```* (bool): Determines whether or not to use agents with a low fev battery car type. Default is False.
* ```skip_phev_agents```* (bool):Determines whether or not to use agents with a phev battery car type. Default is False.

##### simulation
* ```stop_condition``` (str): Type of stop condition the simulation should use. Options are "time" and "nr_activities_executed_per_agent". With the option "time" the simulation stops after a certain maximum time. With the option "nr_activities_executed_per_agent" the simulation stops after a specified number of activities per agent have been executed.
* ```stop_condition_parameters``` (Dict): The parameters fitting the stop condition should be supplied here. With stop condition "time" the dictionary should contain the key "max_time" with a string value of the form "dd-mm-yyyy". With the stop condition "nr_activities_executed_per_agent" the dictionary should contain the key "min_nr_activities_executed_per_agent" with an int value specifying the minimum number of activities.
* ```warmup_period_in_days```* (int): The number of days of the warmup period of the simulation. For validation and visualization the warmup period isn't taken into consideration. Recommended value is 7.
* ```start_time``` (str): Start date of the simulation in the form "dd-mm-yyyy".
* ```agent_creation_method``` (str): Method to use for creating agents in the simulation. Possible options are "random", "given" or "previous". With the "random" option a specified number (in the parameter "nr_of_agents") of random agents available in the dataset are created. With the "given" option the by the user specified RFIDs in the parameter agent_IDs are used to create agents. With the "previous" option the same RFIDs as in the previous run are used.
* ```number_of_agents```* (int): Amount of random agents to create when the use_RFIDs parameter is set to "random".
* ```agent_IDs``` (List): List of strings where each string is the RFID of an agent in the dataset. An example is ["ABCD123", "EFGH456", "IJKL789"].
* ```agent_initialization```* (str): The method of initialization for the agents. Possible options are "create", with which agents are only created, "create_and_use", with which agents are created and can be used for simulation, "create_and_store", with which agents are created and stored in the agent database and "load_and_use", with which agents are loaded from the agent database and can be used for simulation.
* ```filepath_agent_database```* (str): The path to the agent database.
* ```IDs_from_memory```* (bool): Determines whether the IDs of the agent are loaded from memory (when using a "create*") option for agent_initialization). To load IDs from memory there should be a file containing the IDs of valid agents (which can be created using experiments.py).
* ```environment_from_memory``` (bool): Determines whether the environment is read from memory or not.
* ```distance_metric```* (str): The distance metric to determine the distance between centers and charging stations. Possible options are "walking" and "as_the_crow_flies". Recommended value is "walking".

#### Example file
``` json
{
    "info_printer": true,
    "data_handler": {
        "path_to_parking_zone_dict": "../data/real/parking_zone_dict.pkl",
        "geojson_amsterdam": "../data/real/parkeergebied_amsterdam.geojson",
        "geojson_den_haag": "../data/real/parkeergebied_den_haag.geojson",
        "geojson_rotterdam": null,
        "geojson_utrecht": null,
        "path_to_real_file": "../data/real/ChargeSessions",
        "start_date_training_data": "01-01-2014",
        "end_date_training_data": "01-01-2016",
        "start_date_test_data": "01-01-2016",
        "end_date_test_data": "01-01-2017",
        "max_gap_sessions_agent": 90,
        "clustering_lon_lat_scale": 8.0,
        "clustering_lon_shift": 47.4,
        "clustering_birch_threshold": 1.5,
        "clustering_birch_branching_factor": 50,
        "minimum_nr_sessions_cs": 10,
        "minimum_nr_sessions_center": 20,
        "threshold_fraction_sessions": 0.08,
        "bin_size_dist": 20,
        "weighted_centers": true,
        "preprocess_info": {
            "general_preprocess": false,
            "city": "all",
            "merge_cs": true
        }
    },
    "agent": {
        "habit_probability": 0.4,
        "time_retry_center": 20,
        "minimum_radius": 150
    },
    "simulation": {
        "stop_condition": "time",
        "stop_condition_parameters": {"max_time": "01-01-2017"},
        "warmup_period_in_days": 7,
        "start_date_simulation": "01-01-2016",
        "agent_creation_method": "random",
        "number_of_agents": 100,
        "agent_IDs": [],
        "agent_initialization": "",
        "filepath_agent_database": "",
        "IDs_from_memory": true,
        "environment_from_memory": true,
        "distance_metric": "walking"
    }
}
```

Built With
--
- [Keras](keras.io) - Used to build the neural network structure
- [TensoFlow](tensorflow.org) - Keras backend


Contributors
--
Jurjen Helmus - J.R.Helmus@hva.nl [corresponding contributor]

Igna Vermeulen - ignavermeulen@hotmail.com

Seyla Wachlin - seylawachlin@gmail.com

Vincent Gorka - vcgorka@me.com

Alexander Easton - Easton.ae@gmail.com

Questions and bug reports
--
Any questions on the usage of the model or bug reports can be send to any of the contributors.
