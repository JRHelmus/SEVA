'''
data_handler.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Last updated on May 2017
'''

import pandas
import numpy
import sys
import os
import random
import sklearn.cluster
import datetime
import pickle
import time
import datetime
import parking_zones

class DataHandler():
    ''' The DataHandler class deals with creating and manipulating
        distributions and processing data.

    Args:
        parameters (Dict[str, Any]): The DataHandler parameters.
        info_printer (bool): Parameter to decide whether to print information
            about run times.
        overwrite_parameters (Dict[str, Any]): Parameter values to use instead
            of the values specified in the input file. Often used for
            experiments. Parameters that can be overwritten can be found in the
            readme.

    Attributes:
        info_printer (bool): Parameter to decide whether to print information
            about run times.
        path_to_data (str): The path to the file containing the real data.
        start_date_training_data (DateTime): The start date (inclusive) of
            which we want to use training data.
        end_date_training_data (DateTime): The end date (inclusive) of which
            we want to use training data.
        start_date_test_data (DateTime): The start date (inclusive) of which
            we want to use test data.
        end_date_test_data (DateTime): The end date (inclusive) of which we
            want to use test data.
        clustering_lon_lat_scale (float): Scaling parameter for the clustering
            algorithm. A higher value will cause the distance to have more
            importance.
        clustering_lon_shift (float): Shifting value of the longitude compared
            to latitude for the clustering algorithm. This should make sure
            the longtitude and latitude are in the same range of values.
        clustering_birch_threshold (float): Threshold parameter for the input
            of the sklearn Birch clustering method.
        minimum_nr_sessions_cs (int): Number of sessions that a charging station
            has to have in order to be considered for clustering.
        minimum_nr_sessions_center (int): Minimum number of sessions of a center.
        bin_size_dist (int): The number of minutes contained in each bin in the
            simulation. This determines the amount of bins in the distributions.
        training_data (DataFrame): The training data. This data will
            contain the columns 'location_key', 'amount_of_sockets','ID',
            'start_connection', 'end_connection', 'kWh', 'city', 'parking_zone'
            'region_abbreviation', 'provider', 'address', 'postal_code',
            'district', 'subdistrict', 'latitude' and 'longitude'.
        test_data (DataFrame): The test data. This data will
            contain the columns 'location_key', 'amount_of_sockets','ID',
            'start_connection', 'end_connection', 'kWh', 'city', 'parking_zone'
            'region_abbreviation', 'provider', 'address', 'postal_code',
            'district', 'subdistrict', 'latitude' and 'longitude'.
        offset (DateTime): An offset used to convert TimeDelta object to
            DateTime objects.
    '''

    def __init__(self, parameters, overwrite_parameters, info_printer):
        self.info_printer = info_printer
        if not self._load_and_check_attribute_parameters(parameters,
            overwrite_parameters):
            sys.exit()

        if not self._check_preprocess_parameters(parameters['preprocess_info'],
            overwrite_parameters):
            sys.exit()
        if parameters['preprocess_info']['general_preprocess'] or (
            'general_preprocess' in overwrite_parameters and
            overwrite_parameters['general_preprocess']):
            self._preprocess_general(parameters['preprocess_info'],
                parameters['path_to_parking_zone_dict'], parameters) #NOTE: might be nicer to just give parameters
        self._preprocess_detailed(parameters['preprocess_info'])

        self.offset = pandas.to_datetime('01-01-2000', format='%d-%m-%Y')

        self.car_type_data = pandas.read_csv('data/extra/RFID_DETAILS.csv')


    def _load_and_check_attribute_parameters(self, parameters, overwrite_parameters):
        ''' This method loads the parameters into the attributes of the class.
            Furthermore it checks whether the parameters contain valid values.

        Args:
            parameters (dict): The parameters to load.

        Updates:
            path_to_data
            start_date_training_data
            end_date_training_data
            start_date_test_data
            end_date_test_data
            max_gap_sessions_agent
            clustering_lon_lat_scale
            clustering_lon_shift
            clustering_birch_threshold
            minimum_nr_sessions_cs
            minimum_nr_sessions_center
            threshold_fraction_sessions
            bin_size_dist

        Returns:
            (bool): Returns True if all parameters are valid.
        '''

        if not os.path.isfile(parameters['path_to_data'] + '_raw.pkl'):
            print('ERROR: path_to_data (%s) + \'_raw.pkl\''  %
                parameters['path_to_data'] + 'does not exist.')
            return False
        if 'general_preprocess' in overwrite_parameters:
            if (not overwrite_parameters['general_preprocess']) and \
                not os.path.isfile(parameters['path_to_data'] + '_general.pkl'):
                print('ERROR: path_to_data (%s) + \'_general.pkl\''  %
                parameters['path_to_data'] + 'does not exist.')
                return False
        
        else:
            if not parameters['preprocess_info']['general_preprocess'] and \
            not os.path.isfile(parameters['path_to_data'] + '_general.pkl'):
                print('ERROR: path_to_data (%s) + \'_general.pkl\''  %
                parameters['path_to_data'] + 'does not exist.')
                return False
        self.path_to_data = parameters['path_to_data']

        if 'path_to_parking_zone_dict' not in parameters:
            print('ERROR: path_to_parking_zone_dict is not specified.')
            return False
        if not os.path.isfile(parameters['path_to_parking_zone_dict']):
            if self.info_printer:
                print('INFO: No parking zone dict yet, creating a new one.')
            self._create_parking_zone_dict(parameters)

        if 'start_date_training_data' in overwrite_parameters:
            self.start_date_training_data = pandas.to_datetime(
            overwrite_parameters['start_date_training_data'],
            format = '%d-%m-%Y', errors = 'coerce')
        else:
            self.start_date_training_data = pandas.to_datetime(
                parameters['start_date_training_data'], format = '%d-%m-%Y',
                errors = 'coerce')
        if not isinstance(self.start_date_training_data, pandas.Timestamp):
            print('ERROR: start_date_training_data (%s) is not a DateTime.' %
                self.start_date_training_data)
            return False
        if 'end_date_training_data' in overwrite_parameters:
            self.end_date_training_data = pandas.to_datetime(
                overwrite_parameters['end_date_training_data'],
                format = '%d-%m-%Y', errors = 'coerce')
        else:
            self.end_date_training_data = pandas.to_datetime(
                parameters['end_date_training_data'], format = '%d-%m-%Y',
                errors = 'coerce')
        if not isinstance(self.end_date_training_data, pandas.Timestamp):
            print('ERROR: end_date_training_data (%s) is not a DateTime.' %
            parameters['end_date_training_data'])
            return False
        if self.start_date_training_data >= self.end_date_training_data:
            print('ERROR: start_date_training_data (%s) does not come before ' %
                self.start_date_training_data + 'end_date_training_data (%s).' %
                self.end_date_training_data)
            return False

        if 'start_date_test_data' in overwrite_parameters:
            self.start_date_test_data = pandas.to_datetime(
                overwrite_parameters['start_date_test_data'],
                format = '%d-%m-%Y', errors = 'coerce')
        else:
            self.start_date_test_data = pandas.to_datetime(
                parameters['start_date_test_data'], format = '%d-%m-%Y',
                errors = 'coerce')
        if 'end_date_test_data' in overwrite_parameters:
            self.end_date_test_data = pandas.to_datetime(
                overwrite_parameters['end_date_test_data'],
                format = '%d-%m-%Y', errors = 'coerce')
        else:
            self.end_date_test_data = pandas.to_datetime(
                parameters['end_date_test_data'], format = '%d-%m-%Y',
                errors = 'coerce')
        if not isinstance(self.end_date_test_data, pandas.Timestamp):
            print('ERROR: end_date_test_data (%s) is not a DateTime.' %
            parameters['end_date_test_data'])
            return False
        if self.start_date_test_data >= self.end_date_test_data:
            print('ERROR: start_date_test_data (%s) does not come before ' %
                self.start_date_test_data + 'end_date_test_data (%s).' %
                self.end_date_test_data)
            return False
        if self.end_date_training_data > self.start_date_test_data:
            print('WARNING: start_date_test_data (%s) does not come after ' %
                self.start_date_test_data + 'end_date_training_data (%s).' %
                self.end_date_training_data)
            
#         if 'environment_check_sockets_date' in overwrite_parameters:
#             self.environment_check_sockets_date = pandas.to_datetime(
#                 overwrite_parameters['environment_check_sockets_date'],
#                 format = '%d-%m-%Y', errors = 'coerce')
#         else:
#             self.environment_check_sockets_date = pandas.to_datetime(
#                 parameters['environment_check_sockets_date'], format = '%d-%m-%Y',
#                 errors = 'coerce')
#         if not isinstance(self.environment_check_sockets_date, pandas.Timestamp):
#             print('ERROR: environment_check_sockets_date (%s) is not a DateTime.' %
#             parameters['environment_check_sockets_date'])
#             return False


        if 'max_gap_sessions_agent' in overwrite_parameters:
            self.max_gap_sessions_agent = overwrite_parameters['max_gap_sessions_agent']
        else:
            self.max_gap_sessions_agent = parameters['max_gap_sessions_agent']
        if not isinstance(self.max_gap_sessions_agent, int):
            print('ERROR: max_gap_sessions_agent (%s) should be an int.' %
                self.max_gap_sessions_agent)
            return False
        if self.max_gap_sessions_agent < 0:
            print('ERROR: max_gap_sessions_agent (%s) should be non negative.' %
                self.max_gap_sessions_agent)
            return False
        self.max_gap_sessions_agent = pandas.to_timedelta('%d days' %
            self.max_gap_sessions_agent)

        if 'clustering_lon_lat_scale' in overwrite_parameters:
            self.clustering_lon_lat_scale = overwrite_parameters['clustering_lon_lat_scale']
        else:
            self.clustering_lon_lat_scale = parameters['clustering_lon_lat_scale']
        if not isinstance(self.clustering_lon_lat_scale, float):
            print('ERROR: clustering_lon_lat_scale (%s) is not a float.' %
                str(self.clustering_lon_lat_scale))
            return False

        if not isinstance(parameters['clustering_lon_shift'], float):
            print('ERROR: clustering_lon_shift (%s) is not a float.' %
                str(parameters['clustering_lon_shift']))
            return False
        self.clustering_lon_shift = parameters['clustering_lon_shift']

        if 'clustering_birch_threshold' in overwrite_parameters:
            self.clustering_birch_threshold = overwrite_parameters['clustering_birch_threshold']
        else:
            self.clustering_birch_threshold = parameters['clustering_birch_threshold']
        if not isinstance(self.clustering_birch_threshold, float):
            print('ERROR: clustering_birch_threshold (%s) is not a float.' %
                str(self.clustering_birch_threshold))
            return False

        if not isinstance(parameters['clustering_birch_branching_factor'], int):
            print('ERROR: clustering_birch_branching_factor (%s) is not an int.' %
                str(parameters['clustering_birch_branching_factor']))
            return False
        self.clustering_birch_branching_factor = parameters['clustering_birch_branching_factor']

        if 'minimum_nr_sessions_center' in overwrite_parameters:
            self.minimum_nr_sessions_center = overwrite_parameters['minimum_nr_sessions_center']
        else:
            self.minimum_nr_sessions_center = parameters['minimum_nr_sessions_center']
        if not isinstance(self.minimum_nr_sessions_center, int):
            print('ERROR: minimum_nr_sessions_center (%s) is not an int.' %
                str(self.minimum_nr_sessions_center))
            return False

        if 'minimum_nr_sessions_cs' in overwrite_parameters:
            self.minimum_nr_sessions_cs = overwrite_parameters['minimum_nr_sessions_cs']
        else:
            self.minimum_nr_sessions_cs = parameters['minimum_nr_sessions_cs']
        if not isinstance(self.minimum_nr_sessions_cs, numpy.int):
            print('ERROR: minimum_nr_sessions_cs (%s) is not an int.' %
                str(self.minimum_nr_sessions_cs))
            return False

        if 'threshold_fraction_sessions' in overwrite_parameters:
            self.threshold_fraction_sessions = overwrite_parameters['threshold_fraction_sessions']
        else:
            self.threshold_fraction_sessions = parameters['threshold_fraction_sessions']
        if not isinstance(self.threshold_fraction_sessions, float):
            print('ERROR: threshold_fraction_sessions (%s) is not a float.' %
                str(self.threshold_fraction_sessions))
            return False
        if self.threshold_fraction_sessions < 0 or \
                self.threshold_fraction_sessions > 1:
            print('ERROR: threshold_fraction_sessions (%.2f) is not between 0 and 1.' %
                self.threshold_fraction_sessions)
            return False

        if 'bin_size_dist' in overwrite_parameters:
            self.bin_size_dist = overwrite_parameters['bin_size_dist']
        else:
            self.bin_size_dist = parameters['bin_size_dist']
        if not isinstance(self.bin_size_dist, int):
            print('ERROR: bin_size_dist (%s) is not a int.' %
                str(self.bin_size_dist))
            return False
        if not (1440 % self.bin_size_dist == 0):
            print('ERROR: bin_size_dist (%d) does not divide day into equal ' %
                str(self.bin_size_dist) + 'partitions.')
            return False

        if 'weighted_centers' in overwrite_parameters:
            self.weighted_centers = overwrite_parameters['weighted_centers']
        else:
            self.weighted_centers = parameters['weighted_centers']
        if not isinstance(self.weighted_centers, bool):
            print('ERROR: weighted_centers (%s) is not a boolean.' %
                self.weighted_centers)
            return False

        if 'city' in overwrite_parameters:
            parameters['preprocess_info']['city'] = overwrite_parameters['city']
            print("WARNING: Will only use data from %s." % overwrite_parameters['city'])
        
        if 'preprocess_info' not in parameters:
            print('ERROR: preprocess_info is not in parameters of DataHandler.')
            return False

        return True

    def _create_parking_zone_dict(self, parameters):
        ''' This method checks the parameters concerning the parking zones
            passes them on to the method in parking_zones.py in order for
            it to create the nessecary file.
        '''

        if parameters['geojson_amsterdam'] != None:
            if not isinstance(parameters['geojson_amsterdam'], str):
                print('WARNING: geojson_amsterdam (%s) is not null or a string, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_amsterdam']))
                parameters['geojson_amsterdam'] = None
            if not os.path.isfile(parameters['geojson_amsterdam']):
                print('WARNING: Path to geojson_amsterdam (%s) does not exist, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_amsterdam']))

        if parameters['geojson_den_haag'] != None:
            if not isinstance(parameters['geojson_den_haag'], str):
                print('WARNING: geojson_den_haag (%s) is not null or a string, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_den_haag']))
                parameters['geojson_den_haag'] = None
            if not os.path.isfile(parameters['geojson_den_haag']):
                print('WARNING: Path to geojson_den_haag (%s) does not exist, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_den_haag']))

        if parameters['geojson_rotterdam'] != None:
            if not isinstance(parameters['geojson_rotterdam'], str):
                print('WARNING: geojson_rotterdam (%s) is not null or a string, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_rotterdam']))
                parameters['geojson_rotterdam'] = None
            if not os.path.isfile(parameters['geojson_rotterdam']):
                print('WARNING: Path to geojson_rotterdam (%s) does not exist, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_rotterdam']))

        if parameters['geojson_utrecht'] != None:
            if not isinstance(parameters['geojson_utrecht'], str):
                print('WARNING: geojson_utrecht (%s) is not null or a string, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_utrecht']))
                parameters['geojson_utrecht'] = None
            if not os.path.isfile(parameters['geojson_utrecht']):
                print('WARNING: Path to geojson_utrecht (%s) does not exist, \
                    it will not be used to create the parking zone dictionary.' %
                    str(parameters['geojson_utrecht']))

        parking_zones.create_parking_zones_dict(self.path_to_data + '_raw.pkl',
            parameters['path_to_parking_zone_dict'],
            geojson_amsterdam = parameters['geojson_amsterdam'],
            geojson_den_haag = parameters['geojson_den_haag'],
            geojson_rotterdam = parameters['geojson_rotterdam'],
            geojson_utrecht = parameters['geojson_utrecht'])

    def _check_preprocess_parameters(self, parameters, overwrite_parameters):
        ''' This method checks whether the parameters needed for preprocessing
             contain valid values.

        Args:
            parameters (Dict[str, Any]): The parameters for preprocessing.

        Returns:
            (bool): True if all parameters are valid.
        '''

        if 'general_preprocess' not in parameters:
            print('ERROR: general_preprocess does not exist in preprocess parameters.')
            return False
        if not isinstance(parameters['general_preprocess'], bool):
            print('ERROR: general_preprocess (%s) is not a boolean.' %
                parameters['general_preprocess'])
            return False
        if 'general_preprocess' in overwrite_parameters:
            if not isinstance(overwrite_parameters['general_preprocess'], bool):
                print('ERROR: general_preprocess (%s) is not a boolean.' %
                    parameters['general_preprocess'])
                return False

        if 'city' not in parameters:
            print('ERROR: city does not exist in preprocess parameters.')
            return False
        if not isinstance(parameters['city'], str):
            print('ERROR: city (%s) not a string.' % parameters['city'])
            return False

        if 'merge_cs' not in parameters:
            print('ERROR: merge_cs does not exist in preprocess parameters.')
            return False
        if not isinstance(parameters['merge_cs'], bool):
            print('ERROR: merge_cs (%s) is not a boolean.' % parameters['merge_cs'])
            return False

        if 'general_preprocess' not in parameters:
            print('ERROR: general_preprocess does not exist in preprocess parameters.')
            return False
        if not isinstance(parameters['general_preprocess'], bool):
            print('ERROR: general_preprocess (%s) is not a boolean.' %
                str(parameters['general_preprocess']))
            return False

        return True

    def get_sessions(self, data_type):
        ''' This method creates the data variable by loading in the data
            from the file specified in the path_to_file.

        Args:
            data_type (str): Type of data to load sessions from. Path to data
                will be set on path_to_data of DataHandler + '_' +
                data_type + '.pkl'. Examples: 'raw' or 'general'.

        Returns:
            data (DataFrame): Containing the sessions in the specified file.
        '''

        path_to_file = self.path_to_data + '_' + data_type + '.pkl'
        if not os.path.isfile(path_to_file):
            print('ERROR: get_sessions of DataHandler got invalid input (%s). ' %
                str(data_type) + 'Path %s does not exist.' %str(path_to_file))
            sys.exit()

        if self.info_printer:
            begin = time.process_time()

        with open(path_to_file, 'rb') as data_file:
            data = pickle.load(data_file)

        if 'start_connection' in data.columns:
            data['start_connection'] = pandas.to_datetime(data['start_connection'])
            data['end_connection'] = pandas.to_datetime(data['end_connection'])
        else:
            data['StartConnectionDateTime'] = pandas.to_datetime(data['StartConnectionDateTime'])
            data['EndConnectionDateTime'] = pandas.to_datetime(data['EndConnectionDateTime'])
        if self.info_printer:
            print('\tINFO: Loading sessions took %s' %
                self.get_time_string(time.process_time() - begin))

        return data

    def _preprocess_general(self, parameters, path_to_parking_zone_dict,
        parameters_for_check):
        ''' This method loads the raw charging sessions then preprocesses that
            data in the following way:
            - Drop entries that have NaN coordinates.
            - Removes entries that do not have a valid ID.
            - Make the column amount_of_sockets based on the ChargePoint_ID column.
            - Drop the columns 'chargesession_skey', 'ChargePoint_ID', 'Socket_ID'
              and 'Status'.
            - Give columns consistant names.
            - Replace inconsistant city names.
            - Merge charging stations with the same location.
            And finally stores the general data to memory.

        Args:
            parameters (dict[str, Any]): Contains the keys:
                'city' (str): Contains the name of the city in which users
                    should be taken, use 'all' to get all the data.
                'merge_cs' (bool): Indicates whether to merge charging stations
                    which have the same longitude, latitude coordinates.
            path_to_parking_zone_dict (str): Path at which the parking zone
                dictionary is stored.
            #NOTE parameters as extra arg
        '''

        if self.info_printer:
            begin = time.process_time()
            print('INFO: Starting general preprocess at %s.' % datetime.datetime.now())

        raw_data = self.get_sessions('raw')

        with open(path_to_parking_zone_dict, 'rb') as parking_zone_file:
            parking_zone_dict = pickle.load(parking_zone_file)
        if self.info_printer:
            print('\tINFO: parking_zone_dict loaded at %s.' % datetime.datetime.now())

        check = self._check_raw_data(raw_data, parking_zone_dict, parameters_for_check)
        if not check[0]:
            sys.exit()
        if not check[1]:
            with open(path_to_parking_zone_dict, 'rb') as parking_zone_file:
                parking_zone_dict = pickle.load(parking_zone_file)
            if self.info_printer:
                print('\tINFO: parking_zone_dict loaded at %s.' % datetime.datetime.now())

        raw_data = raw_data[~numpy.isnan(raw_data['Longitude'])]
        raw_data = raw_data[raw_data.RFID != '-1']

        raw_data['FirstActiveDateTime'] = pandas.to_datetime(raw_data['FirstActiveDateTime'],
                                                             format = '%Y-%m-%d %H:%M:%S', errors = 'raise')
        raw_data['LastActiveDateTime'] = pandas.to_datetime(raw_data['LastActiveDateTime'],
                                                             format = '%Y-%m-%d %H:%M:%S', errors = 'raise')
        
        
        if self.info_printer:
            print('\tINFO: amount_of_sockets added to raw_data at %s.' % datetime.datetime.now())

        raw_data['parking_zone'] = [parking_zone_dict[cs] for cs in raw_data.Location_skey ]
        if self.info_printer:
            print('\tINFO: Parking zones added to raw_data at %s.' % datetime.datetime.now())

        raw_data = raw_data.rename(index = str, columns = {'Location_skey': 'location_key',
            'RFID': 'ID', 'StartConnectionDateTime': 'start_connection',
            'EndConnectionDateTime': 'end_connection',
            'ConnectionTimeHours': 'connection_time_hours', 'kWh': 'kWh',
            'City': 'city', 'Region': 'region_abbreviation',
            'Provider': 'provider', 'Address': 'address',
            'PostalCode': 'postal_code', 'District': 'district',
            'SubDistrict': 'subdistrict', 'SubSubDistrict': 'subsubdistrict', 
            'Latitude': 'latitude', 'Longitude': 'longitude',
            'NumberOfSockets': 'amount_of_sockets'})
        
        columns_general_data = ['location_key', 'ChargePoint_ID', 'ID', 'start_connection',
            'end_connection', 'connection_time_hours', 'kWh', 'city',
            'region_abbreviation', 'provider', 'address', 'postal_code',
            'district', 'subdistrict', 'subsubdistrict', 'latitude', 'longitude',
            'amount_of_sockets', 'parking_zone', 'UseType', 'FirstActiveDateTime',
            'LastActiveDateTime', 'IsFastCharger', 'IsChargeHub', 'status']
        columns_not_in_raw_data = []
        for column in raw_data.columns:
            if column not in columns_general_data:
                columns_not_in_raw_data.append(column)
                
                if self.info_printer:
                    print('\tINFO: Will drop column %s from raw data.' %column)
        raw_data = raw_data.drop(columns_not_in_raw_data, 1)

#         if parameters['merge_cs']:
#             if self.info_printer:
#                 print('INFO: Merging charging stations at %s.' % datetime.datetime.now())
#             location_key_converter = raw_data.groupby(
#                 ['longitude','latitude'])['location_key'].unique().reset_index()
#             for row in location_key_converter.itertuples():
#                 if len(row.location_key) > 1:
#                     for location_key in row.location_key:
#                         raw_data.loc[raw_data.location_key == location_key,
#                             'location_key'] = str(row.location_key)
# #                         raw_data.loc[raw_data.location_key == location_key,
# #                             'FirstActiveDateTime'] = str(row.FirstActiveDateTime)
        raw_data['location_key'] = raw_data['location_key'].astype(str)
        
        raw_data['ID'] = raw_data['ID'].astype(str)

        '''
        def process_ids(row):
            if row['UseType'] == 'Car2GO':
                new_id = 'car2go'
                return new_id
            else:
                new_id = 'car_' + row.ID
                return new_id
        '''
        def process_ids(row):
            new_id = 'car_' + row.ID
            return new_id

        raw_data['ID'] = raw_data['ID'].apply(lambda ID: 'car_' + ID)

        raw_data.to_pickle(self.path_to_data + '_general.pkl')

        if self.info_printer:
            print('INFO: General preprocessing took %s. End time is %s' %
                (self.get_time_string(time.process_time() - begin),  datetime.datetime.now()))

    def _check_raw_data(self, raw_data, parking_zone_dict, parameters):
        #NOTE: needs description
        required_columns_raw_data = ['Location_skey', 'RFID',
            'StartConnectionDateTime', 'EndConnectionDateTime',
            'ConnectionTimeHours', 'kWh', 'City', 'Region', 'Provider',
            'Address', 'PostalCode', 'District', 'SubDistrict', 'Latitude',
            'Longitude']
        for column in required_columns_raw_data:
            if column not in raw_data.columns:
                print('ERROR: Required column %s not found in raw data.' %column)
                return [False, None]

        for cs in raw_data['Location_skey'].unique():
            if cs not in parking_zone_dict.keys():
                if self.info_printer:
                    print('INFO: Parking zone dict not in most recent version, ' +
                        'creating new parking zone dict.')
                self._create_parking_zone_dict(parameters)
                return [True, False]

        return [True, True]

    def _preprocess_detailed(self, parameters):
        ''' This method loads the general charging sessions and preprocesses
            that data in the following way:
            - Filter the data such that only sessions between the requested
              start date and end date are left (these values are different
              for training and test data).
            - Drop entries that are not in 'city'.
            Finally it stores the preprocessed data in its training_data and
            test_data attributes.

        Args:
            parameters (dict[str, Any]): Contains the keys:
                'city' (str): Contains the name of the city in which users
                    should be taken, use 'all' to get all the data.
                'merge_cs' (bool): Indicates whether to merge charging stations
                    which have the same longitude, latitude coordinates.

        Updates:
            training_data
            test_data
        '''

        if self.info_printer:
            begin = time.process_time()

        general_data = self.get_sessions('general')

        self.training_data = general_data.copy()
        self.training_data = self.training_data[(self.training_data['start_connection'] >=
            self.start_date_training_data) & (self.training_data['end_connection'] <=
            self.end_date_training_data)]
        if parameters['city'] != 'all':
            self.training_data = self.training_data[
                self.training_data.city == parameters['city']]
        if self.training_data.empty:
            print('ERROR: training_data is empty.')
            sys.exit()

        self.test_data = general_data.copy()
        self.test_data = self.test_data[(self.test_data['start_connection'] >=
            self.start_date_test_data) & (self.test_data['end_connection'] <=
            self.end_date_test_data)]
        if parameters['city'] != 'all':
            self.test_data = self.test_data[self.test_data.city == parameters['city']]
        if self.test_data.empty:
            print('ERROR: test_data is empty.')
            sys.exit()

        if self.info_printer:
            print('\tINFO: Detailed preprocessing took %s' %
                self.get_time_string(time.process_time() - begin))

    def check_gap_agent_sessions(self, sessions_agent):
        ''' This method checks the sessions for gaps of longer than some period
        of time (max_gap_sessions from parameters file). When a gap is detected
        the sessions taking place before this gap will be removed from the
        sessions and from the preprocessed data.

        Args:
            sessions_agent (DataFrame): Sessions of agent to check for gaps.

        Returns:
            sessions_agent (DataFrame): Sessions with sessions before the last
                gap removed.

        Updates:
            training_data
        '''

        sessions_agent = sessions_agent.sort_values(by = 'start_connection', ascending = False)
        shifted_sessions = sessions_agent.copy()
        shifted_sessions.start_connection = shifted_sessions.start_connection.shift(-1)
        shifted_sessions['disconnection_duration'] = shifted_sessions.apply(lambda row: \
            row.end_connection - row.start_connection, axis = 1)
        shifted_sessions = shifted_sessions[shifted_sessions.disconnection_duration != 'NaT']
        try:
            shifted_sessions = shifted_sessions[shifted_sessions.disconnection_duration >= pandas.to_timedelta('0s')]
            max_sessions = shifted_sessions[shifted_sessions.disconnection_duration \
                > self.max_gap_sessions_agent]
        except TypeError as e:
            #NOTE: shifted_sessions.disconnection_duration is [NaT], then failself.
            print('TypeError: %s' %str(e))
            print('Did not check gap for agent %s' %list(sessions_agent.ID)[0])
            print('Something weird with the disconnection_duration?')
            print(list(shifted_sessions.disconnection_duration))
            raise('WARNING: No sessions left after checking for gaps for agent (%s).' %
                list(sessions_agent.ID)[0])
            return sessions_agent

        if len(max_sessions) == 0:
            return sessions_agent
        drop_date = max_sessions['start_connection'].iloc[0]
        self.training_data = self.training_data[
            (self.training_data.ID != sessions_agent.ID.iloc[0]) |
            (self.training_data.start_connection >= drop_date)]
        return sessions_agent[sessions_agent.start_connection >= drop_date]

    def get_centers(self, sessions_agent):
        ''' This method determines the centers based on the sessions of an agent.

        Args:
            sessions_agent (DataFrame): The sessions of the agent.

        Returns:
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center
                (in meters) as value.
            centers_info (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys with their corresponding values a dictionary
                with info about the center:
                - nr_of_sessions_in_center: amount of sessions in center;
                - center_nr: unique number of the center (for visualization);
        '''

        clusters = self._get_clusters(sessions_agent)
        centers_css = {}
        centers_info = {}
        center_nr = 0
        for cluster in clusters:
            nr_of_sessions_in_center = 0
            locations_css = []
            if self.weighted_centers:
                weights = []
            else:
                weights = None
            for cs in cluster:
                nr_of_sessions = len(sessions_agent[
                    sessions_agent['location_key'] == cs])
                nr_of_sessions_in_center += nr_of_sessions
                lon = sessions_agent[sessions_agent['location_key'] ==
                    cs].longitude.values[0]
                lat = sessions_agent[sessions_agent['location_key'] ==
                    cs].latitude.values[0]
                if self.weighted_centers:
                    weights.append(nr_of_sessions)
                locations_css.append((lon, lat))
            if nr_of_sessions_in_center / len(sessions_agent) > \
                    self.threshold_fraction_sessions and \
                    nr_of_sessions_in_center >= self.minimum_nr_sessions_center:
                center = tuple(numpy.average(locations_css, axis = 0, weights = weights))
                centers_css[center] = {}
                centers_css[center]['habit'] = cluster
                centers_css[center]['distance'] = set()
                centers_css[center]['sessions'] = {}
                
                for cs in cluster:
                    nr_of_sessions_per_cs = len(sessions_agent[
                        sessions_agent['location_key'] == cs])
                    centers_css[center]['sessions'][cs] = nr_of_sessions_per_cs
                    
                centers_info[center] = {'nr_of_sessions_in_center':
                    nr_of_sessions_in_center, 'center_nr': center_nr}
                center_nr += 1
            elif False:
                if nr_of_sessions_in_center / len(sessions_agent) <= \
                        self.threshold_fraction_sessions:
                    print("INFO: Fraction of sessions in center (%s) too low." %
                        (nr_of_sessions_in_center / len(sessions_agent)))
                if nr_of_sessions_in_center < self.minimum_nr_sessions_center:
                    print("INFO: Number of sessions (%s) in center too low." %
                        nr_of_sessions_in_center)
        return centers_css, centers_info

    def _get_clusters(self, sessions_agent):
        ''' Creates clusters of the charging stations used by the agent.

        Args:
            sessions_agent (DataFrame): The sessions of the agent.

        Returns:
            clusters (Set[Tuple[str]]): The clusters of the agent, with
                each cluster containing the location keys of the charging
                stations belonging to that cluster.
        '''

        clustering_data_css, infrequent_css = self._get_clustering_data_agent(
            sessions_agent)
        if len(clustering_data_css) > 0:
            matrix = clustering_data_css.to_numpy()
            birch = sklearn.cluster.Birch(
                threshold = self.clustering_birch_threshold,
                n_clusters = None,
                branching_factor = self.clustering_birch_branching_factor)
            birch.fit(matrix)
            results = pandas.DataFrame(data = birch.labels_,
                columns=['cluster'], index = clustering_data_css.index)
            clusters = {tuple(cluster[1].index)
                for cluster in results.groupby('cluster')}
        else:
            clusters = set()
        for cs in infrequent_css:
            clusters.add(cs)
        return clusters

    def _get_clustering_data_agent(self, sessions_agent):
        ''' This method calculates the activity patterns of the agent for each
            of its (frequent) charging stations.

        Args:
            sessions_agent (DataFrame): The sessions of the agent.

        Returns:
            clustering_data_css (DataFrame): Each columns represents the frequency of
                use within a bin of the normalized activity pattern
                (144 columns if bin size of 10 minutes and activity over 1 day)
                and the longtitude and latitude (making 146 columns in total).
                Each row represents a (frequent) charging station used by the
                agent.
            infrequent_css (List): Location keys of the charging stations with
                less than 5 sessions by this agent.
        '''

        css = set(sessions_agent.location_key.unique())
        css_to_cluster = []
        clustering_data_css = []
        infrequent_css = set()
        times_day = pandas.date_range(self.offset,
            self.offset + pandas.to_timedelta('23:59:59'),
             freq = '%dmin' % self.bin_size_dist)

        for cs in css:
            cs_sessions = sessions_agent.loc[sessions_agent['location_key'] == cs]
            if len(cs_sessions) <= self.minimum_nr_sessions_cs:
                infrequent_css.add(tuple([cs]))
                continue
            css_to_cluster.append(cs)
            all_connection_ranges = [list(connection_range)
                for connection_range in set(cs_sessions.apply(lambda session:
                    self._get_connection_range(session), axis=1))]
            
            connection_ranges_norm = []
            for connection_range in all_connection_ranges:
                for date_time in connection_range:
                    connection_ranges_norm.append(date_time.replace(
                        year = self.offset.year, month = self.offset.month,
                        day = self.offset.day))
            activity = pandas.DataFrame({'times': connection_ranges_norm})
            activity.set_index('times', drop = False, inplace = True)
            activity = activity.groupby(
                pandas.Grouper(freq='%dmin' % self.bin_size_dist)).count()

            if(len(activity) < datetime.timedelta(days = 1) / datetime.timedelta(
                minutes = self.bin_size_dist)):
                missing_indices = [time for time in times_day
                    if time not in list(activity.times.index)]
                missing_series = pandas.Series([0]*len(missing_indices),
                    index = missing_indices)
                activity = activity.times.append(missing_series)
                activity = list(activity.sort_index())
            else:
                activity = list(activity.times)
            activity = numpy.asarray(activity)/max(activity)
            # activity /= max(activity)
            activity = list(activity)
            activity.append(self.clustering_lon_lat_scale *
                (pandas.to_timedelta('%dmin' % self.bin_size_dist).seconds // 60) *
                (self.clustering_lon_shift +
                cs_sessions.longitude.values[0]))
            activity.append(self.clustering_lon_lat_scale *
                (pandas.to_timedelta('%dmin' % self.bin_size_dist).seconds // 60) *
                cs_sessions.latitude.values[0])
            clustering_data_css.append(activity)

        if len(clustering_data_css) == 0 :
            return pandas.DataFrame(), infrequent_css
        col_names = ['activity_%d' %i for i in range(len(clustering_data_css[0]) - 2)]
        col_names.append('longitude')
        col_names.append('latitude')
        clustering_data_css = pandas.DataFrame(clustering_data_css, columns = col_names,
            index = css_to_cluster)
        return clustering_data_css, infrequent_css

    def _get_connection_range(self, session):
        ''' This method generates the range of a charging session with the step
            size given in the bin_size_dist variable.

        Args:
            session (Series): The pandas Series from which to get the start and
                end times of the range.

        Returns:
            (DatetimeIndex): Containing Timestamps of the times between the
                start of the session and the end of the session with an
                interval between those times of the bin size of the simulation.
        '''

        range_session = pandas.date_range(session['start_connection'],
                session['end_connection'], freq = '%dmin' % self.bin_size_dist)
        return (time for time in range_session)

    def get_activity_patterns_centers(self, agent_sessions, centers_css):
        ''' This method calculates the activity patterns for each center of
            an agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center (in
                meters) as value.

        Returns:
            activity_patterns (Dict[Tuple[float, float], List[float]]): A
                dictionary of activity patterns with the centers as keys.
                The activity patterns are lists of probabilities per bin with the
                probabilities indicating the chance of being connected to the
                center in the time interval of the bin.
        '''

        dists = {}
        if len(agent_sessions) > 0:
            for center in centers_css.keys():
                cluster_dataframes = [agent_sessions.loc[
                    agent_sessions['location_key'] == cs] for cs in centers_css[center]['habit']]
                cluster_sessions = pandas.concat(cluster_dataframes)
                dists[center] = self.get_activity_pattern(cluster_sessions)
            return dists

        return {center: self.get_activity_pattern(pandas.DataFrame({}))
            for center in centers_css.keys()}

    def get_activity_pattern(self, sessions):
        ''' This method calculates the activity pattern using the given sessions.

        Args:
            sessions (DataFrame): The sessions that will be used to
                create the activity pattern.

        Returns:
            (List[float]): The activity pattern being a list of probabilities
                per bin with the probabilities indicating the chance of being
                connected to the in the time interval of the bin.
        '''

        times_day = pandas.date_range(self.offset,
            self.offset + pandas.to_timedelta('23:59:59'),
            freq = '%dmin' % self.bin_size_dist)

        if len(sessions) == 0:
            df = pandas.DataFrame({'times': times_day})
            df['0'] = pandas.Series([0] * len(times_day))
            df.set_index('times', drop = True, inplace = True)
            return [0] * len(times_day)
           
        
        list_all_dates = [list(dates) for dates in sessions.apply(lambda row:
            self._get_connection_range(row), axis = 1)]
        times = []
        for dates in list_all_dates:
            for date in dates:
                times.append(date.replace(year = self.offset.year,
                    month = self.offset.month, day = self.offset.day))
        df = pandas.DataFrame({'times': times})
        df.set_index('times', drop = False, inplace = True)
        df = df.groupby(pandas.Grouper(freq = '%dmin' % self.bin_size_dist)).count()

        if (len(df) < datetime.timedelta(days = 1) / datetime.timedelta(
            minutes = self.bin_size_dist)):
            missing_indices = [time for time in times_day
                if time not in list(df.times.index)]
            missing_series = pandas.Series([0] * len(missing_indices),
                index = missing_indices)
            return df.times.append(missing_series).sort_index()
        else:
            return df.times

    def get_disconnection_duration_dists(self, agent_sessions):
        ''' This method calculates the disconnection duration distributions for
            an agent. For each bin, this method calculates the disconnection
            duration distribution belonging to that bin.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.

        Returns:
            disconnection_duration_dists (List[DataFrame]): Disconnection
                duration distributions of the agent. Each element of the list
                indicates a bin and that element is a DataFrame containing the
                probabilities of the disconnection being of the length indicated
                by the index (this length lies between the index multiplied
                with the bin size and the index + 1 multiplied by the bin size).
        '''

        bins = pandas.date_range(start = self.offset,
            end = self.offset + pandas.to_timedelta('23:59:59'),
            freq = '%dmin' % self.bin_size_dist)
        sessions = agent_sessions.sort_values(by='start_connection', inplace = False)
        sessions.start_connection = sessions.start_connection.shift(-1)
        sessions['disconnection_duration'] = sessions.apply(lambda row:
            row.start_connection - row.end_connection, axis = 1)
        sessions = sessions[sessions.disconnection_duration != 'NaT']
        sessions = sessions[sessions.disconnection_duration >= pandas.to_timedelta('0s')]
        sessions.disconnection_duration += self.offset

        return [self._get_disconnection_duration_dist(sessions, curr_bin)
            for curr_bin in bins]

    def _get_disconnection_duration_dist(self, sessions, curr_bin):
        ''' This method calculates the disconnection duration distribution for
            one agent at the specified bin.

        Args:
            sessions (DataFrame): The sorted sessions of a agent containing a
                column 'disconnection_duration'.
            curr_bin (Timestamp): The timestamp of the bin of which the
                disconnection duration will be determined.

        Returns:
            (DataFrame): Disconnection duration distribution of the agent. This
                is a DataFrame containing the probabilities of the disconnection
                being of the length indicated by the index (this length lies
                between the index multiplied with the bin size and the index + 1
                multiplied by the bin size).
        '''

        df = sessions.copy()
        df['in_bin'] = df.apply(lambda row: row.end_connection.replace( \
            year = self.offset.year, month = self.offset.month, \
            day = self.offset.day) >= curr_bin and \
            row.end_connection.replace(year = self.offset.year, \
            month = self.offset.month, day = self.offset.day) < \
            curr_bin + pandas.to_timedelta('%dmin' % self.bin_size_dist), axis=1)
        df = df.loc[df.in_bin]

        if not len(df):
            times_day = pandas.date_range(self.offset,
                self.offset + pandas.to_timedelta('23:59:59'),
                freq = '%dmin' % self.bin_size_dist)
            df = pandas.DataFrame({'times': times_day})
            df['0'] = pandas.Series([0] * len(times_day))
            df.set_index('times', drop = True, inplace = True)
        else:
            df.set_index('disconnection_duration', drop = False, inplace = True)
            df = df.groupby(pandas.Grouper(freq = '%dmin' %
                self.bin_size_dist)).count()
            times_day = pandas.date_range(self.offset, df.index[0],
                freq = '%dmin' % self.bin_size_dist)
            missing_begin_indices = [time for time in times_day if time not in
                                     list(df.disconnection_duration.index)]
            missing_begin_series = pandas.Series([0] * len(missing_begin_indices),
                                                 index = missing_begin_indices)
            df = df.disconnection_duration.append(missing_begin_series).sort_index()
        return df

    def get_connection_duration_dists(self, agent_sessions, centers_css):
        ''' This method calculates the connection duration distributions for
            each center of an agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center (in
                meters) as value.

        Returns:
            (Dict[Tuple[float, float], List[DataFrame]]): A dictionary of
                connection duration distributions with centers as keys and the
                distributions as values. The distributions are lists where each
                element of the list indicates a bin and that element is a
                DataFrame containing the probabilities of the connection being
                of the length indicated by the index (this length lies between
                the index multiplied with the bin size and the index + 1
                multiplied by the bin size).
        '''

        bins = pandas.date_range(start = self.offset,
            end = self.offset + pandas.to_timedelta('23:59:59'),
            freq = '%dmin' % self.bin_size_dist)

        return {center: self._get_connection_duration_dists_for_center(agent_sessions,
            centers_css, center, bins) for center in centers_css.keys()}

    def _get_connection_duration_dists_for_center(self, agent_sessions, centers_css,
        center, bins):
        ''' This method calculates the connection duration distributions for
            a specific center of an agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center (in
                meters) as value.
            center (Tuple[float, float]): Longitude and latitude tuple of the
                center.
            bins (DatetimeIndex): Contains the DateTime of the bins.

        Returns:
            (List[DataFrame]): Connection duration distributions of the center.
                Each element of the list indicates a bin and that element is a
                DataFrame containing the probabilities of the connection being
                of the length indicated by the index (this length lies between
                the index multiplied with the bin size and the index + 1
                multiplied by the bin size).
        '''

        center_dataframes = []
        css = centers_css[center]['habit']
        for cs in css:
            center_dataframes.append(agent_sessions.loc[
                agent_sessions['location_key'] == cs])
        sessions_center = pandas.concat(center_dataframes)
        sessions_center['connection_duration'] = sessions_center.apply(
            lambda row: row.end_connection - row.start_connection, axis = 1)
        sessions_center = sessions_center[
            sessions_center.connection_duration >= pandas.to_timedelta('0s')]
        sessions_center.connection_duration += self.offset

        return [self._get_connection_duration_dist(sessions_center, curr_bin)
            for curr_bin in bins]

    def _get_connection_duration_dist(self, sessions, curr_bin):
        ''' This method calculates the connection duration distributions for
            a specific center of an agent at the specified bin.

        Args:
            sessions (DataFrame): Sorted sessions containing a column
                'connection_duration'.
            curr_bin (Timestamp): The timestamp of the bin of which the
                connection duration will be determined.

        Returns:
            connection_duration_dist (DataFrame): Connection duration
                distribution of of the sessions. This is a DataFrame containing
                the probabilities of the connection being of the length indicated
                by the index (this length lies between the index multiplied
                with the bin size and the index + 1 multiplied by the bin size).
        '''
        # print('_get_connection_duration_dist with bin %s at %s' % (curr_bin, datetime.datetime.now()))
        # df = sessions.copy()
        df = sessions
        df['in_bin'] = df.apply(lambda row: row.start_connection.replace(
            year = self.offset.year, month = self.offset.month,
            day = self.offset.day) >= curr_bin and
            row.start_connection.replace(year = self.offset.year,
            month = self.offset.month, day = self.offset.day) < curr_bin +
            pandas.to_timedelta('%dmin' % self.bin_size_dist), axis = 1)

        df_ = df.loc[df.in_bin]
        if not len(df_):
            times_day = pandas.date_range(self.offset,
                self.offset + pandas.to_timedelta('23:59:59'),
                freq = '%dmin' % self.bin_size_dist)
            df_ = pandas.DataFrame({'times': times_day})
            df_['0'] = pandas.Series([0] * len(times_day))
            df_.set_index('times', drop = True, inplace = True)
        else:
            df_.set_index('connection_duration', drop = False, inplace = True)
            df_ = df_.groupby(pandas.Grouper(freq = '%dmin' %
                self.bin_size_dist)).count()
            times_day = pandas.date_range(self.offset, df_.index[0],
                freq = '%dmin' % self.bin_size_dist)
            missing_begin_indices = [time for time in times_day if time not in
                list(df_.connection_duration.index)]
            missing_begin_series = pandas.Series([0] * len(missing_begin_indices),
                index = missing_begin_indices)
            df_ = df_.connection_duration.append(missing_begin_series).sort_index()
        return df_

    def get_arrival_dists(self, agent_sessions, centers_css):
        ''' This method calculates the arrival distribution for each center of
            an agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center (in
                meters) as value.

        Returns:
            arrival_dists (Dict[Tuple[float, float], List[float]]): A dictionary
                of arrival distributions with the centers as keys. The
                distributions are lists of probabilities per bin with the
                probabilities indicating the chance of connecting to the center
                in the time interval of the bin.
        '''

        times_day = pandas.date_range(self.offset,
            self.offset + pandas.to_timedelta('23:59:59'),
             freq = '%dmin' % self.bin_size_dist)

        return {center: self._get_arrival_dist(agent_sessions, centers_css,
            center, times_day) for center in centers_css}

    def _get_arrival_dist(self, agent_sessions, centers_css, center, times_day):
        ''' This method calculates the arrival distribution for a specific
            center of an agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictio
            center (Tuple[float, float]): longitude and latitude tuple of the
                center.
            times_day (DatetimeIndex): containing the DateTime of the bins.

        Returns:
            List[float]]: An arrival distribution, namely a list of
                probabilities per bin with the probabilities indicating the
                chance of connecting to the center in the time interval of the
                bin.
        '''

        cluster_dataframes = []
        css = centers_css[center]['habit']
        for cs in css:
            cluster_dataframes.append(agent_sessions.loc[agent_sessions['location_key'] == cs])
        cluster_sessions = pandas.concat(cluster_dataframes)

        cluster_sessions['start_connection_no_date'] = cluster_sessions.apply(
            lambda row: row.start_connection.replace(year = self.offset.year,
                month = self.offset.month, day = self.offset.day), axis = 1)

        cluster_sessions.set_index('start_connection_no_date', drop = False,
            inplace = True)
        cluster_sessions = cluster_sessions.groupby(pandas.Grouper(freq =
            '%dmin' % self.bin_size_dist)).count()

        if (len(cluster_sessions) < datetime.timedelta(days = 1) /
            datetime.timedelta(minutes = self.bin_size_dist)):
            missing_indices = [time for time in times_day
                if time not in list(cluster_sessions.start_connection_no_date.index)]
            missing_series = pandas.Series([0] * len(missing_indices),
                index = missing_indices)
            return cluster_sessions.start_connection_no_date.append(
                missing_series).sort_index()
        else:
            return cluster_sessions.start_connection_no_date

    def get_preferences(self, agent_sessions, css):
        ''' This method calculates the amount of times each charging station in
            css (charging stations) is used by the agent.

        Args:
            agent_sessions (DataFrame): The sessions of the agent.
            css (List[str]): A list of the location keys of the charging stations.

        Returns:
            preferences (Dict[str, int]): The amount of times each charging
                station is used.
        '''

        grouped_css = agent_sessions.groupby('location_key').count()
        css_in_agent_sessions = agent_sessions['location_key'].unique()
        return {cs: grouped_css.loc[grouped_css.index == cs].values[0][0]
            for cs in css if cs in css_in_agent_sessions}

    def index_dist(self, time):
        ''' This method calculates the index of the given time, using the
            bin size.

        Args:
            time (DateTime): The datetime for which to calculate the index.

        Returns:
            index (int): The index of the datetime.
        '''

        midnight = datetime.datetime(time.year, time.month, time.day)
        return int((time - midnight) / ('%dmin' % self.bin_size_dist))

    def sample(self, p):
        ''' This method samples from an array p of probabilities and couples
            this sample with a duration making use of the bin size.

        Args:
            p (List[float]): Contains the probabilities.

        Returns:
            sampled_time (TimeDelta):
        '''

        times = [datetime.timedelta(0) + (bin_nr + random.random()) *
             datetime.timedelta(minutes = self.bin_size_dist)
             for bin_nr in range(len(p))]
        try:
            #t = datetime.datetime.now()

            p = p / numpy.sum(p)

            # print('1aa took %s' % (datetime.datetime.now() - t))
            #t = datetime.datetime.now()

            sampled_time = numpy.random.choice(times, p = p)
            # print(len(times))
            # print('1ab took %s' % (datetime.datetime.now() - t))


            sampled_time = sampled_time - datetime.timedelta(

                seconds = sampled_time.seconds % 60,
                microseconds = sampled_time.microseconds)
            return sampled_time
        except Exception as e:
#             print('WARNING: Sample empty, thus DataHandler.sample() will ' +
#                 'return the bin size.')
            return datetime.timedelta(minutes = self.bin_size_dist)

    def convert_activity_pattern_data(self, list_activity_patterns_per_center):
        ''' This method converts the list of activity patterns per center to
            the more usable format for a dictionary.

        Args:
            list_activity_patterns_per_center (List[Dict[Tuple[float, float], DataFrame]):
                A list of dictionaries with centers as keys and activity patterns
                as values.

        Returns:
            all_activity_patterns (Dict[Tuple[float, float], List[DataFrame]]):
                A dictionary with centers as keys and a list of activity patterns
                as values.
        '''

        all_activity_patterns = {}
        for activity_pattern_per_center in list_activity_patterns_per_center:
            for center, activity_pattern in activity_pattern_per_center.items():
                if center in all_activity_patterns:
                    if numpy.sum(activity_pattern) != 0:
                        activity_pattern /= numpy.sum(activity_pattern)
                    all_activity_patterns[center].append(activity_pattern)
                else:
                    if numpy.sum(activity_pattern) != 0:
                        activity_pattern /= numpy.sum(activity_pattern)
                    all_activity_patterns[center] = [activity_pattern]

        return all_activity_patterns

    def get_error(self, expected_pattern, simulated_pattern, method = 'relMAE'):
        ''' This method calculates the error between two activity patterns.
            Each bin in the expected activity pattern is compared to the
            simulated activity pattern and the error is calculated. The method
            returns the mean error over all the bins in the patterns. Two types
            of error calculation are supported: the Mean Abolute Error (MEA)
            and the relative MAE. The MAE of a bin is defined as the difference
            between the simulated and real value.  The relative error of a bin
            is defined as the difference between the simulated and real value
            the bin divided by the maximum error of this bin. The maximum error
            is the maximum distance of the real bin to either 0 or 1. The error
            is then (in both cases) multiplied by 100 to get the error range
            between 0 and 100.

        Args:
            expected_pattern (List[float]): The expected activity pattern.
            simulated_pattern (List[float]): The simulated activity pattern.

        Kwargs:
            method (str): Method of validation. 'MAE' for Mean Absolute Error
                or 'relMAE' for relative Mean Absolute Error. Default 'relMAE'.

        Returns:
            (float): The error between the expected and the simulated pattern.
        '''

        # update 2019 JRH old version was not correct needs update 
        
        print('start get error for' + str(self))
        print('expected_pattern length' +str(len(expected_pattern)))
        print('simulated_pattern length' +str(len(simulated_pattern)))
        
        nr_bins = len(expected_pattern)
        error = 0
        for i in range(nr_bins):
            expected = expected_pattern[i]
            simulated = simulated_pattern[i]

            if method == 'relMAE':
                difference = numpy.abs(simulated - expected)
                max_distance = max(simulated,1-simulated,1 - expected, expected) # this part is updated
                error += 100 * ( difference / max_distance)
            elif method == 'MAE':
                error += 100 * numpy.abs(simulated - expected)
            else:
                raise Exception('ERROR: Invalid method (%s) in get_error. ' %
                    method + 'Valid options are MAE and relMAE.')

        return error / nr_bins

    def get_time_string(self, seconds):
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
