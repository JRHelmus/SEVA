'''
environment.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Last updated on August 2017
'''

import pandas
import requests
import numpy
import time

class Environment():
    ''' The Environment class deals with the charging stations and their
        occupancy.

    Args:
        data (DataFrame): Containing preprocessed sessions. This data will
            contain the columns 'location_key', 'amount_of_sockets', 'ID',
            'start_connection', 'end_connection', 'kWh', 'city',
            'region_abbreviation', 'provider', 'address', 'postal_code',
            'district', 'subdistrict', 'latitude' and 'longitude'.
        info_printer (bool): Parameter to decide whether to print information
            about run times.

    Attributes:
        css_occupation (Dict[str, List[Any]): Location keys as keys and
            occupied as value for each charging station. Occupied is a list with
            its length being the number of sockets of the cs and each element
            containing either the ID of the agent or None depending on if the
            socket is being used.
        css_info (Dict[str, Dict[str, Any]]): Location keys as keys and a
            dictionary as value. This dictionary contains the keys 'city',
            'region_abbreviation', 'provider', 'address', 'postal_code',
            'district', 'subdistrict', 'amount_of_sockets', 'latitude' and
            'longitude'.
    '''

    def __init__(self, data, info_printer, start_date_simulation):
        self.info_printer = info_printer
        self.start_date_simulation = start_date_simulation
        
        if self.info_printer:
            print('\tINFO: Started to get the data for all %d css' % ( len(data.location_key.unique() )))

        previous_time = time.process_time()
        
        self.css_info = {}
        for i, cs in enumerate(data.location_key.unique()):
            if i % int(len(data.location_key.unique())/10) == 0:
                print("\t\tINFO: Currently at %d percent with loading info for css in %.2f minutes" % \
                       (int(i/len(data.location_key.unique())*100), (time.process_time() - previous_time)/60))
                  
                previous_time = time.process_time()
                        
            cs_data =  data.loc[data.location_key == cs]
            
            self.css_info[cs] = {'city': cs_data.city.values[0],
            'region_abbreviation': cs_data.region_abbreviation.values[0],
            'provider': cs_data.provider.values[0],
            'address': cs_data.address.values[0],
            'postal_code': cs_data.postal_code.values[0],
            'district': cs_data.district.values[0],
            'subdistrict': cs_data.subdistrict.values[0],
            'subsubdistrict': cs_data.subsubdistrict.values[0],
            
            'latitude': cs_data.latitude.values[0],
            'longitude': cs_data.longitude.values[0],
            'parking_zone': cs_data.parking_zone.values[0],
            'placement_date': self._get_first_appearance(cs, cs_data),
            'status': cs_data.status.values[0],
            'FirstActiveDateTime': cs_data.FirstActiveDateTime.unique(),
            'LastActiveDateTime': cs_data.LastActiveDateTime.unique(),
            'Chargepoint_IDs':{chargepoint:
                               {
            'FirstActiveDateTime': cs_data.loc[cs_data.ChargePoint_ID == chargepoint]['FirstActiveDateTime'].unique()[0], \
            'LastActiveDateTime': cs_data.loc[cs_data.ChargePoint_ID == chargepoint]['LastActiveDateTime'].unique()[0], \
            'status': cs_data.loc[cs_data.ChargePoint_ID == chargepoint]['status'].values[0]} \
                     for chargepoint in cs_data.ChargePoint_ID.values
                              },
            
            'IsFastCharger': cs_data.IsFastCharger.values[0], 
            'IsChargeHub': cs_data.IsChargeHub.values[0]
            }
            
            self.css_info[cs]['amount_of_sockets'] = self._get_amount_of_sockets_at_current_time(cs, data, cs_data)
         
         
                  
        self.css_occupation = {cs: [None] * self.css_info[cs]['amount_of_sockets']
            for cs in self.css_info}
        self.css_cascaded = {cs: {'cascaded_time': self.css_info[cs]['placement_date'], 
                                  'counter': 0} for cs in self.css_info}

    def _get_first_appearance(self, cs, data):
        ''' This method gets the date of the first appearance in the data of a
            specified charging station.

        Args:
            cs (str): The location key of the charging station of which to
                determine the first appearance date.
            data (DataFrame): The dataframe containing the general data which
                has at least the column 'start_connection'.

        Returns:
            (DateTime): The date at which the charging station first appeared
                in the data.
        '''

        cs_sessions = data
        cs_sessions = cs_sessions.sort_values(by = 'start_connection', ascending = True)
        return cs_sessions['start_connection'].iloc[0]
                         
    def _get_amount_of_sockets_at_current_time(self, cs, data, cs_data = 'undefined'):
        ''' This method gets the amount of sockets at the start of the simulation of a specified charging station.

        Args:
            cs (str): The location key of the charging station of which to
                determine the amount of sockets.
            data (DataFrame): The dataframe containing the general data which
                has at least the column 'start_connection'.

        Returns:
            (DateTime): The amount of sockets at the start date of the simulation.
        '''    
                         
        amount_of_sockets = 0 
        
            
        cs_sessions = cs_data
                        
        for chargepoint in self.css_info[cs]['Chargepoint_IDs']:
#             if len(self.css_info[cs]['Chargepoint_IDs'][chargepoint]['FirstActiveDateTime']) > 1:
#                 print("WARNING: FirstActiveDateTime of ChargePoint_ID (%s) has a length of more than one" %
#                      (chargepoint))
#             else:
                first_active_date = self.css_info[cs]['Chargepoint_IDs'][chargepoint]['FirstActiveDateTime']
                last_active_date = self.css_info[cs]['Chargepoint_IDs'][chargepoint]['LastActiveDateTime']
                
                if self.start_date_simulation > first_active_date and self.start_date_simulation < last_active_date:
                    if type(cs_data) == str:
                        amount_of_sockets += data.loc[data.ChargePoint_ID == \
                                                         chargepoint].amount_of_sockets.values[0]
                    else:
                        amount_of_sockets += cs_sessions.loc[cs_sessions.ChargePoint_ID == \
                                                         chargepoint].amount_of_sockets.values[0]

        return amount_of_sockets            

    def reset_environment_sockets(self, start_date_simulation, data):
        ''' This method resets the amount of sockets given the start date of the simulation. 
            Especially when loading the environment, this is necessary to get the right amount of available sockets

        Args:
            start_date_simulation (str): The given start date of the simulation.

        Updates:
            css_info (the amount of sockets)
            css_occupation
            css_cascaded
        '''
        
        previous_time = time.process_time()
        for i, cs in enumerate(self.css_info):
            if self.info_printer:
                if len(self.css_info) > 0:
                    if i % int(len(self.css_info)/5) == 0 and i != 0:
                        print("\t\tINFO: Progress for resetting environment sockets is now at %d percent. \
                              \n\t\t\tThis took %.2f minutes" % \
                              (int(round(i/len(self.css_info)*100)), \
                               (time.process_time() - previous_time)/60))
                        previous_time = time.process_time()

            self.css_info[cs]['amount_of_sockets'] = self._get_amount_of_sockets_at_current_time(cs, data)
            
        self.css_occupation = {cs: [None] * self.css_info[cs]['amount_of_sockets']
            for cs in self.css_info}    
        self.css_cascaded = {cs: self.css_info[cs]['placement_date'] for cs in self.css_info}
        self._check_fast_chargers()
        
    def _check_fast_chargers(self):
        ''' This function sets amount of sockets of fast chargers to zero in the environment object.
        
        Updates:
        environment
        '''
        
        for cp in self.css_info:
            if self.css_info[cp]['IsFastCharger'] == True:
                self.css_info[cp]['amount_of_sockets'] = 0 

    def is_occupied(self, cs):
        ''' This method checks if the given charging station is occupied.

        Args:
            cs (str): The given charging station.

        Returns:
            (bool): True if the charging station is occupied.
        '''
        if cs == '3812':
            
            print(cs)
            print(self.css_occupation[cs].count(None))
            print(self.css_occupation[cs])
        return self.css_occupation[cs].count(None) == 0

    def who_occupies(self, cs):
        return self.css_occupation[cs]
    
    def is_cascaded(self, cs, time, check = False, cascade = False):
        ''' This method 1) checks if the given charging station is cascaded and 2) sets it to cascaded 
            when this is the case.

        Args:
            cs (str): The given charging station.

        Updates:
            css_cascaded
        '''
        
        if check:
            if time < self.css_cascaded[cs]['cascaded_time']:
#                 print("time", time)
#                 print("self.css_cascaded[cs]['cascaded_time']", self.css_cascaded[cs]['cascaded_time'])
                return True
            else:
                return False
        else:
            if cascade:
                self.css_cascaded[cs]['counter'] += 1
            else:
                self.css_cascaded[cs]['cascaded_time'] = time
            
    

    def connect_agent(self, agent_ID, cs):
        ''' This method connects an agent to a charging station. If the charging
            station has no free sockets, False (failure) is returned.

        Args:
            agent_ID (str): The ID of the agent which need to be connected.
            cs (str): The charging station the agent needs to be connected to.

        Updates:
            css_occupation

        Returns:
            (bool): True if the agent has been connected to the charging station.
        '''

        success = False

        for socket_index in range(len(self.css_occupation[cs])):
            if not self.css_occupation[cs][socket_index]:
                self.css_occupation[cs][socket_index] = agent_ID
                return True
        print('ERROR: Unable to connect agent with ID (%s) to ' % agent_ID +
            'charging station (%s).' % cs)
        return False

    def disconnect_agent(self, agent_ID, cs):
        ''' This method disconnects an agent from a charging station.

        Args:
            agent_ID (str): The ID of the agent which need to be connected.
            cs (str): The charging station the agent needs to be connected to.

        Updates:
            css_occupation
        '''

        try:
            index_socket = self.css_occupation[cs].index(agent_ID)
        except Exception as e:
            print('ERROR: Tried to disconnect agent with ID (%s) from ' %
                agent_ID + 'charging station (%s), but agent was not ' %
                cs + 'connected to the charging station.')
        else:
            self.css_occupation[cs][index_socket] = None

    def get_nearby_css(self, centers_css, center, training_data,
        walking_preparedness, city = None):
        ''' Using the centers of the agent and a specific center this method
            determines the charging stations that are nearby. Here nearby is
            within the parking zone(s) of the charging stations that are in
            the center (if parking zone is available) or the city of the center
            (if parking zone not available). Nearby charging stations should
            also be closer than the walking preparedness to the center.

        Args:
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center
                (in meters) as value.
            center (Tuple[float, float]): Location (lon, lat) of the center.
            training_data (DataFrame): Dataframe with the training data of the
                simulation.
            walking_preparedness (int): Walking preparedness of the agent in
                meters.
        Kwargs:
            city (str): The city to use as default. If None, city is gotten
                from the training data.

        Returns:
            css_with_distances (Dict[str, float]): As keys the location keys of
                the charging stations within the area of the center and as
                values their distance to the center in meters.
        '''
        # Check if a center has multiple cities, but irrelevant for my research and gives errors, 
        # since it does not find training_data for every cs in the center, weirdly enough.
        # Have commented out this part for now.
            
#         parking_zones_center = set()

#         if city == None:
#             cities_center = set()
            
#             for cs in centers_css[center]['habit']:
#                 parking_zone = training_data.loc[
#                     training_data['location_key'] == cs]['parking_zone'].values[0]
#                 parking_zones_center.add(parking_zone)

#                 city = training_data.loc[
#                     training_data['location_key'] == cs]['city'].values[0]
#                 cities_center.add(city)

#             if len(cities_center) > 1:
#                 print('WARNING: One center (%.2f, %.2f) ' % (center[0], center[1]) +
#                     'has charging stations in multiple cities, namely %s' %
#                     cities_center)
#             city = cities_center.pop()

        css_in_city = {cs for cs in self.css_info if self.css_info[cs]['city'] == city}
        
        css_with_distances = {cs: self.get_distance(cs, center) for cs in css_in_city}
        css_with_distances = {cs: distance for (cs, distance) in
            css_with_distances.items() if distance <= walking_preparedness}

        return css_with_distances

    def get_distance(self, cs, center):
        ''' Calculates the distances (in  meters) between a charging station
            and a center.

        Args:
            cs (str): Location key of the charging station.
            center (Tuple[float, float]): Tuple of (lon, lat) of the center.

        Returns:
            distance (float): Distance as the crow flies in meters.
        '''

        if self.distance_metric == 'as_the_crow_flies':
            lon1, lat1 =  self.css_info[cs]['longitude'], self.css_info[cs]['latitude']
            lon2, lat2 = center[0], center[1]
            lon1 = round(lon1, 7)
            lon2 = round(lon2, 7)
            lat1 = round(lat1, 7)
            lat2 = round(lat2, 7)

            if lon1 == lon2 and lat1 == lat2:
                return 0.0
            deg_to_rad = numpy.pi / 180.0
            phi1 = (90.0 - lat1) * deg_to_rad
            phi2 = (90.0 - lat2) * deg_to_rad
            theta1 = lon1 * deg_to_rad
            theta2 = lon2 * deg_to_rad
            cos = (numpy.sin(phi1) * numpy.sin(phi2) *
                numpy.cos(theta1 - theta2) + numpy.cos(phi1) * numpy.cos(phi2))
            arc = numpy.arccos(cos)
            earth_radius_km = 6373
            distance = arc * earth_radius_km
            return distance * 1000
        elif self.distance_metric == 'walking':
            return self._get_distance_orsm(cs, center)
        else:
            raise Exception('distance_metric (%s) is not implemented.' %
                self.distance_metric)
            
    def get_distance_general(self, location_1, location_2):
        ''' Calculates the distances (in  meters) between two locations

        Args:
            location_1 (Tuple[float, float]): Tuple of (lon, lat) of the first location.
            location_2 (Tuple[float, float]): Tuple of (lon, lat) of the second location.

        Returns:
            distance (float): Distance as the crow flies in meters.
        '''

        if self.distance_metric == 'as_the_crow_flies':
            lon1, lat1 =  location_1[0], location_1[1]
            lon2, lat2 = location_2[0], location_2[1]
            lon1 = round(lon1, 7)
            lon2 = round(lon2, 7)
            lat1 = round(lat1, 7)
            lat2 = round(lat2, 7)

            if lon1 == lon2 and lat1 == lat2:
                return 0.0
            deg_to_rad = numpy.pi / 180.0
            phi1 = (90.0 - lat1) * deg_to_rad
            phi2 = (90.0 - lat2) * deg_to_rad
            theta1 = lon1 * deg_to_rad
            theta2 = lon2 * deg_to_rad
            cos = (numpy.sin(phi1) * numpy.sin(phi2) *
                numpy.cos(theta1 - theta2) + numpy.cos(phi1) * numpy.cos(phi2))
            arc = numpy.arccos(cos)
            earth_radius_km = 6373
            distance = arc * earth_radius_km
            return distance * 1000
        elif self.distance_metric == 'walking':
            return self._get_distance_orsm(location_1, location_2)
        else:
            raise Exception('distance_metric (%s) is not implemented.' %
                self.distance_metric)

    def _get_distance_orsm(self, cs, center):
        ''' Calculates the distances (in  meters) between a charging station
            and a center.

        Args:
            cs (str): location key of the charging station.
            center (Tuple[float, float]): tuple of (lon, lat) of the center.

        Returns:
            distance (float): walking distance in meters.
        '''

        lon1, lat1 =  self.css_info[cs]['longitude'], self.css_info[cs]['latitude']
        lon2, lat2 = center[0], center[1]

        url = 'http://0.0.0.0:5322/route/v1/foot/%f,%f;%f,%f?overview=false' % \
            (lon1, lat1, lon2, lat2)
        return requests.get(url).json()['routes'][0]['distance']

    def reset(self):
        ''' This method clears the whole environment, ensuring no charging
            stations are occupied.
        '''

        for cs in self.css_occupation:
            self.css_occupation[cs] = [None] * len(self.css_occupation[cs])

    def get_maximum_distance_cs_center(self, centers_css):
        ''' This method calculates the maximum distance between centers and
            their charging stations.

        Args:
            centers_css (Dict[Tuple[float, float], Dict[str, Any]]): Centers
                (lon, lat) as keys and each center being a dictionary with the
                keys 'habit' and 'distance'. The value of the habit key is a set
                of charging stations (location keys), while the value of the
                distance key is a dictionary with the charging stations
                (location keys) as keys and their distance to the center
                (in meters) as value.

        Returns:
            max_distance (List[Float]): The maximum distance found between
                a center and one of its charging stations.
        '''

        max_distance = 0.0
        for center in centers_css:
            for cs in centers_css[center]['habit']:
                distance = self.get_distance(cs, center)
                if distance > max_distance:
                    max_distance = distance
        return [max_distance]
