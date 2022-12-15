import pickle
import numpy
import pandas

def get_distance(center1, center2):
    lon1, lat1 =  center1[0], center1[1]
    lon2, lat2 = center2[0], center2[1]
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

def get_home_location_database():
    # replace with path to csv
    path_to_file = 'testing_center_locations/RFID_HomeLocation.csv'
    with open(path_to_file, 'rb') as data_file:
        data = pandas.read_csv(data_file)

    # assuming data contains three columns, one with hashed RFID, one with Longitude and one with Latitude
    home_longitude = pandas.Series(data.Longitude.values, index=data['RFID']).to_dict()
    home_latitude = pandas.Series(data.Latitude.values, index=data.RFID).to_dict()
    return home_longitude, home_latitude

def main():
	# replace with path to pkl
	filename = 'testing_center_locations/agent_home_locations.pkl'
	with open(filename, 'rb') as agent_file:
	    agent_home_locations = pickle.load(agent_file)

	home_longitude, home_latitude = get_home_location_database()

	distances_between_predictions = {}
	for agent in agent_home_locations:
	    if agent in home_longitude:
	        distances_between_predictions[agent] = {}
	        for center in agent_home_locations[agent]:
	            distance = get_distance(center, (home_longitude[agent], home_latitude[agent]))
	            distances_between_predictions[agent][center] = distance

	# replace with wanted path
	filename = 'testing_center_locations/distance_differences.pkl'
	with open(filename, 'wb') as agent_file:
	    pickle.dump(distances_between_predictions, agent_file)

	print('distances between predictions writen to file %s' % filename)
	print('%d agent(s) in both simulation and given location file.' % len(distances_between_predictions.keys()))

if __name__ == '__main__':
	main()