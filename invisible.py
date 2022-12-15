"""
invisible.py

Written by:
    Seyla Wachlin
    Igna Vermeulen

Last updated on May 2017

Description:
    This file can be ignored by anyone using the model. It merely contains some
    temporary functions that were not (yet) ready to be put in the model.
"""

import pandas
import datetime
import bqplot
import ipyleaflet
import sklearn.cluster
import numpy
import colorsys
import IPython



def plot_clusters(sessions, centers_css, centers_info, textarea_cs, textarea_user, map,
                    fig_activity, x_sc_activity, y_sc_activity,
                    fig_intensity, x_sc_intensity, y_sc_intensity,
                    fig_connection, x_sc_connection, y_sc_connection,
                    fig_arrival, x_sc_arrival, y_sc_arrival,
                    agent, circles_around_css = True):
    '''
    Visualizes the clusters.

    Args:
        sessions: Pandas DataFrame containing sessions to consider in visualization
        # clusters: List of lists containing the location keys belonging to one cluster
        # centers: List of lists containing the mean location, stddev of the location,
        #          number of sessions and cluster number per center
        textarea_cs: Textarea widget to display information about the charging stations and clusters
        textarea_user: Textarea widget to display information about the user
        map: ipyleaflet Map object which, when displayed, shows a map centered on Amsterdam
        fig_activity: a bqplot Figure
        x_sc_activity: a bqplot Scale for the x-axis
        y_sc_activity: a bqplot Scale for the y-axis
        fig_intensity: a bqplot Figure
        x_sc_intensity: a bqplot Scale for the x-axis
        y_sc_intensity: a bqplot Scale for the y-axis
        circles_around_css: a Boolean, if true then we plot circles around the css,
                           if false we plot circles around the centers

    Returns:
        the data layer that was added to the map
    '''

    all_css = set(sessions["location_key"])

    distinct_colors_hex = get_colors(len(centers_css))

    features = []

    MAX_SIZE = 500

    def get_frequency(cs):
        return len(sessions[sessions['location_key'] == cs])

    sorted_css = sorted(all_css, key=get_frequency)
    max_nr_session_of_cs = len(sessions[sessions['location_key'] == sorted_css[-1]])

    layers = []

    for center in centers_info.keys():
        cluster_nr = centers_info[center]['center_nr']
        frequency_of_use = centers_info[center]['nr_of_sessions_in_center']
        cluster_color = distinct_colors_hex[cluster_nr]
        sessions_to_concat = [sessions[sessions['location_key'] == cs] for cs in centers_css[center]['habit']]
        sessions_cluster = pandas.concat(sessions_to_concat).to_json(date_format='iso')
        size = 50 + frequency_of_use / max_nr_session_of_cs * MAX_SIZE

        feature = {"type": "Feature", "properties": {"type": "cluster",
                                                     "cluster_nr": cluster_nr,
                                                     "center": center,
                                                     "frequency_of_use": frequency_of_use,
                                                     "nr_cs": len(centers_css[center]['habit']),
                                                     "css": centers_css[center]['habit'],
                                                     "sessions_cluster": sessions_cluster,
                                                     "style": {"fillOpacity": 0.8, "smoothFactor": 0, "stroke": True,
                                                               "fillColor": cluster_color, "color": "#000000"}},
                   # "geometry": {"type": "Polygon", "coordinates":
                   #              [polygon_generator(offset=[center[0], center[1]], size = size, shape = 'hexagon')]}}
                "geometry": {"type": "Point", "coordinates": list(center)}}
        features.append(feature)

        if not circles_around_css:
            c = ipyleaflet.Circle(location=[center[1], center[0]], radius = int(size), color= cluster_color,
                          fill_color = cluster_color, fillOpacity = 0.8, clickable = False)
            layers.append(c)
            map.add_layer(c)

    for nr, cs in enumerate(sorted_css):
        coordinates = [sessions[sessions['location_key'] == cs].longitude.values[0],
                       sessions[sessions['location_key'] == cs].latitude.values[0]]
        cluster_nr = -1
        for center in centers_css.keys():
            if cs in centers_css[center]['habit']:
                cluster_nr = centers_info[center]['center_nr']
                break
        frequency_of_use = len(sessions[sessions['location_key'] == cs])
        cluster_color = distinct_colors_hex[cluster_nr]
        size = 50 + frequency_of_use / max_nr_session_of_cs * MAX_SIZE
        sessions_cs = sessions[sessions['location_key'] == cs].to_json(date_format='iso')

        feature = {"type": "Feature", "properties": {"type": "charging station",
                                                     "cs": cs,
                                                     "cluster_nr": cluster_nr,
                                                     "frequency_of_use": frequency_of_use,
                                                     "cluster_color": cluster_color,
                                                     "size": size,
                                                     "sessions_cs": sessions_cs,
                                                     "style": {"fillOpacity": 0.8, "smoothFactor": 0, "stroke": True,
                                                               "color": "#000000", "fillColor": cluster_color}},
                   "geometry": {"type": "Polygon", "coordinates":
                                [polygon_generator(offset=coordinates, size = 0.2, shape = 'square')]}}
        features.append(feature)

        def cs_in_center(centers_css, cs):
            for center in centers_css.keys():
                if cs in centers_css[center]['habit']:
                    return True
            return False

        if circles_around_css and cs_in_center(centers_css, cs):
            c = ipyleaflet.Circle(location=[coordinates[1], coordinates[0]], radius = int(size), color= cluster_color,
                          fill_color = cluster_color, fillOpacity = 0.8, clickable = False)
            layers.append(c)
            map.add_layer(c)

    data = {"type": "FeatureCollection", "features": features}

    layer = ipyleaflet.GeoJSON(data=data, hover_style={'color': 'grey', 'fillColor': 'grey'})
    layers.append(layer)

    def click_handler(event=None, id=None, properties=None):
        if properties['type'] == "charging station":
            sessions_cs = pandas.read_json(properties["sessions_cs"])
            sessions_cs['start_connection'] = pandas.to_datetime(sessions_cs['start_connection'])
            sessions_cs['end_connection'] = pandas.to_datetime(sessions_cs['end_connection'])

            plot_activity(sessions_cs.copy(), x_sc_activity, y_sc_activity, fig_activity)
            plot_intensity_user(sessions_cs.copy(), x_sc_intensity, y_sc_intensity, fig_intensity)

            sessions_cs = sessions_cs.sort('start_connection')
            first_appearance = sessions_cs['start_connection'].iloc[0].date()
            last_appearance = sessions_cs['start_connection'].iloc[-1].date()

            textarea_cs.value = ("You clicked on a charging station.\n"
                                 "Location key(s): \t\t%s\n"
                                 "Nr of sessions: \t\t%d\n"
                                 "Active between: \t\t%s and %s\n"
                                 "Cluster: \t\t\t\t%d" % (properties['cs'], properties['frequency_of_use'],
                                                  first_appearance, last_appearance, properties['cluster_nr']))
        else:
            sessions_cluster = pandas.read_json(properties["sessions_cluster"])
            sessions_cluster['start_connection'] = pandas.to_datetime(sessions_cluster['start_connection'])
            sessions_cluster['end_connection'] = pandas.to_datetime(sessions_cluster['end_connection'])
            plot_activity(sessions_cluster.copy(), x_sc_activity, y_sc_activity, fig_activity)
            plot_intensity_user(sessions_cluster.copy(), x_sc_intensity, y_sc_intensity, fig_intensity)
            plot_distribution(agent.connection_duration_dists[tuple(properties["center"])],
                x_sc_connection, y_sc_connection,
                datetime.datetime(2000, 1, 1, 23, 59), fig_connection,
                "Connection duration dist for center (%.2f, %.2f)" % (tuple(properties["center"])))
            plot_distribution(agent.arrival_dists[tuple(properties["center"])],
                x_sc_arrival, y_sc_arrival,
                datetime.datetime(2000, 1, 1, 23, 59), fig_arrival,
                "Arrival dist for center (%.2f, %.2f)" % (tuple(properties["center"])))



            sessions_cluster = sessions_cluster.sort('start_connection')
            first_appearance = sessions_cluster['start_connection'].iloc[0].date()
            last_appearance = sessions_cluster['start_connection'].iloc[-1].date()

            textarea_cs.value = ("You clicked on a center.\n"
                                 "Nr of sessions: \t\t\t%s\n"
                                 "Nr of charging stations: \t\t%s\n"
                                 "Cluster: \t\t\t\t\t%s\n"
                                 "Charging stations: \t\t\t%s\n"
                                 "Active between: \t\t\t%s and %s" % (properties['frequency_of_use'], properties['nr_cs'],
                                                                    properties['cluster_nr'],
                                                                    properties['css'],
                                                                    sfirst_appearance, last_appearance))


    layer.on_click(click_handler)

    map.add_layer(layer)

    sessions = sessions.sort('start_connection')
    first_appearance = sessions['start_connection'].iloc[0]
    last_appearance = sessions['start_connection'].iloc[-1]

    sessions.set_index('start_connection', drop=False, inplace=True)
    df = sessions.groupby(pandas.TimeGrouper(freq='D')).count()
    average_intensity = df['start_connection'].mean()

    textarea_user.value = ("Nr of sessions: \t\t%s\n"
                           "Nr of centers: \t\t\t%s\n"
                           "Average intensity:\t\t%.4f\n"
                           "Active between:\t\t%s and %s"% (len(sessions), len(centers_css),
                                                            average_intensity, first_appearance, last_appearance))

    # sys.stdout.flush()
    # sys.stdout.write("\r%d clusters, namely: %s" % (len(clusters), str(clusters)))
    # sys.stdout.flush()

    return layers

def plot_activity(sessions, sc_x1, sc_y1, fig_activity, freq = '10Min'):
    '''
    This function plots the user activity plot once display is called upon.

    Args:
        sessions: Pandas DataFrame containing sessions to consider in visualization
        sc_x1: a bqplot Scale for the x-axis
        sc_y1: a bqplot Scale for the y-axis
        fig_activity: a bqplot Figure

    Kwargs:
        freq: String containing the frequency of the plot (default = '10Min').

    '''
    def get_date_range(row):
        thing = pandas.date_range(row['start_connection'], row['end_connection'], freq = freq)
        return (t for t in thing)

    sessions.all_dates = sessions.apply(lambda row: get_date_range(row), axis=1)

    # needed to convert generator object to list
    list_all_dates = [list(dates) for dates in sessions.all_dates]

    # create one big list from the list of lists (list_all_hours) and get only the times
    times = []
    for dates in list_all_dates:
        for date in dates:
            times.append(date.replace(year = 2015, month = 9, day = 25))

    df = pandas.DataFrame({'times': times})
    df.set_index('times', drop=False, inplace=True)
    df = df.groupby(pandas.TimeGrouper(freq=freq)).count()

    sc_x1.min = datetime.datetime(2015, 9, 25, 0, 0)
    sc_x1.max = datetime.datetime(2015, 9, 25, 23, 59)
    sc_y1.min = 0
    sc_y1.max = float(max(df.times))

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x=df['times'].index, y= df['times'], scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_activity.marks = [bar_chart]
    fig_activity.axes= [bar_x, bar_y]
    # fig_activity.title = "Activity pattern"

    return


def polygon_generator(offset, size = 1, edge_length_lon = 0.0005, edge_length_lat = 0.0005, shape = 'hexagon'):
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

def plot_background_image_amsterdam():
    '''
    Creates a figure showing the citymap of Amsterdam.

    Returns:
        ipyleaflet Map object which, when displayed, shows a map centered on Amsterdam.
    '''

    map_center = [52.179059695, 4.78514509466]
    m = ipyleaflet.Map(center=map_center, zoom=9)

    return m

def plot_intensity_user(sessions, sc_x1, sc_y1, fig_intensity):
    '''
    This function plots the user intensity plot once display is called upon.

    Args:
        sessions: Pandas DataFrame containing sessions to consider in visualization
        sc_x1: a bqplot Scale for the x-axis
        sc_y1: a bqplot Scale for the y-axis
        fig_intensity: a bqplot Figure

    '''
    sessions = sessions.sort('start_connection')
    first_appearance = sessions['start_connection'].iloc[0]
    last_appearance = sessions['start_connection'].iloc[-1]
    # if last_appearance - first_appearance < datetime.timedelta(days = 62):
    #     return

    sessions.set_index('start_connection', drop=False, inplace=True)
    df = sessions.groupby(pandas.TimeGrouper(freq='W')).count()

    sc_x1.min = datetime.datetime(2014, 1, 1, 0, 0)
    sc_x1.max = datetime.datetime(2016, 10, 1, 0, 0)

    # sc_x1.min = # df.index[0]
    # sc_x1.max = # df.index[-1]
    sc_y1.min = 0
    sc_y1.max = float(max(df.start_connection))

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x=df['start_connection'].index, y= df['start_connection'], scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_intensity.marks = [bar_chart]
    fig_intensity.axes= [bar_x, bar_y]
    # fig_intensity.title = "Intensity of RFID %s over time" %sessions['RFID'].iloc[0]

    return

def plot_distribution(dist, sc_x1, sc_y1, max_datetime, fig_distribution, title):
    sc_x1.min = datetime.datetime(2000, 1, 1, 0, 0)
    sc_x1.max = max_datetime
    sc_y1.min = 0
    sc_y1.max = float(max(dist))

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x = dist.index, y = dist,
                            scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_distribution.marks = [bar_chart]
    fig_distribution.axes = [bar_x, bar_y]
    fig_distribution.title = title


def clear_activity(sc_x1, sc_y1, fig_activity):
    '''
    This function clears the current fig_activity Figure.

    Args:
        sc_x1: a bqplot Scale for the x-axis
        sc_y1: a bqplot Scale for the y-axis
        fig_activity: a bqplot Figure

    '''
    sc_x1.min = datetime.datetime(2015, 9, 25, 0, 0)
    sc_x1.max = datetime.datetime(2015, 9, 25, 23, 59)
    sc_y1.min = 0
    sc_y1.max = 1.

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x=[], y= [], scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_activity.marks = [bar_chart]
    fig_activity.axes= [bar_x, bar_y]

    return

def clear_intensity(sc_x1, sc_y1, fig_intensity):
    '''
    This function clears the current fig_intensity Figure.

    Args:
        sc_x1: a bqplot Scale for the x-axis
        sc_y1: a bqplot Scale for the y-axis
        fig_intensity: a bqplot Figure

    '''
    sc_x1.min = datetime.datetime(2014, 1, 1, 0, 0)
    sc_x1.max = datetime.datetime(2016, 10, 1, 0, 0)

    sc_y1.min = 0
    sc_y1.max = 1.

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x=[], y= [], scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_intensity.marks = [bar_chart]
    fig_intensity.axes= [bar_x, bar_y]

    return

def clear_fig_distribution(sc_x1, sc_y1, max_datetime, fig_distribution, title):
    sc_x1.min = datetime.datetime(2000, 1, 1, 0, 0)
    sc_x1.max = max_datetime
    sc_y1.min = 0
    sc_y1.max = float(1)

    bar_x = bqplot.Axis(scale=sc_x1, orientation = 'horizontal')
    bar_y = bqplot.Axis(label='Nr of sessions', scale=sc_y1, orientation='vertical', tick_format='0.0f', grid_lines='solid')

    bar_chart = bqplot.Bars(x = [], y = [],
                            scales = {'x': sc_x1, 'y': sc_y1}, color_mode = 'element')

    fig_distribution.marks = [bar_chart]
    fig_distribution.axes = [bar_x, bar_y]
    fig_distribution.title = title


def get_colors(num_colors=10):
    colors=[]
    for i in numpy.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + numpy.random.rand() * 10)/100.
        saturation = (90 + numpy.random.rand() * 10)/100.
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_ = '#%02x%02x%02x' % (rgb[0]*255, rgb[1]*255, rgb[2]*255)
        colors.append(hex_)
    return colors

def get_centers(sessions, sklearn = True):
    '''
    Calculates the centers based on known sessions.

    Args:
        sessions: Pandas DataFrame containing the sessions of one user

    Kwargs:
        sklearn: boolean that decides whether to use sklearn algorithm or not

    Returns:
         clusters: List of lists containing the location keys belonging to one cluster
         layer: array of ipyleaflet layers
         all_centers: List of [[2d array of mean], [2d array of standard dev], nr of sessions in cluster, cluster number, nr of css in cluster].

    '''

    threshold_fraction_sessions = 0.05

    if sklearn:
        clusters = get_clusters_sklearn(sessions)
    else:
        distance_matrix, location_key_to_indices = get_distance_matrix(sessions)
        clusters = get_clusters(distance_matrix, location_key_to_indices)

    all_centers = []
    for cluster_nr, cluster in enumerate(clusters):
        # mean, stddev, nr_sessions, cluster_nr, nr_cs, css in cluster
        center = [[], [], 0, cluster_nr, len(cluster), cluster]
        nr_of_sessions_in_cluster = 0
        all_locations = []
        for cs in cluster:
            nr_of_sessions_in_cluster += len(sessions[sessions['location_key'] == cs])
            lon, lat = sessions[sessions['location_key'] == cs].longitude.values[0], \
                       sessions[sessions['location_key'] == cs].latitude.values[0]
            all_locations.append([lon, lat])

        if nr_of_sessions_in_cluster / len(sessions) > threshold_fraction_sessions:
            center[0] = numpy.mean(all_locations, axis = 0)
            center[1] = numpy.std(all_locations, axis = 0)
            center[2] = nr_of_sessions_in_cluster
            all_centers.append(center)

    return sessions, clusters, all_centers

def get_clusters_sklearn(sessions):
    '''
    Creates clusters of the charging stations used by a single user.

    Args:
        sessions: Pandas DataFrame containing sessions of the user

    Returns:
        List of lists containing the location keys belonging to one cluster.
    '''

    cluster_df, random_css = get_user_activities(sessions)
    if len(cluster_df) > 0:
        mat = cluster_df.as_matrix()
        # Using sklearn
        km = sklearn.cluster.Birch(threshold = 1.5, n_clusters = None)
        km.fit(mat)
        # Get cluster assignment labels
        labels = km.labels_
        # Format results as a DataFrame
        results = pandas.DataFrame(data=labels, columns=['cluster'], index = cluster_df.index)
        # Get the desired output
        clustered_css =  [list(group[1].index) for group in results.groupby('cluster')]
    else:
        clustered_css = []
    for cs in random_css:
        clustered_css.append([cs])
    return clustered_css

def get_user_activities(sessions, freq = '10Min'):
    '''
    This function calculates the activity patterns of a user for each charging station along with their location.

    Args:
        sessions: Pandas DataFrame containing sessions of the user

    Kwargs:
        freq: String containing the frequency of the plot (default = '10Min').

    Returns:
        Pandas DataFrame with as columns the values at each timeunit of the normalized activity pattern and the lon, lat of the
        cs and as rows the charging stations of the user.
    '''


    def get_date_range(row):
        thing = pandas.date_range(row['start_connection'], row['end_connection'], freq = freq)
        return (t for t in thing)

    lon_lat_scale = 50.
    css = list(sessions.location_key.unique())
    clustered_css = list(sessions.location_key.unique())
    activities = []
    times_day = pandas.date_range('2015-09-25 00:00:00', '2015-09-25 23:59:59', freq = '10Min')
    random_css = []
    for i, cs in enumerate(css):
        cs_sessions = sessions.loc[sessions['location_key'] == cs]
        #Skip the css with less than 5 sessions, we don't want to cluster those
        if len(cs_sessions) < 5:
            random_css.append(cs)
            clustered_css.remove(cs)
            continue
        cs_sessions.all_dates = cs_sessions.apply(lambda row: get_date_range(row), axis=1)
        list_all_dates = [list(dates) for dates in cs_sessions.all_dates]

        # create one big list from the list of lists (list_all_hours) and get only the times
        times = []
        for dates in list_all_dates:
            for date in dates:
                times.append(date.replace(year = 2015, month = 9, day = 25))
        df = pandas.DataFrame({'times': times})
        df.set_index('times', drop=False, inplace=True)
        df = df.groupby(pandas.TimeGrouper(freq=freq)).count()

        #add missing times
        if(len(df) < 144) :
            missing_indices = [time  for time in times_day if time not in list(df.times.index)]
            missing_series = pandas.Series([0]*len(missing_indices), index = missing_indices)
            activity = df.times.append(missing_series)
            activity = list(activity.sort_index())
        else:
            activity = list(df.times)
        activity /= max(activity)
        activity = list(activity)
        activity.append(lon_lat_scale*(47.4+cs_sessions.longitude.values[0]))
        activity.append(lon_lat_scale*cs_sessions.latitude.values[0])
        activities.append(activity)

    if len(activities) == 0 :
        return pandas.DataFrame(), random_css
    col_names = ['act_%d' %i for i in range(len(activities[0])-2)]
    col_names.append('lon')
    col_names.append('lat')
    cluster_df = pandas.DataFrame(activities, columns = col_names, index = clustered_css)
    return cluster_df, random_css


def create_qgrid(agents):
    '''
    This function creates a pandas DataFrame containing the first appearance, last appearance, nr of centers, main city, main
    postal code and average nr of sessions per week of the unique users. Also creates a Dictionary to translate the numbers of
    users with their RFID. It will return and save both.

    Args:
        agents: List of agent objects to create the qgrid for

    Returns:
        Pandas DataFrame of the table
        Dictionary with nr's as keys and RFID's as corresponding values
    '''


    user_dict = {ID: {'ID': ID, 'User': i, 'First appearance': appearance(ID, agents[ID].training_sessions, 'first'),
                        'Last appearance': appearance(ID, agents[ID].training_sessions, 'last'),
                        'Nr centers': len(agents[ID].centers_css.keys()),
                        'Main city': main_city(ID, agents[ID].training_sessions),
                        'Main postal code': main_postal_code(ID, agents[ID].training_sessions),
                        'Max kWh': max(agents[ID].training_sessions[agents[ID].training_sessions['ID'] == ID]['kWh'])} for i, ID in enumerate(agents.keys())}
    # print("Tranforming it to a dataframe")
    table_df = pandas.DataFrame.from_dict(user_dict, orient = 'index')
    # print("Creating the RFID translator")
    # creating a dictionary that keeps track of the nr each RFID has
    RFID_translator = {user_dict[RFID]['User']: RFID for RFID in user_dict.keys()}

    # print("setting the right columns")
    #Setting column and index name of table dataframe
    table_df = table_df[['User', 'ID', 'First appearance', 'Last appearance', 'Nr centers', 'Main city', 'Main postal code', 'Max kWh']]

    table_df = table_df.set_index('User')

    # print("Adding nr sessions")
    ''' Add nr of sessions for user '''
    new_values = [len(agents[table_df['ID'][i]].training_sessions[agents[table_df['ID'][i]].training_sessions['ID'] == RFID_translator[i]])  for i in table_df.index]
    table_df["Nr sessions"] = new_values

    # print("adding average sessions per week")
    list_sessions_per_week = [row['Nr sessions'] /
                              (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                               - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days
                              / 7)
                              for index, row in table_df.iterrows()]
    table_df['Avg sessions / week'] = list_sessions_per_week
    table_df.drop('Nr sessions', axis=1, inplace=True)



    '''print("saving the files")
    # saving the dataframe and translation dict to a csv and pkl file
    table_df.to_csv('table_df.csv')
    with open('RFID_translator.pkl', 'wb') as f:
        pickle.dump(RFID_translator, f, pickle.HIGHEST_PROTOCOL)
    '''
    return table_df, RFID_translator

def create_qgrid_cs(df):
    '''
    This function creates a pandas DataFrame containing the nr of unique users, average users p/w, avg sessions p/w and avg kwh
    p/w with the location keys of each cs as index. It will return and save the DataFrame.

    Args:
        df: Pandas DataFrame containing all sessions to consider

    Returns:
        Pandas DataFrame of the table
    '''

    # print("Getting unique css")
    unique_css = df['location_key'].unique()
    # sessions_per_cs = {cs: df[df['location_key'] == cs] for cs in unique_css}


    # print("Creating the cs dictionary")
    # creating the dictionary
    cs_dict = {cs: {'CS': cs, '# unique users': len(df.loc[df['location_key'] == cs].ID.unique()),
                   'First appearance': appearance_cs(df[df['location_key'] == cs], 'first').date(),
                   'Last appearance': appearance_cs(df[df['location_key'] == cs], 'last').date(),
                   'City': df.loc[df['location_key'] == cs]['city'].values[0],
                   'Postal code': df.loc[df['location_key'] == cs]['postal_code'].values[0],
                   'Coefficient of variation': get_cv_cs(df, cs)}
               for cs in unique_css}
    # print("Transforming to dataframe")
    table_df = pandas.DataFrame.from_dict(cs_dict, orient = 'index')

    # print("Setting columns")
    #Setting column and index name of table dataframe
    table_df = table_df[['CS', '# unique users', 'Coefficient of variation', 'First appearance', 'Last appearance',
                         'City', 'Postal code']]
    table_df = table_df.set_index('CS')
    # print("created df with nr unique users")

    ''' Add nr of sessions for user '''
    # print("Adding nr sessions")
    new_values = [len(df[df['location_key'] == cs]) for cs in table_df.index]
    table_df["#sessions"] = new_values

    # print("created nr sessions column")

    # print("Creating avg sessions p/w")
    list_sessions_per_week = [row['#sessions'] /
                (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days
                / 7)
                if (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days)
                > 0
                else 0 for index, row in table_df.iterrows()]
    # print("Converting to table")
    table_df['Avg sessions p/w'] = list_sessions_per_week

    # print('created avg sessions p/w column')

    # print("Added kwh charged")
    new_values = [sum(df[df['location_key'] == cs].kWh) for cs in table_df.index]
    table_df['Total kWh'] = new_values

    # print("Adding avg kwh p/w")
    list_kWh_pw = [row['Total kWh'] /
                    (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                    - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days
                        / 7)
                   if (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                    - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days)
                   > 0
                   else 0 for index, row in table_df.iterrows()]
    table_df['Avg kWh p/w'] = list_kWh_pw

    # print("Dropping leftover columns")
    table_df.drop(['#sessions', 'Total kWh'] , axis=1, inplace=True)

    # print("Saving to csv")
    # # saving the dataframe to a csv file
    # table_df.to_csv('table_df_cs.csv')


    return table_df


def create_qgrid_cs_old(df):
    '''
    This function creates a pandas DataFrame containing the nr of unique users, average users p/w, avg sessions p/w and avg kwh
    p/w with the location keys of each cs as index. It will return and save the DataFrame.

    Args:
        df: Pandas DataFrame containing all sessions to consider

    Returns:
        Pandas DataFrame of the table
    '''

    # print("Getting unique css")
    unique_css = df['location_key'].unique()
    # sessions_per_cs = {cs: df[df['location_key'] == cs] for cs in unique_css}


    # print("Creating the cs dictionary")
    # creating the dictionary
    cs_dict = {cs: {'CS': cs, '# unique users': len(df.loc[df['location_key'] == cs].RFID.unique()),
                   'First appearance': appearance_cs(df[df['location_key'] == cs], 'first').date(),
                   'Last appearance': appearance_cs(df[df['location_key'] == cs], 'last').date(),
                   'City': df.loc[df['location_key'] == cs]['city'].values[0],
                   'Postal code': df.loc[df['location_key'] == cs]['postal_code'].values[0],
                   'Coefficient of variation': get_cv_cs(df, cs)}
               for cs in unique_css}
    # print("Transforming to dataframe")
    table_df = pandas.DataFrame.from_dict(cs_dict, orient = 'index')

    # print("Setting columns")
    #Setting column and index name of table dataframe
    table_df = table_df[['CS', '# unique users', 'Coefficient of variation', 'First appearance', 'Last appearance',
                         'City', 'Postal code']]
    table_df = table_df.set_index('CS')
    # print("created df with nr unique users")

    ''' Add nr of sessions for user '''
    # print("Adding nr sessions")
    new_values = [len(df[df['location_key'] == cs]) for cs in table_df.index]
    table_df["#sessions"] = new_values

    # print("created nr sessions column")

    # print("Creating avg sessions p/w")
    list_sessions_per_week = [row['#sessions'] /
                (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days
                / 7)
                if (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days)
                > 0
                else 0 for index, row in table_df.iterrows()]
    # print("Converting to table")
    table_df['Avg sessions p/w'] = list_sessions_per_week

    # print('created avg sessions p/w column')

    # print("Added kwh charged")
    new_values = [sum(df[df['location_key'] == cs].kWh) for cs in table_df.index]
    table_df['Total kWh'] = new_values

    # print("Adding avg kwh p/w")
    list_kWh_pw = [row['Total kWh'] /
                    (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                    - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days
                        / 7)
                   if (((row['Last appearance'] - datetime.timedelta(days = row['Last appearance'].weekday()))
                    - (row['First appearance'] - datetime.timedelta(days = row['First appearance'].weekday()))).days)
                   > 0
                   else 0 for index, row in table_df.iterrows()]
    table_df['Avg kWh p/w'] = list_kWh_pw

    # print("Dropping leftover columns")
    table_df.drop(['#sessions', 'Total kWh'] , axis=1, inplace=True)

    # print("Saving to csv")
    # # saving the dataframe to a csv file
    # table_df.to_csv('table_df_cs.csv')


    return table_df

def appearance(user, df, first_or_last):
    '''
    Determines date of first or last appearance of the user.

    Args:
        user: RFID of the user
        df: Pandas DataFrame containing (at least) the sessions of the user
        first_or_last: boolean with value 'first' or 'last' depending on where you want the first or last appearance

    Returns:
        Datetime containing the first or last appearance of the user.
    '''

    sessions_user = df.loc[df['ID'] == user]
    sessions_user = sessions_user.sort('start_connection')
    if first_or_last == 'first':
        try:
            return sessions_user['start_connection'].iloc[0]
        except BaseException as e:
            print(e)
            print(user)
            print(sessions_user)
            return None
    if first_or_last == 'last':
        try:
            return sessions_user['start_connection'].iloc[-1]
        except BaseException as e:
            print(e)
            print(user)
            print(sessions_user)
            return None
    print("Invalid 'first_or_last' argument given to my_dashboard.appearance()")
    return

def appearance_cs(sessions_cs, first_or_last):
    '''
    Determines date of first or last appearance of the user.

    Args:
        user: RFID of the user
        df: Pandas DataFrame containing (at least) the sessions of the user
        first_or_last: boolean with value 'first' or 'last' depending on where you want the first or last appearance

    Returns:
        Datetime containing the first or last appearance of the user.
    '''

    sessions_cs = sessions_cs.sort('start_connection')
    if first_or_last == 'first':
        try:
            return sessions_cs['start_connection'].iloc[0]
        except BaseException as e:
            print(e)
            print(user)
            print(sessions_cs)
            return None
    if first_or_last == 'last':
        return sessions_cs['start_connection'].iloc[-1]
    print("Invalid 'first_or_last' argument given to my_dashboard.appearance()")
    return

def get_cv_cs(sessions, cs):
    '''
    Calculates the coefficient of variation of the charging station activity pattern

    Args:
    sessions: Pandas DataFrame with all sessions
    cs: location key of the charging stations on which to get the standard deviation

    Returns:
    A double with the coefficient of variation
    '''

    sessions = sessions.loc[sessions['location_key'] == cs]
    unique_users = sessions.ID.unique()

    activities = []
    for user_rfid in unique_users:
        sessions_per_user = sessions.loc[sessions['ID'] == user_rfid]
        if len(sessions_per_user) > 0:
            activity = get_activity_pattern(sessions_per_user)
            activities.append(activity)
    m = numpy.mean(numpy.sum(activities, axis = 0))
    if m == 0:
        return -1
    return numpy.std(numpy.sum(activities, axis = 0))/ m

def main_city(user, df):
    '''
    Determines the most used city by the user.

    Args:
        user: RFID of the user
        df: Pandas DataFrame containing the sessions of the user

    Returns:
        String of the most used city.

    '''
    sessions_user = df.loc[df['ID'] == user]
    return sessions_user['city'].value_counts().idxmax()

def main_postal_code(user, df):
    '''
    Determines the most used postal code by the user.

    Args:
        user: RFID of the user
        df: Pandas DataFrame containing the sessions of the user

    Returns:
        String of the most used postal code.

    '''
    sessions_user = df.loc[df['ID'] == user]
    return sessions_user['postal_code'].value_counts().idxmax()

def get_activity_pattern(sessions, freq = '10Min'):
    def get_date_range(row):
        thing = pandas.date_range(row['start_connection'], row['end_connection'], freq = freq)
        return (t for t in thing)

    sessions.all_dates = sessions.apply(lambda row: get_date_range(row), axis=1)

    times_day = pandas.date_range('2015-09-25 00:00:00', '2015-09-25 23:59:59', freq = '10Min')

    # needed to convert generator object to list
    list_all_dates = [list(dates) for dates in sessions.all_dates]

    # create one big list from the list of lists (list_all_hours) and get only the times
    times = []
    for dates in list_all_dates:
        for date in dates:
            times.append(date.replace(year = 2015, month = 9, day = 25))

    df = pandas.DataFrame({'times': times})
    df.set_index('times', drop=False, inplace=True)
    df = df.groupby(pandas.TimeGrouper(freq=freq)).count()

    if(len(df) < 144):
        missing_indices = [time for time in times_day if time not in list(df.times.index)]
        missing_series = pandas.Series([0]*len(missing_indices), index = missing_indices)
        activity = df.times.append(missing_series)
        activity = activity.sort_index()
    else:
        activity = df.times

    if len(activity) < 144:
        print("ERROR!")

    return activity
