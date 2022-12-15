import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import seaborn
import pickle
import pandas
import numpy
import math
import os
from scipy.stats import spearmanr
import parking_zones
import json
import shapely
import re

conversion_card_provider = {'ANWB': 'ANWB per sessie',
                            'Travelcard': 'Travelcard*',
                            'LMS': '?',
                            'Eneco': '?',
                            'Nuon': 'Nuon',
                            'NL-FLW': '?',
                            'GreenFlux': 'Greenflux',
                            'Essent': '?',
                            'TheNewMotion': 'The new motion*',
                            'EVBOX': 'EVBox'}

conversion_charge_point_operator = {'ALFEN': 'Alfen (Den Haag)',
                                    'BALLASTNEDAM': 'Ballast Nedam/Enovates (Utrecht)',
                                    'ESSENT': 'Essent',
                                    'EVBOX': 'Engie/EV box (Rotterdam)',
                                    'EVNET': 'EvnetNL',
                                    'NUON': 'Nuon (Amsterdam)'}


def get_raw_data():
    path_to_file = 'data/sessions/ChargeSessionsUnmerged_raw.pkl'
    with open(path_to_file, 'rb') as data_file:
        data = pickle.load(data_file)
    return data

def get_all_zones_den_haag(parking_zones_den_haag):
    all_zones_den_haag = {}
    days = {'Ma': 0, 'Di': 1, 'Wo': 2, 'Do': 3, 'Vr': 4, 'Za': 5, 'Zo': 6}
    for zone in parking_zones_den_haag['features']:
        # if zone['properties']['CODE'][-4:] == "(PT)":
        clean = re.sub(r" max \d+ min", "", zone['properties']['WERKINGSTI'])
        clean = re.sub(r" max \d+ u", "", clean)
        clean = clean.replace('\r\n', '')
        clean = clean.replace('onbeperkt,', '')
        clean = clean.replace('onbeperkt', '')
        clean = clean.replace('Binnenstad Max 30 min', '')
        clean = clean.replace('Binnenstad', '')
        clean = clean.replace('uten Gentsestraat', '')
        clean = clean.replace(', 5,00 max dagkaart GSM 2259', '')
        clean = clean.replace('Nachtregeling', 'Ma/Zo 0/2')
        clean = clean.replace('Do koopavond 21:00', 'Do 18/21 1,70 p/u')
        clean = clean.replace('Zon', 'Zo')
        clean = clean.replace('vr', 'Vr')
        clean = clean.replace('Ma/Woe/Vrij/Za', 'Ma/Wo/Vr/Za')
        clean = clean.replace('  ', ' ')
        clean = clean.replace('Ma/Zo 10/24, Vr 24/02 Za 24/02, 1,70 p/u', 'Ma/Zo 10/24, Vr 24/02 Za 24/02; 1,70 p/u')
        clean = clean.replace('Ma/Za 9/24 Zo 18/24 1,70', 'Ma/Za 9/24 Zo 18/24; 1,70 p/u')
        clean = clean.replace('Ma/Za 10/24 Zo 13/24 2,10', 'Ma/Za 10/24 Zo 13/24; 2,10 p/u')
        clean = clean.replace('Ma/Wo/Vr/Za 9/24 Di/Do/Zo 18/24 2,10/1,70 p/u', 'Ma/Wo/Vr/Za 9/24, Di/Do/Zo 18/24; 9/18 2,10 p/u, 18/24 1,70 p/u')
        clean = clean.replace('2,10 18/24 1,70 p/u', '2,10 p/u, 18/24 1,70 p/u')
        clean = clean.replace('Ma/Za 10/24, Zo 13/24 2,10 p/u', 'Ma/Za 10/24, Zo 13/24; 2,10 p/u')
        data = zone['properties']['WERKINGSTI']
        if 'p/u, ' in clean and clean != 'Ma/Wo/Vr/Za 9/24, Di/Do/Zo 18/24; 9/18 2,10 p/u, 18/24 1,70 p/u' and \
            clean != 'Ma/Za 9/24, Zo 18/24; 9/18 2,10 p/u, 18/24 1,70 p/u':
            clean = clean.replace(' p/u', '')
            clean = clean.split(', ')
            data = {days[day]: [] for day in days}

            for c in clean:
                start_end, times, price = c.split(' ')
                if '/' in start_end:
                    start_day, end_day = start_end.split('/')
                else:
                    start_day, end_day = start_end, start_end
                start_time, end_time = times.split('/')
                price = float(price.replace(',', '.'))
                for i in range(days[start_day], days[end_day] + 1):
                    data[i].append({'start_time': int(start_time), 'end_time': int(end_time), 'price': float(price)})
        elif ';' in clean:
            clean = clean.replace(' p/u', '')
            if clean == 'Ma/Wo/Vr/Za 9/24, Di/Do/Zo 18/24; 9/18 2,10, 18/24 1,70' or \
                clean == 'Ma/Wo/Vr/Za 9/24; 9/18 2,10, 18/24 1,70 Di/Do/Zo 18/24 1,70 ':
                data = {0: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 18, 'end_time': 24, 'price': 1.70}]}
            elif clean == 'Ma/Za 9/24, Zo 18/24; 9/18 2,10, 18/24 1,70':
                data = {0: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 9, 'end_time': 18, 'price': 2.10},
                            {'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 18, 'end_time': 24, 'price': 1.70}]}
            elif clean == 'Ma/Za 10/24, Zo 13/24; 2,10' or clean == 'Ma/Za 10/24 Zo 13/24; 2,10':
                 data = {0: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        1: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        2: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        3: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        4: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        5: [{'start_time': 10, 'end_time': 24, 'price': 2.10}],
                        6: [{'start_time': 13, 'end_time': 24, 'price': 2.10}]}
            elif clean == 'Ma/Do 9/24, Vr/Za 9/02, Zo 13/24; 2,10':
                 data = {0: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        1: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        2: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        3: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        4: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        5: [{'start_time': 9, 'end_time': 24, 'price': 2.10}],
                        6: [{'start_time': 13, 'end_time': 24, 'price': 2.10}]}
            elif clean == 'Ma/Wo/Vr/Za 9/24; 9/19 0,80 19/24 1,70 Di/Do/Zo 18/24 1,70 ':
                data = {0: [{'start_time': 9, 'end_time': 19, 'price': 0.8},
                            {'start_time': 19, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 9, 'end_time': 19, 'price': 0.8},
                            {'start_time': 19, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 18, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 9, 'end_time': 19, 'price': 0.8},
                            {'start_time': 19, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 9, 'end_time': 19, 'price': 0.8},
                            {'start_time': 19, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 18, 'end_time': 24, 'price': 1.70}]}
            elif clean == 'Ma/Za 9/24 Zo 18/24; 1,70':
                 data = {0: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 18, 'end_time': 24, 'price': 1.70}]}
            elif clean == 'Ma/Zo 10/24, Vr 24/02 Za 24/02; 1,70':
                 data = {0: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 10, 'end_time': 24, 'price': 1.70}]}
        elif ', Z' in clean:
            clean = clean.replace(' 1,', ' 1,')
            if clean == 'Ma/Za 9/24, Zo 18/24 1,70 p/u':
                 data = {0: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 9, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 18, 'end_time': 24, 'price': 1.70}]}
            elif clean == 'Ma/Za 10/24, Zo 13/24 1,70 p/u':
                 data = {0: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        1: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        2: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        3: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        4: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        5: [{'start_time': 10, 'end_time': 24, 'price': 1.70}],
                        6: [{'start_time': 13, 'end_time': 24, 'price': 1.70}]}
        else:
            clean = clean.replace(' p/u', '').replace(', ', ' ').replace(' P/u', '')
            if ' ' not in clean:
                data = {days[day]: [] for day in days}
            else:
                start_end, times, price = clean.split(' ')
                start_day, end_day = start_end.split('/')
                start_time, end_time = times.split('/')
                price = float(price.replace(',', '.'))
                data = {}
                for day in days:
                    if days[day] < days[start_day] or days[day] > days[end_day]:
                        data[days[day]] = []
                    else:
                        data[days[day]] = [{'start_time': int(start_time), 'end_time': int(end_time), 'price': price}]
        all_zones_den_haag[zone['properties']['CODE']] = data
    return all_zones_den_haag

def get_zone_information():
    geojson_amsterdam = "data/parkingzones/parkeergebied_amsterdam.geojson"
    geojson_den_haag = "data/parkingzones/parkeergebied_den_haag.geojson"

    with open(geojson_amsterdam) as f:
        parking_zones_amsterdam = json.load(f)
    with open(geojson_den_haag) as f:
        parking_zones_den_haag = json.load(f)
    parking_zones_den_haag = parking_zones.preprocess_parking_zones_den_haag(parking_zones_den_haag)

    return parking_zones_amsterdam, parking_zones_den_haag

def get_zone(parking_zones_amsterdam, parking_zones_den_haag, lon, lat):
    location = shapely.geometry.Point(lon, lat)

    for feature in parking_zones_amsterdam['features']:
        if not feature['properties']['E_DAT_GEBI']:
            parking_zone = shapely.geometry.shape(feature['geometry'])
            if parking_zone.contains(location):
                return feature['properties']['GEBIED_COD']

    for feature in parking_zones_den_haag['features']:
        parking_zone = shapely.geometry.shape(feature['geometry'])
        if parking_zone.contains(location):
            return feature['properties']['CODE']

    return None


def get_all_prices():
    price_data = pandas.read_csv('data/extra/overzicht_laadtarieven.csv', skiprows = [1, 2, 3], usecols = range(3, 24))

    def get_price(price):
        if '-' in price:
            return 0
        return float(price[2:])

    laadpas_providers = list(price_data['Laadpasprovider'])
    all_prices = {}

    previous_column = ''
    for i, column in enumerate(price_data.columns):
        if i % 2 == 1:
            all_prices[column] = {}
            column_prices = price_data[column]
            for i, price in enumerate(column_prices):
                charge_point_operator = laadpas_providers[i]
                if isinstance(charge_point_operator, str):
                    all_prices[column][charge_point_operator] = [get_price(price)]
            previous_column = column
        elif i > 0:
            column_prices = price_data[column]
            for i, price in enumerate(column_prices):
                charge_point_operator = laadpas_providers[i]
                if isinstance(charge_point_operator, str):
                    all_prices[previous_column][charge_point_operator].append(get_price(price))


    essent_charge_point_operator = {'Travelcard*': [0, 0.36], 'EVBox': [0, 0.36], 'Greenflux': [0, 0.37], 'Nuon': [0, 0.3], 'Essent': [0.61, 0.36], 'The new motion*': [0.35, 0.36], 'ANWB maandelijks': [0, 0.36], 'ANWB per sessie': [0.34, 0.36], 'Flow Charging': [0.12, 0.36], 'Movenience': [0.3, 0.3], 'Eneco': [0.61, 0.36], 'Radiuz': [0, 0.36]}

    for laadpas_provider in all_prices:
        if laadpas_provider in essent_charge_point_operator:
            all_prices[laadpas_provider]['Essent'] = essent_charge_point_operator[laadpas_provider]

    return all_prices

def get_all_providers(agents):
    all_providers = []
    for a in agents:
        all_providers += list(a['training_sessions'].provider.unique())

    return set(all_providers)

def get_all_charging_speeds():
    path_to_file = 'data/extra/ChargePoint_ChargingSpeed_Estimate.csv'
    with open(path_to_file, 'rb') as data_file:
        data = pandas.read_csv(data_file)
    charging_speeds = pandas.Series(data.def_ampere.values,index=data.ChargePoint_skey).to_dict()
    return charging_speeds

def get_all_card_providers():
    path_to_file = '../data/extra/RFID_ServiceProvider.csv'
    with open(path_to_file, 'rb') as data_file:
        data = pandas.read_csv(data_file)
    all_card_providers = pandas.Series(data.ServiceProvider.values,index=data.RFID).to_dict()
    return all_card_providers

def get_choices(environment, agent, center, choices, parameter, all_prices, all_charging_speeds, card_provider, raw_data, parking_zones_amsterdam, parking_zones_den_haag):

    if parameter == 'prices_relative':
        kWh_charged = 14.4

        prices = []
        used = []
        for i, cs in enumerate(choices):
            provider = environment.css_info[cs]['provider']
            converted_provider = conversion_charge_point_operator[provider]
            price = all_prices[card_provider][converted_provider]

            prices.append(price[0] + kWh_charged * price[1])

        prices = [price / min(prices) for price in prices]

        return prices
    elif parameter == 'prices_absolute':
        kWh_charged = 14.4

        prices = []
        used = []
        for i, cs in enumerate(choices):
            provider = environment.css_info[cs]['provider']
            converted_provider = conversion_charge_point_operator[provider]
            price = all_prices[card_provider][converted_provider]

            prices.append(price[0] + kWh_charged * price[1])
        return prices
    elif parameter == 'distances':
        return [choices[choice] for choice in choices]
    elif parameter == 'distances_relative':
        max_distance = max([distance for cs, distance in choices.items()])
        if max_distance > 0:
            return [choices[choice] / max_distance for choice in choices]
        else:
            return [0 for choice in choices]
    elif parameter == 'options':
        return choices
    elif parameter == 'providers':
        providers = [environment.css_info[cs]['provider'] for cs in choices]
        return providers
    elif parameter == 'parking_zones':
        return [environment.css_info[choice]['parking_zone'] for choice in choices]
    elif parameter == 'inside_parking_zones':
        center_zone = get_zone(parking_zones_amsterdam, parking_zones_den_haag, center[0], center[1])
        return [environment.css_info[choice]['parking_zone'] == center_zone or environment.css_info[choice]['parking_zone'] ==  None for choice in choices]
    elif parameter == 'inside_parking_zones_freq':
        max_used_cp = '-1'
        max_used = 0
        for cp in agent['preferences'][center]:
            if agent['preferences'][center][cp] > max_used:
                max_used = agent['preferences'][center][cp]
                max_used_cp = cp
        center_zone = environment.css_info[max_used_cp]['parking_zone']
        return [environment.css_info[choice]['parking_zone'] == center_zone or environment.css_info[choice]['parking_zone'] ==  None for choice in choices]
    elif parameter == 'charging_speeds':
        speeds = []
        for cs in choices:
            charging_speed = -1
            try:
                if int(cs) in all_charging_speeds:
                    charging_speed = all_charging_speeds[int(cs)]
            except Exception:
                pass
            speeds.append(charging_speed)
        return speeds
    else:
        print('parameter (%s) is unknown' % parameter)
        return 0

def get_preferences(data, css):
    grouped_css = data.groupby('location_key').count()
    css_in_agent_sessions = data['location_key'].unique()
    return {cs: grouped_css.loc[grouped_css.index == cs].values[0][0]
        for cs in css if cs in css_in_agent_sessions}

def get_agents(number_of_agents):
    path_to_file = 'data/sessions/ChargeSessionsUnmerged_general.pkl'
    with open(path_to_file, 'rb') as data_file:
        general_data = pickle.load(data_file)

    start_date_training_data = pandas.to_datetime('01-01-2014', format = '%d-%m-%Y', errors = 'coerce')
    end_date_training_data = pandas.to_datetime('01-01-2016', format = '%d-%m-%Y', errors = 'coerce')

    general_data = general_data.loc[(general_data['start_connection'] >= start_date_training_data) &
        (general_data['end_connection'] <= end_date_training_data)]

    all_agents = []
    agent_database = 'data/agent_database/all_agents_unmerged_with_utrecht/'
    for agent_ID in list(set([files[:-4] for files in os.listdir(agent_database) if files[0] != '.']))[:number_of_agents]:
        with open(agent_database + agent_ID + '.pkl', 'rb') as agent_file:
            data = pickle.load(agent_file)

        data['training_sessions'] = general_data.loc[general_data.ID == agent_ID]
        data['ID'] = agent_ID

        data['preferences'] = {center: get_preferences(
            data['training_sessions'], data['centers_css'][center]['habit'])
            for center in data['centers_css']}
        all_agents.append(data)

    return all_agents

def plot_distribution(data, title, filename, x_axis):
    fig = plt.figure(figsize=(16, 8))
    seaborn.set(style='whitegrid', font_scale=2)

    # if isinstance(data[0], float) or isinstance(max(data), float):
    #     bins = numpy.arange(min(data), max(data) + (max(data) - min(data)) / 10, (max(data) - min(data)) / 10)
    # else:
    #     bins = range(min(data), max(data) + 1, 1)
    # print(bins)
    seaborn.distplot(pandas.Series(data, name = x_axis), kde = False, norm_hist = False)

    seaborn.despine(left=True)
    seaborn.plt.ylim(0,)
    plt.title(title, fontsize = 30)
    plt.savefig('selection_process_data_analysis/%s.png' % filename, bbox_inches='tight')

def plot_distribution_double(x_data, hue_data, title, filename, x_axis, hue, y_max = -1, stepsize = 1):
    fig = plt.figure(figsize=(16, 8))
    seaborn.set(style='whitegrid', font_scale=2)

    max_value = max(max(x_data), max(hue_data))
    seaborn.distplot(pandas.Series(x_data, name = x_axis), kde = False, norm_hist = False, bins = numpy.arange(0, max_value + stepsize, stepsize), label = x_axis)
    seaborn.distplot(pandas.Series(hue_data, name = hue), kde = False, norm_hist = False, bins = numpy.arange(0, max_value + stepsize, stepsize), label = hue)
    plt.legend()
    seaborn.despine(left=True)
    if y_max != -1:
         seaborn.plt.ylim(0, y_max)
    else:
        seaborn.plt.ylim(0,)
    plt.title(title, fontsize = 30)
    plt.savefig('selection_process_data_analysis/%s.png' % filename, bbox_inches='tight')


def plot_correlation(x_data, y_data, filename, x_axis, y_axis):
    seaborn.set(style='whitegrid', font_scale=2)
    grid = seaborn.jointplot(x = x_axis, y = y_axis, s = 100, stat_func = spearmanr, data = pandas.DataFrame({x_axis: x_data, y_axis: y_data}))
    grid.fig.set_figwidth(10)
    grid.fig.set_figheight(10)

    seaborn.despine(left = True)
    seaborn.plt.ylim(0,)
    seaborn.plt.xlim(0,)
    plt.savefig('selection_process_data_analysis/%s.png' % filename, bbox_inches='tight')

def plot_pointplot(x_data, y_data, filename, x_axis, y_axis):
    fig = plt.figure(figsize=(16, 8))
    seaborn.set(style='whitegrid', font_scale=2)

    seaborn.pointplot(x = x_axis, y = y_axis, capsize=.2, data = pandas.DataFrame({y_axis: y_data,
        x_axis: [round(d, 2) for d in x_data]}))

    seaborn.despine(left = True)
    seaborn.plt.ylim(0,)
    plt.ylabel(y_axis)
    plt.title('Effect of %s on %s' % (x_axis, y_axis), fontsize = 30)
    plt.savefig('selection_process_data_analysis/%s.png' % filename, bbox_inches='tight')

def main():
    plot_options, plot_providers, plot_prices, plot_distances, plot_speeds, plot_parking_zones = False, False, False, False, False,  True

    number_of_agents = 2313
    print('loading %s agents' % number_of_agents)
    all_agents = get_agents(number_of_agents)
    print('loaded %s agents' % number_of_agents)

    all_providers = get_all_providers(all_agents)
    all_prices = get_all_prices()
    all_charging_speeds = get_all_charging_speeds()
    all_card_providers = get_all_card_providers()
    raw_data = get_raw_data()
    with open('data/simulation_pkls/environment.pkl', 'rb') as environment_file:
        environment = pickle.load(environment_file)
    parking_zones_amsterdam, parking_zones_den_haag = get_zone_information()
    all_zones_den_haag = get_all_zones_den_haag(parking_zones_den_haag)

    results = {parameter: {'choices': [], 'walking_preparedness': [], 'total_choices': [], 'frequency_of_use': []} for parameter in ['distances', 'distances_relative', 'providers', 'parking_zones', 'inside_parking_zones', 'inside_parking_zones_freq', 'prices_relative', 'prices_absolute', 'charging_speeds', 'options']}
    counts = {parameter: [] for parameter in ['distances', 'distances_relative', 'providers', 'parking_zones', 'inside_parking_zones', 'inside_parking_zones_freq', 'prices_relative', 'prices_absolute', 'charging_speeds', 'options']}

    number_of_centers = 0

    all_found_card_providers= {}
    unknown = 0

    for agent in all_agents:
        try:
            card_provider = all_card_providers[agent['ID']]
            if card_provider in all_found_card_providers:
                all_found_card_providers[card_provider] += 1
            else:
                all_found_card_providers[card_provider] = 1
            card_provider = conversion_card_provider[card_provider]
        except Exception as e:
            unknown += 1
            card_provider = 'unknown'

        if card_provider == '?':
            card_provider = 'unknown'

        for center in agent['centers_css']:
            choices_in_center = agent['centers_css'][center]['distance']
            for parameter in counts:
                if not ((parameter == 'prices_relative' or parameter == 'prices_absolute') and card_provider == 'unknown'):
                    choices_in_center_based_on_parameter = get_choices(environment, agent, center, choices_in_center, parameter, all_prices, all_charging_speeds, card_provider, raw_data, parking_zones_amsterdam, parking_zones_den_haag)
                    results[parameter]['choices'].append(choices_in_center_based_on_parameter)
                    results[parameter]['walking_preparedness'].append(agent['walking_preparedness'])
                    results[parameter]['total_choices'].append(choices_in_center)
                    results[parameter]['frequency_of_use'].append([agent['preferences'][center][cs] if cs in agent['preferences'][center] else 0 for cs in choices_in_center])

                    counts[parameter].append(len(set(choices_in_center_based_on_parameter)))
            number_of_centers += 1

    print('----------')
    print('%d agents have %d centers.' % (number_of_agents, number_of_centers))
    for parameter in counts:
        print('%d centers have more than one choice based on %s.' % (len([value for value in counts[parameter] if value > 1]), parameter))


    if plot_options:
        ''' Options '''
        number_cps_per_center = [len(choice) for choice in results['options']['choices']]
        plot_distribution(number_cps_per_center,
            'Number of CPs per center (%d CPs in total)' % numpy.sum(number_cps_per_center),
            'options/cp_spead_%d' % number_of_agents, 'CP Spread')
        plot_correlation(results['options']['walking_preparedness'], number_cps_per_center,
            'options/correlation_walking_preparedness_cps_in_center_%d' % number_of_agents, 'Walking Preparedness', 'Number CPs per Center')

    if plot_providers:
        ''' Providers '''
        provider_combinations = {}
        for providers_in_center in results['providers']['choices']:
            set_providers = frozenset(providers_in_center)
            if set_providers in provider_combinations:
                provider_combinations[set_providers] += 1
            else:
                provider_combinations[set_providers] = 1

        print('all_found_card_providers = %s' % all_found_card_providers)
        print('all found charge pole providers in one center = %s' % provider_combinations)
        print('%d unknown card providers' % unknown)

    if plot_prices:
        ''' Prices '''
        total_choices_when_multiple_price_options = [len(results['prices_relative']['total_choices'][i]) for i in range(len(results['prices_relative']['total_choices'])) if len(set(results['prices_relative']['choices'][i])) > 1]
        if len(total_choices_when_multiple_price_options) > 1:
            plot_distribution_double([len(choice) for choice in results['prices_relative']['total_choices']],
                total_choices_when_multiple_price_options, 'Number of total options',
                'prices/histogram_of_total_options_with_price_options_%d' % number_of_agents, 'Total Options',
                'Options when multiple price options', y_max = 30)

            plot_distribution([price for prices in results['prices_absolute']['choices'] for price in prices],
                'Price spread (absolute)', 'prices/price_spead_absolute_%d' % number_of_agents, 'Price Spread')
            plot_distribution([price for prices in results['prices_relative']['choices'] for price in prices],
                'Price spread (relative)', 'prices/price_spead_relative_%d' % number_of_agents, 'Price Spread')
            plot_distribution([max(prices) - min(prices) for prices in results['prices_absolute']['choices'] if max(prices) - min(prices) > 0],
                'Max Price Difference', 'prices/max_price_difference_%d' % number_of_agents, 'Max Price Difference')
            relative_prices_multiple_options = [price for prices in results['prices_relative']['choices'] for price in prices if len(set(prices)) > 1]
            frequency_of_use_multiple_options = [use for i in range(len(results['prices_relative']['choices'])) if len(set(results['prices_relative']['choices'][i])) > 1 for use in results['prices_relative']['frequency_of_use'][i]]
            plot_correlation(relative_prices_multiple_options, frequency_of_use_multiple_options,
                'prices/correlation_relative_price_frequency_%d' % number_of_agents, 'Relative Price', 'Frequency of Use')
            plot_pointplot(relative_prices_multiple_options, frequency_of_use_multiple_options,
                'prices/pointplot_relative_price_frequency_%d' % number_of_agents, 'Relative Price', 'Frequency of Use')
        else:
            print('no choices when multiple_price_options')

    if plot_distances:
        ''' Distances '''
        distances_multiple_options = [distance for distances in results['distances']['choices'] for distance in distances if len(set(distances)) > 1]
        frequency_of_use_multiple_options = [use for i in range(len(results['distances']['choices'])) if len(set(results['distances']['choices'][i])) > 1 for use in results['distances']['frequency_of_use'][i]]

        plot_correlation(distances_multiple_options, frequency_of_use_multiple_options,
            'distances/correlation_distance_frequency_%d' % number_of_agents, 'Distances', 'Frequency of Use')

        distances_multiple_options = [distance for distances in results['distances_relative']['choices'] for distance in distances if len(set(distances)) > 1]
        frequency_of_use_multiple_options = [use for i in range(len(results['distances_relative']['choices'])) if len(set(results['distances_relative']['choices'][i])) > 1 for use in results['distances_relative']['frequency_of_use'][i]]

        plot_correlation(distances_multiple_options, frequency_of_use_multiple_options,
            'distances/correlation_relative_distance_frequency_%d' % number_of_agents, 'Relative Distances', 'Frequency of Use')

        relative_distance_unused_cps = [results['distances_relative']['choices'][i][j] for i in range(len(results['distances_relative']['choices'])) for j in range(len(results['distances_relative']['frequency_of_use'][i])) if results['distances_relative']['frequency_of_use'][i][j] == 0]

        plot_distribution(relative_distance_unused_cps,
            'Relative Distance Unused CPs', 'distances/relative_distance_unused_cps_%d' % number_of_agents, 'Relative Distance Unused CPs')

        relative_distance_used_cps = [results['distances_relative']['choices'][i][j] for i in range(len(results['distances_relative']['choices'])) for j in range(len(results['distances_relative']['frequency_of_use'][i])) if results['distances_relative']['frequency_of_use'][i][j] != 0]

        plot_distribution(relative_distance_used_cps,
            'Relative Distance Used CPs', 'distances/relative_distance_used_cps_%d' % number_of_agents, 'Relative Distance Used CPs')

    if plot_speeds:
        speeds_multiple_options = [distance for distances in results['charging_speeds']['choices'] for distance in distances if len(set(distances)) > 1 and distance != -1]
        frequency_of_use_multiple_options = [results['charging_speeds']['frequency_of_use'][i][j] for i in range(len(results['charging_speeds']['choices'])) if len(set(results['charging_speeds']['choices'][i])) > 1 for j in range(len(results['charging_speeds']['frequency_of_use'][i])) if results['charging_speeds']['choices'][i][j] != -1]

        plot_correlation(speeds_multiple_options, frequency_of_use_multiple_options,
            'speeds/correlation_speed_frequency_%d' % number_of_agents, 'Speeds', 'Frequency of Use')

        plot_pointplot(speeds_multiple_options, frequency_of_use_multiple_options,
            'speeds/pointplot_speed_frequency_%d' % number_of_agents, 'Speed', 'Frequency of Use')

    if plot_parking_zones:
        # data = [len(set(zones_in_center)) for zones_in_center in results['parking_zones']['choices']]
        # plot_distribution(data, 'Parking Zones in Center', 'parking_zones/parking_zones_in_center_%d' % number_of_agents, 'Parking Zones in Center')

        # multiple_options = [distance for distances in results['inside_parking_zones']['choices'] for distance in distances if len(set(distances)) > 1]
        # frequency_of_use_multiple_options = [results['inside_parking_zones']['frequency_of_use'][i][j] for i in range(len(results['inside_parking_zones']['choices'])) if len(set(results['inside_parking_zones']['choices'][i])) > 1 for j in range(len(results['inside_parking_zones']['frequency_of_use'][i]))]
        # plot_correlation(multiple_options, frequency_of_use_multiple_options, 'parking_zones/multiple_choices_correlation_parking_zone_frequency_%d' % number_of_agents, 'In parkingzone', 'Frequency of use')
        # plot_pointplot(multiple_options, frequency_of_use_multiple_options, 'parking_zones/multiple_choices_pointplot_parking_zone_frequency_%d' % number_of_agents, 'In parkingzone', 'Frequency of use')

        ratios = []
        count_choices_pz = 0
        for i in range(len(results['inside_parking_zones']['choices'])):
            if len(set(results['inside_parking_zones']['choices'][i])) > 1:
                inside_sessions = 0
                outside_sessions = 0
                count_choices_pz += 1
                for j in range(len(results['inside_parking_zones']['frequency_of_use'][i])):
                    if results['inside_parking_zones']['choices'][i][j]:
                        inside_sessions += results['inside_parking_zones']['frequency_of_use'][i][j]
                    else:
                        outside_sessions += results['inside_parking_zones']['frequency_of_use'][i][j]
                ratio = outside_sessions / (inside_sessions + outside_sessions)
                ratios.append(ratio * 100)
        plot_distribution(ratios, 'Percentage sessions outside parkingzone',
            'parking_zones/percentage_sessions_outside_%d' % number_of_agents, 'Percentage sessions outside parkingzone')

        # ratios = []
        # count_choices_pz = 0
        # for i in range(len(results['inside_parking_zones_freq']['choices'])):
        #     if len(set(results['inside_parking_zones_freq']['choices'][i])) > 1:
        #         inside_sessions = 0
        #         outside_sessions = 0
        #         count_choices_pz += 1
        #         for j in range(len(results['inside_parking_zones_freq']['frequency_of_use'][i])):
        #             if results['inside_parking_zones_freq']['choices'][i][j]:
        #                 inside_sessions += results['inside_parking_zones_freq']['frequency_of_use'][i][j]
        #             else:
        #                 outside_sessions += results['inside_parking_zones_freq']['frequency_of_use'][i][j]
        #         ratio = outside_sessions / (inside_sessions + outside_sessions)
        #         ratios.append(ratio * 100)
        # plot_distribution(ratios, 'Percentage sessions outside parkingzone',
        #     'parking_zones/percentage_sessions_outside_freq_%d' % number_of_agents, 'Percentage sessions outside parkingzone')


if __name__ == '__main__':
    main()
