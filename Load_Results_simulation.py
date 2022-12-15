"""
Load_Results_simunlation.py

Written by:
    Jurjen R Helmus
    
Last updated on November 2019

Description:
    The current model spits out pkl files which are not easy to read and convert into figures
    This python module contains scripts that 
    
    
"""

import pandas
import datetime
import IPython
import pandas as pd
import os

cwd = os.getcwd()




def get_data_initialization(clustering, measures, number_of_agents):    
    # save dataframe per parameter/value
    #or append per parameter/value
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        for parameter, values in clustering.items():
           
            for value in values:
                experiment_dir = cwd + "/data/experiment_results/clustering_metrics/experiment_initialization_measures_%s_varying_%s/" % (measure, parameter) 
                experiment_file = "%.2f_%s_%d_agents.pkl" % (value, parameter, number_of_agents)
                if os.path.isfile(experiment_dir + experiment_file) :

                    partial_results=pd.read_pickle(experiment_dir + experiment_file)

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value))

                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0)

                    tmpdf=tmpdf[0:0]
    
    
    tmp_all_data.to_csv(cwd + "/data/experiment_results/clustering_metrics/" + "clustering_metrics_agents.csv" )   
    return(tmp_all_data)







def get_and_save_data_initialization(clustering, measures, number_of_agents):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        for parameter, values in clustering.items():
           
            for value in values:
                experiment_dir = cwd + "/data/experiment_results/clustering_metrics/experiment_initialization_measures_%s_varying_%s/" % (measure, parameter) 
                experiment_file = "%.2f_%s_%d_agents.pkl" % (value, parameter, number_of_agents)
                if os.path.isfile(experiment_dir + experiment_file):

                    partial_results=pd.read_pickle(experiment_dir + experiment_file)

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0)

                    tmpdf=tmpdf[0:0]
                else:
                    print(experiment_dir + experiment_file)
    
    tmp_all_data.to_csv(cwd + "/data/experiment_results/clustering_metrics/" + "clustering_metrics_"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)







def get_and_save_data_time_retry_center_validation(parameter,values,measures,number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
           
        
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/time_retry_center_validation/experiment_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.2f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)

            if os.path.isfile(experiment_dir + experiment_file) :

                partial_results=pd.read_pickle(experiment_dir + experiment_file)


                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)


    tmp_all_data.to_csv(cwd + "/data/experiment_results/time_retry_center_validation/" + "time_retry_center_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)












def get_and_save_data_gap_sessions_validation(parameter,measures,values,number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/gap_sessions_validation/experiment_initialization_and_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.0f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file):
            
                partial_results=pd.read_pickle(experiment_dir + experiment_file)


                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]


    tmp_all_data.to_csv(cwd + "/data/experiment_results/gap_sessions_validation/" + "gap_sessions_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)











def get_and_save_data_initialization_and_simulation_binsize(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/bin_size_validation/experiment_initialization_and_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.0f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file):

                partial_results=pd.read_pickle(experiment_dir + experiment_file)


                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]


    tmp_all_data.to_csv(cwd + "/data/experiment_results/bin_size_validation/" + "binsize_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)



def get_and_save_data_initialization_and_simulation_habit_probability(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/habit_probability_validation/experiment_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.2f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file) :
                  
                partial_results=pd.read_pickle(experiment_dir + experiment_file)
            
                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.1f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)



    tmp_all_data.to_csv(cwd + "/data/experiment_results/habit_probability_validation/" + "habit_probability"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)


def get_and_save_data_initialization_and_simulation_minimum_radius(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/walking_preparedness_time_metrics_validation/experiment_initialization_and_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.0f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file) :
                  
                partial_results=pd.read_pickle(experiment_dir + experiment_file)
            
                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.0f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)



    tmp_all_data.to_csv(cwd + "/data/experiment_results/walking_preparedness_time_metrics_validation/" + "walking_preparedness_time_metrics_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)



def get_and_save_data_initialization_and_simulation_warm_up_period(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/warmup_period_validation/experiment_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%.2f_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file) :
                  
                partial_results=pd.read_pickle(experiment_dir + experiment_file)
            
                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.0f" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%.2f" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)



    tmp_all_data.to_csv(cwd + "/data/experiment_results/warmup_period_validation/" + "warmup_period_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)


def get_and_save_data_initialization_and_simulation_distance_measure_validation(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    
        
           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/distance_measure_validation/experiment_initialization_and_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%s_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            
            if os.path.isfile(experiment_dir + experiment_file) :
                  
                partial_results=pd.read_pickle(experiment_dir + experiment_file)
            
                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%s" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%s" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)



    tmp_all_data.to_csv(cwd + "/data/experiment_results/distance_measure_validation/" + "distance_measure_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)


def get_and_save_data_initialization_and_simulation_weighted_centers_validation(parameter,values,measures, number_of_agents,simulation_repeats):    
    
    tmp_all_data=pd.DataFrame()
    
    for measure in measures:
        
        tmpdf = pd.DataFrame({'measure' : []})
    

           
        for value in values:
            experiment_dir = cwd + "/data/experiment_results/distance_measure_validation/experiment_initialization_and_simulation_measures_%s_varying_%s/" % (measure, parameter) 
            experiment_file = "%s_%s_%d_agents_%d_simulation_repeats.pkl" % (value, parameter, number_of_agents,simulation_repeats)
            if os.path.isfile(experiment_dir + experiment_file) :
                  
                partial_results=pd.read_pickle(experiment_dir + experiment_file)
            
                if measure in ['agent_validation', 'charging_station_validation']:

                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%s" % value),index=['training','test'])

                    #first concat the training set 
                    tmpdf['measure']=pd.Series([measure +'_training']).repeat(len(partial_results['training'])) 
                    tmpdf['value']=partial_results['training']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['training']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['training']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty

                    #then concat the test set 
                    tmpdf['measure']=pd.Series([measure +'_test']).repeat(len(partial_results['test']))
                    tmpdf['value']=partial_results['test']
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results['test']))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results['test']))

                    tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                    tmpdf=tmpdf[0:0] #set temp to empty              

                else:
                    partial_results=pd.Series(partial_results,name=parameter +'_'+str("%s" % value)).values
                    tmpdf['measure']=pd.Series([measure]).repeat(len(partial_results))
                    tmpdf['value']=partial_results
                    tmpdf['setting']=pd.Series([value]).repeat(len(partial_results))
                    tmpdf['parameter']=pd.Series([parameter]).repeat(len(partial_results))


                tmp_all_data=pd.concat([tmp_all_data,tmpdf],  axis=0,sort=False)

                tmpdf=tmpdf[0:0]
            else:
                print(experiment_dir + experiment_file)



    tmp_all_data.to_csv(cwd + "/data/experiment_results/weighted_centers_validation/" + "weighted_centers_validation"+ str(number_of_agents) + "_agents.csv" )   
    return(tmp_all_data)



