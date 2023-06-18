#!/usr/bin/env python3


#Original created by John Pesarchick. 5/6/2023
# Edited by Aharon Sebton to work with his ML model. 5/8/2023

import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from ros_myo.msg import EmgArray
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import os
import sklearn
import mne
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#The ros_myo package must be installed before running this program! Additionally, have the rawNode.py program running as well. 


#emg_publisher = rospy.Publisher('/EMG', JointState, queue_size=10)
#emg = JointState()
#imu_publisher = rospy.Publisher('/IMU', Imu, queue_size=10)
#imu = Imu()
#emg.name = ["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"]
#emg.position = [0,0,0,0,0,0,0,0]
move_publisher = rospy.Publisher('/COMMAND', String, queue_size=10)
n = 0 #Index for collected emg samples
j = 0 #Index for collected imu samples
maxits = 99 #Maximum number of samples to be collected at once 
emg_df = pd.DataFrame()
imu_df = pd.DataFrame()

#Here's all the SVM training stuff------------------------

#Read in the data and perform some filtering and feature extraction...all of this runs once only at the start of the program. 
# datapath = '/home/sebtona/biorobotics_ws/src/ros_myo-master/scripts/Data'
# participants = ['/John','/Aharon','/Conor']
# datapath2 = '/One/Thalmic Labs MyoMyo'
actions = ['/backward_','/down_','/fist_','/forward_','/left_','/right_','/resting_','/rotate_down_','/rotate_up_',
'/rotate_right_','/rotate_left_','/twist_left_','/twist_right_','/stop_']
actions_enum = enumerate(actions)
# suffix = '_data.csv'
emg_df = pd.DataFrame()
imu_df = pd.DataFrame()
data = []
index = 0
#

def emg_cb(e):
    #emg_data = []
    global n,j, emg_df, imu_df, maxits
    data = {}
    if n < maxits:
        for idx, d in enumerate(e.data):
            data['EMG_'+str(idx+1)] = d
            
        emg_df = pd.concat([emg_df,pd.DataFrame(data,index=[0])],ignore_index=True)
        n = n+1
        #print("n=",n,"j=",j)
    #else:
        #print("Stopped recording EMG")
        #print(emg_df)
    #print("EMG")
    #print(emg)
    #print("n=\n",n)
    #print("j=\n",j)
    #emg_publisher.publish(emg)
       
    if(n>=maxits and j>=maxits):

        assimilate(emg_df,imu_df)
        n = 0
        j = 0
        emg_df = pd.DataFrame()
        imu_df = pd.DataFrame()
    rospy.sleep(0.0025) 

#It is evident that emg and imu samples are not collected evenly for whatever reason (i.e. one index may count faster than another).
#A solution for this out of sync problem: for each subscriber, concatinate only when n or j are below a threshold. 
#it is necessary for the two to produce an even number of samples. 
#furthermore, you can reduce the number of errors by looking out for a minimum value of samples
    #so instead of >maxtime and j==i, 
    #do >maxtime and j>blah and i > blah
    #even if one keeps counting up, either sub should stop recording after its limit has been reached. 
    #will small sync differences between imu and emg matter? probably not if we're making judgements on entire groups of data. 
    #Then why does the classifier still suck so much? 
def imu_cb(i):
    global n,j,emg_df,imu_df, maxits
    data = {}

    if j < maxits:
        #these assignments should correspond with those in our recordings...
        data['Orientation_W'] = i.orientation.w
        data['Orientation_X'] = i.orientation.x
        data['Orientation_Y'] = i.orientation.y
        data['Orientation_Z'] = i.orientation.z

        data['Acc_X'] = i.linear_acceleration.x
        data['Acc_Y'] = i.linear_acceleration.y
        data['Acc_Z'] = i.linear_acceleration.z

        data['Gyro_X'] = i.angular_velocity.x
        data['Gyro_Y'] = i.angular_velocity.y
        data['Gyro_Z'] = i.angular_velocity.z
        # data['Roll'] = i.angular_velocity.x
        # data['Pitch'] = i.angular_velocity.y
        # data['Yaw'] = i.angular_velocity.z
        imu_df = pd.concat([imu_df,pd.DataFrame(data,index=[0])],ignore_index=True)
        j = j+1
        #print("n=",n,"j=",j)
    else:
        print("Stopped recording IMU")
        #print(imu_df)

    if(n>=maxits and j>=maxits):

        assimilate(emg_df,imu_df)
        n = 0
        j = 0
        emg_df = pd.DataFrame()
        imu_df = pd.DataFrame()

    #print("IMU")
    #imu_publisher.publish(imu)
    rospy.sleep(0.0025)

def filteremg(emg, notch=60, quality=60, sfreq=200, high_band=20, low_band=95):
    """
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    # Zero mean emg signal
    emg = emg - emg.mean()# normalise cut-off frequencies to sampling frequency
    high_band = high_band/(sfreq/2)
    low_band = low_band/(sfreq/2)
    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass', analog=True)
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    # create notch filter and apply to rectified signal to get EMG envelope
    #the cutoff for the filter is defined by the following two lines...
    # high_band = (notch - 3)/(sfreq/2)
    # low_band = (notch + 3)/(sfreq/2)
    notch = notch/(sfreq)
    #...and the filter is creted here.
    # b2, a2 = sp.signal.butter(4, [high_band,low_band], btype='bandstop',analog=True)
    b2, a2 = sp.signal.iirnotch(notch, quality, fs=sfreq)
    emg_envelope = sp.signal.lfilter(b2, a2, emg_rectified)
    return emg_envelope

# Root Mean Square
def get_RMS(signal):
    ans = np.sqrt(np.average(np.square(signal)))
    return ans

# Waveform Length
def get_WL(signal):
    ans = np.log(sum([np.absolute(signal[i+1] - signal[i]) for i in range(len(signal)-1)]))
    return ans

# Mean Absolute Value
def get_MAV(signal):
    ans = np.average(np.absolute(signal))
    return ans

# Variance
def get_VAR(signal):
    ans = np.var(signal)
    return ans

# Max Power
def get_MAXP(signal):
    f, Pxx_den = sp.signal.periodogram(signal, 200)
    ans = max(Pxx_den)
    return ans

# Range of Values
def get_dRange(signal):
    ans = max(signal) - min(signal)
    return ans

# Max Value
def get_MAX(signal):
    ans = max(signal)
    return ans

#Feature extraction:
def getFeatures(df):
    features_emg = {}
    features_imu = {}

    for i in range(1,9):
    #EMG Data
        features_emg['EMG_'+str(i)+'_RMS'] = get_RMS(df['EMG_'+str(i)].dropna().reset_index(drop=True))
        features_emg['EMG_'+str(i)+'_WL'] = get_WL(df['EMG_'+str(i)].dropna().reset_index(drop=True))
        features_emg['EMG_'+str(i)+'_MAV'] = get_MAV(df['EMG_'+str(i)].dropna().reset_index(drop=True))
        features_emg['EMG_'+str(i)+'_VAR'] = get_VAR(df['EMG_'+str(i)].dropna().reset_index(drop=True))
        features_emg['EMG_'+str(i)+'_MAXP'] = get_MAXP(df['EMG_'+str(i)].dropna().reset_index(drop=True))
        features_emg['EMG_'+str(i)+'_MAX'] = get_MAX(df['EMG_'+str(i)].dropna().reset_index(drop=True))

    #IMU Data
    for direction in ['X','Y','Z']:
        #Range in direction
        features_imu['Orientation_'+direction+'_dOrient'] = get_dRange(df['Orientation_'+direction].dropna().reset_index(drop=True))
        #Range in gyro readings
        features_imu['Gyro_'+direction+'_dGyro'] = get_dRange(df['Gyro_'+direction].dropna().reset_index(drop=True))
        #Variance in all IMU readings
        features_imu['Orientation_'+direction+'_VAR'] = get_VAR(df['Orientation_'+direction].dropna().reset_index(drop=True))
        features_imu['Gyro_'+direction+'_VAR'] = get_VAR(df['Gyro_'+direction].dropna().reset_index(drop=True))
        features_imu['Acc_'+direction+'_VAR'] = get_VAR(df['Acc_'+direction].dropna().reset_index(drop=True))
        #Max acceleration
        features_imu['Acc_'+direction+'_MAX'] = get_MAX(df['Acc_'+direction].dropna().reset_index(drop=True))
    #Return the extracted features in two dictionaries
    return features_emg, features_imu

def command(move):
    #This function sends a command to a simulator by equating the numerical output from the SVM to a string:
    if move == 0:
        output = "backward"
    elif move == 1: 
        output = "down"
    elif move == 2: 
        output = "fist"
    elif move == 3:
        output = "forward"
    elif move == 4:
        output = "left"
    elif move == 5: 
        output = "resting"
    elif move == 6:
        output = "right"
    elif move == 7: 
        output = "rotate_down"
    elif move == 8: 
        output = "rotate_left" 
    elif move == 9: 
        output = "rotate_right" 
    elif move == 10: 
        output = "rotate_up"
    elif move == 11:
        output = "stop" 
    elif move == 12:
        output = "twist_left" 
    elif move == 13:
        output = "twist_right"      
    elif move == 14:
        output = "up"            

    move_publisher.publish(output)
    rospy.sleep(0.001)

def assimilate(emg,imu):
    global clf, actions
    #This function should take the completed dataframes, get their features, and perform a classification
    #This should be where the publisher lives. 
    #print(emg)
    #print(imu)
    print("STARTING CLASSIFICATION")
    #rospy.sleep(5) 
    df = pd.concat([emg, imu], ignore_index=False,axis=1)
    # Aharon Preprocessing
    emg_keys = ['EMG_' + str(i) for i in range(1, 9)]
    imu_keys = ['Orientation_X','Orientation_Y','Orientation_Z','Gyro_X','Gyro_Y','Gyro_Z','Acc_X', 'Acc_Y', 'Acc_Z']
    df[emg_keys] = df[emg_keys].apply(filteremg, raw=True)
    # for k in range(1,9):
    #     df['EMG_'+str(k)] = filteremg(df['EMG_'+str(k)],notch=60,quality=60,sfreq=200,high_band=20,low_band=95)
    for col in emg_keys:
        df[col] = df[col].rolling(20).mean()
    for col in imu_keys:
        df[col] = df[col].rolling(5).mean()
    efeats,ifeats = getFeatures(df)
    #print(ifeats)
    #print(efeats)
    #Need to combine these two dictionaries and get their raw values. The order should be the same as that used in training. 
    #efeats = {**efeats,**ifeats}
    
    edata = list(efeats.values())
    idata = list(ifeats.values())
    edata = np.array(edata)
    idata = np.array(idata)
    #print("Length of data:",data)
    result = clf.predict({'EMG_Input': edata.reshape(1,-1), 'IMU_Input': idata.reshape(1,-1)})
    result = np.argmax(result, axis=1)+1
    print(result[0],"=",actions[result[0]])
    command(result[0])
    print("Next gesture in 3...")
    rospy.sleep(1)
    print("Next gesture in 2...")
    rospy.sleep(1)
    print("Next gesture in 1...")
    rospy.sleep(1)
    print("Start!")

clf = keras.models.load_model('Aharon_Fusion_Model')

if __name__ == '__main__':
    # test()
    rospy.init_node('emg_bar_plot')
    rospy.loginfo("Printing plot of EMG values...")

    emg_sub = rospy.Subscriber('/myo_raw/myo_emg',
                               EmgArray,
                               emg_cb,
                               queue_size=10)

    imu_sub = rospy.Subscriber('/myo_raw/myo_imu',
                               Imu,
                               imu_cb,
                               queue_size=10)

    rospy.loginfo("Awaiting publications...")
    rospy.spin()
