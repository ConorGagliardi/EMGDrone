#!/usr/bin/env python3


#Created by John Pesarchick. 5/6/2023

import rospy
import time
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from ros_myo.msg import EmgArray
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
#from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
#The ros_myo package must be installed before running this program! Additionally, have the rawNode.py program running as well. 


#emg_publisher = rospy.Publisher('/EMG', JointState, queue_size=10)
#emg = JointState()
#imu_publisher = rospy.Publisher('/IMU', Imu, queue_size=10)
#imu = Imu()
#emg.name = ["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"]
#emg.position = [0,0,0,0,0,0,0,0]
move_publisher = rospy.Publisher('/COMMAND', String, queue_size=1)
n = 0 #Index for collected emg samples
j = 0 #Index for collected imu samples
maxits = 99 #Maximum number of samples to be collected at once 
emg_df = pd.DataFrame()
imu_df = pd.DataFrame()

#Here's all the SVM training stuff------------------------

#Read in the data and perform some filtering and feature extraction...all of this runs once only at the start of the program. 
datapath = '/home/ros-admin/jfp1222_ws/src/ros_myo/scripts/Data/Data/Data'
participants = ['/John','/Aharon','/Conor']
datapath2 = '/One/Thalmic Labs MyoMyo'
actions = ['/backward_','/down_','/fist_','/forward_','/left_','/right_','/resting_','/rotate_down_','/rotate_up_',
'/rotate_right_','/rotate_left_','/twist_left_','/twist_right_','/stop_']
actions_enum = enumerate(actions)
suffix = '_data.csv'
#emg_df = pd.DataFrame()
#imu_df = pd.DataFrame()
e_df = pd.DataFrame()
i_df = pd.DataFrame()
data = []
index = 0
#
#EMG callback function: this runs every time a new EMG measurement has been taken. 
def emg_cb(e):
    #emg_data = []
    global n,j, emg_df, imu_df, maxits
    data = {}
	#Record a new sample if not enough samples have been accumulated. 
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
	#If both EMG and IMU have enough samples, perform classification via the assimilate function
        assimilate(emg_df,imu_df)
	#When classification is complete, reset everything. 
        n = 0
        j = 0
        emg_df = pd.DataFrame()
        imu_df = pd.DataFrame()
    rospy.sleep(0.0125) 

#IMU Callback function: this runs every time a new IMU measurement has been received. 
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
        #data['Roll'] = i.angular_velocity.x
        #data['Pitch'] = i.angular_velocity.y
        #data['Yaw'] = i.angular_velocity.z
        imu_df = pd.concat([imu_df,pd.DataFrame(data,index=[0])],ignore_index=True)
        j = j+1
        #print("n=",n,"j=",j)
    #else:
        #print("Stopped recording IMU")
        #print(imu_df)

    if(n>=maxits and j>=maxits):
	#If both EMG and IMU have enough samples, perform classification via the assimilate function. 
        assimilate(emg_df,imu_df)
        #When classification is complete, reset everything. 
	n = 0
        j = 0
        emg_df = pd.DataFrame()
        imu_df = pd.DataFrame()

    #print("IMU")
    #imu_publisher.publish(imu)
    rospy.sleep(0.0125)

#This is the same function used to test the SVM in jfp1222_EE636_ProjectCU2.ipnyb
def filteremg(emg, notch, sfreq, high_band, low_band):
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
    high_band = (notch - 3)/(sfreq/2)
    low_band = (notch + 3)/(sfreq/2)
    #...and the filter is creted here.
    b2, a2 = sp.signal.butter(4, [high_band,low_band], btype='bandstop',analog=True)
    emg_envelope = sp.signal.lfilter(b2, a2, emg_rectified)
    return emg_envelope

#Feature extraction:
def getFeatures(df):
    features_emg = {}
    features_imu = {}

    for i in range(1,9):
        #Max EMG channel amplitude
        features_emg['EMG_'+str(i)+'_Max'] = max(df['EMG_'+str(i)])

        #Mean channel amplitude value
        features_emg['EMG_'+str(i)+'_Avg'] = df['EMG_'+str(i)].mean()
        #Channel amplitude standard deviation
        features_emg['EMG_'+str(i)+'_Std'] = df['EMG_'+str(i)].std()
        #Max channel power
        f, Pxx_den = sp.signal.periodogram(df['EMG_' + str(i)], 200) #Default smp rate is 200

        features_emg['EMG_'+str(i)+'_MaxPower'] = max(Pxx_den)

    #IMU Data
    for direction in ['X','Y','Z']:
        #Range in direction
        features_imu['dOrnt_'+direction] = max(df['Orientation_'+direction]) - min(df['Orientation_'+direction])
        #Range in gyro readings
        features_imu['dGyro'+direction] = max(df['Gyro_'+direction]) - min(df['Gyro_'+direction])

        #Max acceleration
        features_imu['Acc_'+direction] = max(abs(df['Acc_'+direction]))
        #features_imu['Acc_'+direction] = max((df['Acc_'+direction])) - min((df['Acc_'+direction]))

    return features_emg, features_imu

def command(move):
    #This function sends a command to a simulator by equating the numerical output from the SVM to a string: 
    #Python sucks
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
        output = "right"
    elif move == 6:
        output = "resting"
    elif move == 7: 
        output = "rotate_down"
    elif move == 8: 
        output = "rotate_up" 
    elif move == 9: 
        output = "rotate_right" 
    elif move == 10: 
        output = "rotate_left"
    elif move == 11:
        output = "twist_left" 
    elif move == 12:
        output = "twist_right" 
    elif move == 13:
        output = "stop"      
    elif move == 14:
        output = "up"  
    else:
        output = "stop"         

    move_publisher.publish(output)
    rospy.sleep(0.001)

def assimilate(emg,imu):
    global clf, actions
    #This function should take the completed dataframes, get their features, and perform a classification
    #print(emg)
    #print(imu)
    print("STARTING CLASSIFICATION")
    #rospy.sleep(5) 
    df = pd.concat([emg, imu], ignore_index=False,axis=1)
	#Filter the received signals
    for k in range(1,9):
        df['EMG_'+str(k)] = filteremg(df['EMG_'+str(k)],notch=60,sfreq=200,high_band=20,low_band=95)
	#Get their features
    efeats,ifeats = getFeatures(df)
    #print(ifeats)
    #print(efeats)
    #Need to combine these two dictionaries and get their raw values. The order should be the same as that used in training. 
    efeats = {**efeats,**ifeats}
    
	#Perform a classification using the extracted features
    data = list(efeats.values())
    data = np.array(data)
    #print("Length of data:",data)
    result = clf.predict(data.reshape(1,-1))
    print(result[0],"=",actions[result[0]])
    command(result[0])
    print("Next gesture in 3...")
    rospy.sleep(1)
    print("Next gesture in 2...")
    rospy.sleep(1)
    print("Next gesture in 1...")
    rospy.sleep(1)
    print("Start!")

#This does the same training procedure on an SVM as jfp1222_EE636_ProjectCU2.ipnyb
#'cause I was too lazy to figure out how to save a model 
print("STARTING TRAINING")
for person in participants:
    #For each move performed...
    for move in actions:
        for i in range(1,6): #For each file...
            df = pd.read_csv(datapath+person+datapath2+move+str(i)+suffix) #...read it...
            for j in range(1,9):
                #...filter each EMG channel according to the function above...
                df['EMG_' + str(j)] = filteremg(df['EMG_' + str(j)], notch=60, sfreq=200, high_band=20, low_band=95)
                #...and extract features for each data type.
                emg, imu = getFeatures(df)
                #if move == '/down_' and person == '/John' and i == 3:
                #    print(emg)
                #    print(imu)
                emg['Class'] = move #The class is saved as an index.
                emg['Participant'] = person
                imu['Participant'] = person
                imu['Class'] = move
                #Add results to developing dataframes.
                e_df = pd.concat([e_df, pd.DataFrame(emg, index=[0])], ignore_index=True)
                i_df = pd.concat([i_df, pd.DataFrame(imu, index=[0])], ignore_index=True)
        index += 1 #
    index = 0

#Now, try SVM with both data types concurrently.
combo_df = pd.concat([e_df, i_df], ignore_index=False,axis=1)
combo_df.drop('Class', axis=1, inplace=True)
combo_df.drop('Participant', axis=1, inplace=True)

#SVM learning begins here for the combined data.
groups = e_df['Participant'].values # Specify groups - the same groups in the same order as the EMG df are present, so
#that df is used here. The concat with the two dataframes produces two columns with
#the class and participant labels.
#X = combo_df.drop(columns=['Class', 'Participant']).values # specify Feature columns
X = combo_df.values
le = LabelEncoder() #CReate labels
le.fit(e_df['Class'].values) # Specify Classes - the EMG df is used here for the same reason.
y = le.transform(e_df['Class'].values)
logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)
#predicted = np.array([])
#true = np.array([])
clf = make_pipeline(StandardScaler(),
PCA(n_components=0.9, svd_solver='full'),
SVC(kernel='linear', gamma='auto', class_weight='balanced'))

for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
clf.fit(X_train, y_train)
print("FINISHED TRAINING")

if __name__ == '__main__':
    # test()
    rospy.init_node('emg_bar_plot')
    rospy.loginfo("Printing plot of EMG values...")

    emg_sub = rospy.Subscriber('/myo_raw/myo_emg',
                               EmgArray,
                               emg_cb,
                               queue_size=1)

    imu_sub = rospy.Subscriber('/myo_raw/myo_imu',
                               Imu,
                               imu_cb,
                               queue_size=1)

    rospy.loginfo("Awaiting publications...")
    rospy.spin()
