# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:35:44 2023

@author: ajinkya
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 00:20:27 2022

@author: ajinkya
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:30:24 2022

@author: ajinkya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from tkinter import *
from tkinter import messagebox
import sys 
import urllib
import urllib.request

symptoms_list=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

df = pd.read_csv('E:/gsf/New folder/dataset.csv')
print(df.head())
#df.describe()
df1 = pd.read_csv('E:/gsf/New folder/Symptom-severity.csv')
print(df1.head())

df.isna().sum()
df.isnull().sum()

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)

df = df.fillna(0)
df.head()

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

#converig to formatale data
d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
df = d.replace('foul_smell_of urine',0)
df.head()

(df[cols] == 0).all()

df['Disease'].value_counts()

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = SVC()
model.fit(x_train, y_train)

preds = model.predict(x_test)
print(preds)

conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm)

def message():
    if (Symptom_1.get() == "None" and  Symptom_2.get() == "None" and Symptom_3.get() == "None" and Symptom_4.get() == "None" and Symptom_5.get() == "None"):
        messagebox.showinfo("ERROR MESSAGE!!", "ENTER  SYMPTOMS PLEASE")
    else :
        SVM()

def SVM():
    psymptoms = [Symptom_1.get(),Symptom_2.get(),Symptom_3.get(),Symptom_4.get(),Symptom_5.get()]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
    psy = [psymptoms + nulls]

    pred2 = model.predict(psy)
    t3.delete("1.0", END)
    t3.insert(END, pred2[0])

    
#GUI    
root = Tk()
root.title(" Disease Prediction From Symptoms")
root.configure(background='#FCE6C9')

Symptom_1 = StringVar()
Symptom_1.set(None)
Symptom_2 = StringVar()
Symptom_2.set(None)
Symptom_3 = StringVar()
Symptom_3.set(None)
Symptom_4 = StringVar()
Symptom_4.set(None)
Symptom_5 = StringVar()
Symptom_5.set(None)


w2 = Label(root, justify=LEFT, text="Disease Prediction using Machine Learning", fg="RED", bg="#FCE6C9")
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=2, column=0, columnspan=2, padx=100)

S1Lb = Label(root,  text="Symptom 1",fg="BLACK", bg="WHITE")
S1Lb.config(font=("Helvetica", 15))
S1Lb.grid(row=7, column=1, pady=10 , sticky=W)

S2Lb = Label(root,  text="Symptom 2",fg="BLACK", bg="WHITE")
S2Lb.config(font=("Helvetica", 15))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root,  text="Symptom 3",fg="BLACK", bg="WHITE")
S3Lb.config(font=("Helvetica", 15))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root,  text="Symptom 4",fg="BLACK", bg="WHITE")
S4Lb.config(font=("Helvetica", 15))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root,  text="Symptom 5",fg="BLACK", bg="WHITE")
S5Lb.config(font=("Helvetica", 15))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)

lr = Button(root, text="Predict",height=2, width=20, command=message)
lr.config(font=("Helvetica", 15))
lr.grid(row=12, column=1,pady=10)

S5Lb = Label(root,  text="Result",fg="BLACK", bg="GREEN")
S5Lb.config(font=("Helvetica", 22))
S5Lb.grid(row=25, column=1, pady=50, sticky=W)

OPTIONS = sorted(symptoms)

#OPTIONS = ["fatigue", "yellowish_skin", "loss_of_appetite", "yellowing_of_eyes", 'family_history',"stomach_pain", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]


S1En = OptionMenu(root, Symptom_1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom_2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom_3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom_4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom_5,*OPTIONS)
S5En.grid(row=11, column=1)

t3 = Text(root, height=2, width=20)
t3.config(font=("Helvetica", 20))
t3.grid(row=25, column=1 , padx=20)

root.mainloop()