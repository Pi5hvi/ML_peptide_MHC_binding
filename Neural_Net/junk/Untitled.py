#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import csv
import pandas as pd
import tensorflow as tf
import re
from tensorflow import keras
import matplotlib.pyplot as plt
#print('done')


# In[4]:


## Thing to grab all of our data from the ic50.csv file
## Thing to grab all of our data from the ic50.csv file

def datagrabber():
    with open('ic50.csv', mode='r') as f:
        r = csv.reader(f)
        row_count = sum(1 for row in r)
        row_count = row_count-1
        
    with open('ic50.csv', mode='r') as f:
        r = csv.reader(f)
        ic50_MHC = ["" for x in range(row_count)]
        ic50_pept = ["" for x in range(row_count)]
        ic50_HL = ["" for x in range(row_count)]
        j=-2
        deleted=0
        for row in r:
            i=0
            j=j+1
            k=0
            for cell in row:
                
                
                
                if i==11:#This is the peptide
                    if j==0:
                        ic50_pept[j]='Pept'
                    else:
                        if len(str(cell))<=15:
                            try:
                                remove_plus=re.search('.*\+', str(cell))
                                remove_plus=str(remove_plus[0])
                                ic50_pept[j]=(remove_plus[:-2])
                            except:
                                ic50_pept[j]=str(cell)
                        else:
                            k=1
                            j=j-1
                            deleted=deleted+1
                            print(deleted)
                            print(str(cell))
                        
                    
                    

                elif i==80 and k==0:#This is the MHC
                    ic50_MHC[0]='MHC'
                    try:
                        allel_type=re.search('.*HLA-',str(cell))#Look for the key term
                        MHC_name=str(cell).replace(allel_type[0], '')
                        if MHC_name[0]=='A':
                            MHC_name='.*'+MHC_name[0]+'\\'+MHC_name[1:]
                            with open('A_prot.fasta') as file:
                                file_contents=file.read()
                                allel=re.search(MHC_name,file_contents)
                                junk1_MHC_junk2=file_contents[allel.end():(allel.end()+1000)]
                                junk1=re.search('.*bp',str(junk1_MHC_junk2))
                                MHC_junk2=str(junk1_MHC_junk2).replace(junk1[0], '')
                                MHC_junk2=MHC_junk2[1:]
                                junk2=re.search('.*HLA:HLA',MHC_junk2)
                                MHC=MHC_junk2[:junk2.start()-1]
                                MHC=str(MHC).replace("\n","")
                                ic50_MHC[j]=MHC
                                #print('*********************************************'+'\n')
                        elif MHC_name[0]=='B':
                            MHC_name='.*'+MHC_name[0]+'\\'+MHC_name[1:]
                            with open('B_prot.fasta') as file:
                                file_contents=file.read()
                                allel=re.search(MHC_name,file_contents)
                                junk1_MHC_junk2=file_contents[allel.end():(allel.end()+1000)]
                                junk1=re.search('.*bp',str(junk1_MHC_junk2))
                                MHC_junk2=str(junk1_MHC_junk2).replace(junk1[0], '')
                                MHC_junk2=MHC_junk2[1:]
                                junk2=re.search('.*HLA:HLA',MHC_junk2)
                                MHC=MHC_junk2[:junk2.start()-1]
                                MHC=str(MHC).replace("\n","")
                                ic50_MHC[j]=MHC
                                #MAVMA...TACKV
                                #print('*********************************************'+'\n')
                        else:
                            MHC_name='.*'+MHC_name[0]+'\\'+MHC_name[1:]
                            with open('C_prot.fasta') as file:
                                file_contents=file.read()
                                allel=re.search(MHC_name,file_contents)
                                junk1_MHC_junk2=file_contents[allel.end():(allel.end()+1000)]
                                junk1=re.search('.*bp',str(junk1_MHC_junk2))
                                MHC_junk2=str(junk1_MHC_junk2).replace(junk1[0], '')
                                MHC_junk2=MHC_junk2[1:]
                                junk2=re.search('.*HLA:HLA',MHC_junk2)
                                MHC=MHC_junk2[:junk2.start()-1]
                                MHC=str(MHC).replace("\n","")
                                ic50_MHC[j]=MHC
                                #MAVMA...TACKV
                                print('.'+'\n')
                    except:   
                        print('exptn:'+str(cell))
                        
                        

                elif i==72 and k==0:#This is the HL
                    if str(cell) != '':
                        if j==0:
                            ic50_HL[j]='HL'
                        else:
                            ic50_HL[j]=str(cell)
                    elif j>=3:
                        j=j-1
                        #print('blank')
                        k=1
                        
                elif i==71:#This is the >/< coloumn
                    if str(cell) != '' and j>=3:
                        #print(str(cell))
                        if str(cell) != '=':#we dont want these readings
                            #print(str(cell)+'yep')
                            j=j-1
                            k=1
                    else:
                        #we good
                        j=j
            
            
                i=i+1
                
    with open('test.csv', mode='w') as f:
        w=csv.writer(f)
        for i in range (row_count):
            row=[]
            if i==0:
                row=['MHC','Peptide','HL']
            else:
                row.append(ic50_MHC[i])
                row.append(ic50_pept[i])
                row.append(ic50_HL[i])
            w.writerow(row)
#    print(ic50_MHC)
#    print(ic50_HL)
#    print(ic50_pept)
print('Starting up datagrabber...\n')
datagrabber()
print('Done')


# In[7]:


## File conversion and getting the maximum length of an MHC



def datacleaner():

    #First find the number of rows in our input file
    with open('test.csv', mode='r') as f:
        r = csv.reader(f)
        row_count = sum(1 for row in r)
        row_count = row_count-1
    
    #Now, import all the data into arrays of strings called MHCstrs,peptstrs,HLstrs
    with open('test.csv', mode='r') as f:
        r = csv.reader(f)
        maxMHC=0
        maxpept=0
        MHCstrs = ["" for x in range(row_count)]
        peptstrs = ["" for x in range(row_count)]
        HLstrs = ["" for x in range(row_count)]
        j=-2
        for row in r:
            i=0
            j=j+1
            for cell in row:
                if i==0: #This is the MHC
                    m=re.search('.*SHS',str(cell))#This makes it so we look only after SHS
                    try:
                        MHC=str(cell).replace(m[0], '')
                        MHCstrs[j]=MHC
                        MHClength=(len(MHC))
                        if maxMHC<MHClength:
                            maxMHC=MHClength                    
                    except:
                        if str(cell)== 'MHC':#If we are in the titles row, this isn't an error
                            i=i# As mentioned before, this isn't really an error
                        else:
                            #HLA*30:02 edited to have SHS
                            print('NO SHS found for'+str(cell),j) # Incase there is no SHS in our MHC
                elif i==1: #This is the peptide
                    peptstrs[j]=str(cell)
                    if maxpept<len(str(cell)):
                        maxpept=len(str(cell))
                elif i==2:
                    try:
                        HLstrs[j]=cell
                        #HLstrs[j]=int(str(cell),10)
                    except:
                        i=i #This will happen when we are in the titles row. Not a real error, so ignore
                i=i+1
                
    #Now we can write all of our data to a new file, input.csv which has a seperate colom for each AA.
    with open('input.csv', mode='w') as f:
        w=csv.writer(f)
        for i in range (row_count+1):
            row=[]
            if i==0:
                for j in range (maxMHC):
                    row.append('MHC'+str(j))
                for j in range (maxpept):
                    row.append('pept'+str(j))
                row.append("HL")
            
            else:
                for j in range (maxMHC):
                    try:
                        row.append(MHCstrs[i-1][j])
                    except:
                        row.append("")
                for j in range (maxpept):
                    try:
                        row.append(peptstrs[i-1][j])
                    except:
                        row.append("")
                row.append(HLstrs[i-1])
            w.writerow(row)


    return maxMHC,maxpept

#Importing the maxMHC and maxpept variables and taking input to pandas
print('Cleaning data...')

maxMHC,maxpept=datacleaner()
ds=pd.read_csv("input.csv")
    

print('Done')


# In[ ]:





# In[8]:


#Pandas converting all the MHC and peptide coloumns to 1HCs

for i in range(maxMHC):

    mhc=ds.pop('MHC'+str(i))
    ds['MHC'+str(i)+'=A']=(mhc=='A')*1.0
    ds['MHC'+str(i)+'=B']=(mhc=='B')*1.0
    ds['MHC'+str(i)+'=C']=(mhc=='C')*1.0
    ds['MHC'+str(i)+'=D']=(mhc=='D')*1.0
    ds['MHC'+str(i)+'=E']=(mhc=='E')*1.0
    ds['MHC'+str(i)+'=F']=(mhc=='F')*1.0
    ds['MHC'+str(i)+'=G']=(mhc=='G')*1.0
    ds['MHC'+str(i)+'=H']=(mhc=='H')*1.0
    ds['MHC'+str(i)+'=I']=(mhc=='I')*1.0
    ds['MHC'+str(i)+'=K']=(mhc=='K')*1.0
    ds['MHC'+str(i)+'=L']=(mhc=='L')*1.0
    ds['MHC'+str(i)+'=M']=(mhc=='M')*1.0
    ds['MHC'+str(i)+'=N']=(mhc=='N')*1.0
    ds['MHC'+str(i)+'=P']=(mhc=='P')*1.0
    ds['MHC'+str(i)+'=Q']=(mhc=='Q')*1.0
    ds['MHC'+str(i)+'=R']=(mhc=='R')*1.0
    ds['MHC'+str(i)+'=S']=(mhc=='S')*1.0
    ds['MHC'+str(i)+'=T']=(mhc=='T')*1.0
    ds['MHC'+str(i)+'=U']=(mhc=='U')*1.0
    ds['MHC'+str(i)+'=V']=(mhc=='V')*1.0
    ds['MHC'+str(i)+'=W']=(mhc=='W')*1.0
    ds['MHC'+str(i)+'=X']=(mhc=='X')*1.0
    ds['MHC'+str(i)+'=Y']=(mhc=='Y')*1.0
    ds['MHC'+str(i)+'=Z']=(mhc=='Z')*1.0
    
for i in range(maxpept):

    mhc=ds.pop('pept'+str(i))
    ds['pept'+str(i)+'=A']=(mhc=='A')*1.0
    ds['pept'+str(i)+'=B']=(mhc=='B')*1.0
    ds['pept'+str(i)+'=C']=(mhc=='C')*1.0
    ds['pept'+str(i)+'=D']=(mhc=='D')*1.0
    ds['pept'+str(i)+'=E']=(mhc=='E')*1.0
    ds['pept'+str(i)+'=F']=(mhc=='F')*1.0
    ds['pept'+str(i)+'=G']=(mhc=='G')*1.0
    ds['pept'+str(i)+'=H']=(mhc=='H')*1.0
    ds['pept'+str(i)+'=I']=(mhc=='I')*1.0
    ds['pept'+str(i)+'=K']=(mhc=='K')*1.0
    ds['pept'+str(i)+'=L']=(mhc=='L')*1.0
    ds['pept'+str(i)+'=M']=(mhc=='M')*1.0
    ds['pept'+str(i)+'=N']=(mhc=='N')*1.0
    ds['pept'+str(i)+'=P']=(mhc=='P')*1.0
    ds['pept'+str(i)+'=Q']=(mhc=='Q')*1.0
    ds['pept'+str(i)+'=R']=(mhc=='R')*1.0
    ds['pept'+str(i)+'=S']=(mhc=='S')*1.0
    ds['pept'+str(i)+'=T']=(mhc=='T')*1.0
    ds['pept'+str(i)+'=U']=(mhc=='U')*1.0
    ds['pept'+str(i)+'=V']=(mhc=='V')*1.0
    ds['pept'+str(i)+'=W']=(mhc=='W')*1.0
    ds['pept'+str(i)+'=X']=(mhc=='X')*1.0
    ds['pept'+str(i)+'=Y']=(mhc=='Y')*1.0
    ds['pept'+str(i)+'=Z']=(mhc=='Z')*1.0

print('Done')


# In[41]:


train_ds=ds.sample(frac=0.8,random_state=0)
test_ds=ds.drop(train_ds.index)
train_labels=train_ds.pop('HL')
test_labels=test_ds.pop('HL')
print('Done')

train_ds.keys()


# In[48]:


#model_hl=tf.keras.models.Sequential()
#model_hl.add(tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=[2]))
#model_hl.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
#model_hl.add(tf.keras.layers.Dense(1))
model = keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,
                                kernel_initializer='random_uniform', bias_initializer='zeros',
                                input_shape=[len(train_ds.keys())]))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

    
optimizer=tf.keras.optimizers.RMSprop(0.1)
    
model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse','acc'])
print('Done')

model.summary()


# In[49]:


#class PrintDot(keras.callbacks.Callback):
#    def on_epoch_end(self,epoch,logs):
#        if epoch % 100 == 0: print('')
#        print('.',end='')

EPOCHS = 100

model.fit(train_ds, train_labels, epochs=EPOCHS, validation_split=0.2,verbose=1)
print('Done')


# In[80]:





# In[50]:


predictions = model.predict([train_ds])
print(predictions)


# In[51]:


train_ds.shape


# In[ ]:




