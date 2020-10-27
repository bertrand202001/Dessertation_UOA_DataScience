import pandas as pd
import glob
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np
import sklearn.tree as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from math import pi
from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold

path = '/Users/yujzhang/Downloads/backup/auckland_rugby_2018_game_data/game_data' # use your path
all_files = glob.glob(path + "/*.csv")

path_position = '/Users/yujzhang/Downloads/backup/auckland_rugby_2018_game_data/game_data/Player_Profiles_2019'
files = glob.glob(path_position + "/position.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


for file in files:
    df = pd.read_csv(file, index_col=None, header=0)

df['Athlete'] = df[['last', 'first']].apply(lambda x: ', '.join(x), axis=1)

df_position = pd.DataFrame(columns=['Athlete','Position','Rank'])
df_position['Athlete'] = df['Athlete']
df_position['Position'] = df['Position ']
df_position['Rank'] = 100 - 10*df['Rank']

merged_data2 = pd.merge(left=frame,right=df_position, how='left', left_on='Athlete', right_on='Athlete')


clean_data = merged_data2[-merged_data2['Position'].isna()]


positions = ['Loose Head','Tight Head','Hooker','Lock','Blindside Flanker','Openside Flanker',
             'Number 8','Scrum Half','Fly Half','Inside Centre','Outside Centre',
             'Winger','Full Back']


#duration_total = int(clean_data['Duration  Total (min:sec)'].split(":")[0])*60+int(clean_data['Duration  Total (min:sec)'].split(":")[1])

dt = []
for dts in clean_data['Duration  Total (min:sec)']:
    if dts == '' or dts == '**':
        dtm = 0
        dt.append(dtm)
    else:
        dtm = int(dts.split(":")[0])*60 + int(dts.split(":")[1])
        dt.append(dtm)
    
    
print(dt)
clean_data['Duration_Total'] = dt

dsh = []
for dshs in clean_data['Duration Speed Hi-Inten (min:sec)']:
    if dshs == '' or dshs == '**':
        dshtp = 0
        dsh.append(dshtp)
    else:
        dshtp = int(dshs.split(":")[0])*60 + int(dshs.split(":")[1])
        dsh.append(dshtp)
clean_data['Duration_Speed_HiInten'] = dsh
        
dhh = []
for dhhs in clean_data['Duration HR Hi-Inten (min:sec)']:
    if dhhs == '' or dhhs == '**':
        dhhtp = 0
        dhh.append(dhhtp)
    else:
        dhhtp = int(dhhs.split(":")[0])*60 + int(dhhs.split(":")[1])
        dhh.append(dhhtp)
clean_data['Duration_HR_HiInten'] = dhh

sdt = []
for sdts in clean_data['Speed Duration  Total (min:sec)']:
    if sdts == '' or sdts == '**':
        sdttp = 0
        sdt.append(sdttp)
    else:
        sdttp = int(sdts.split(":")[0])*60 + int(sdts.split(":")[1])
        sdt.append(sdttp)
clean_data['Speed_Dutation_Total'] = sdt

hdt = []
for hdts in clean_data['HR Duration  Total (min:sec)']:
    if hdts == '' or hdts == '**':
        hdttp = 0
        hdt.append(hdttp)
    else:
        hdttp = int(hdts.split(":")[0])*60 + int(hdts.split(":")[1])
        hdt.append(hdttp)
clean_data['HR_Dutation_Total'] = hdt

dz4 = []
for dz4s in clean_data['DurationHR Zone 4 (min:sec)']:
    if dz4s == '' or dz4s == '**':
        dz4tp = 0
        dz4.append(dz4tp)
    else:
        dz4tp = int(dz4s.split(":")[0])*60 + int(dz4s.split(":")[1])
        dz4.append(dz4tp)
clean_data['DurationHR_Zone_4'] = dz4

dz5 = []
for dz5s in clean_data['DurationHR Zone 5 (min:sec)']:
    if dz5s == '' or dz5s == '**':
        dz5tp = 0
        dz5.append(dz5tp)
    else:
        dz5tp = int(dz5s.split(":")[0])*60 + int(dz5s.split(":")[1])
        dz5.append(dz5tp)
clean_data['DurationHR_Zone_4'] = dz5


MHR = []
for MHRs in clean_data['% MaxHR']:
    if MHRs == '' or MHRs == '**':
        MHRtp = 0
        MHR.append(MHRtp)
    else:
        MHRtp = MHRs.split("%")[0]
        MHR.append(MHRtp)
clean_data['MaxHR'] = MHR

WRR = []
for WRRs in clean_data['Work Recovery Ratio (ratio)']:
    if WRRs == '' or WRRs == '**':
        WRRtp = 0
        WRR.append(WRRtp)
    else:
        WRRtp = (int(WRRs.split(":")[0]))/(int(WRRs.split(":")[1]))
        WRR.append(WRRtp)
clean_data['Work_Recovery_Ratio'] = WRR

print(clean_data.columns)


clean_data3 = clean_data.replace('**',0)



#=============================================

attributs_list = ['Sprints HR Hi-Inten (num)']

for attr in attributs_list:
    mp.figure(attr, figsize=(18, 12.5))   
    bplot=sns.stripplot(y=attr, x='Position', 
                       data=clean_data3, 
                       jitter=True, 
                       marker='o', 
                       alpha=0.5)
    
    sns.boxplot(y=attr, x='Position', 
                     data=clean_data3, 
                     color="white")
    
    attr_name = attr.replace(" ","")
    
    if attr_name.find("/"):
        attr_name = attr.replace("/","_")        
    elif attr_name.find("-"):
        attr_name = attr.replace("-","_")
       
    print(attr_name)
    
    mp.savefig('/Users/yujzhang/Downloads/backup/raw_data_plot/raw_data_'+attr_name+'.png', format='png')
    #mp.show()
#mp.xticks(rotation=90)

#=============================================

       
       
       

        