import pandas as pd
import glob
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np
import sklearn.tree as st

path = '/Users/yujzhang/Downloads/backup/auckland_rugby_2018_game_data/game_data' # use your path
all_files = glob.glob(path + "/*.csv")

path_position = '/Users/yujzhang/Downloads/backup/auckland_rugby_2018_game_data/game_data/Player_Profiles_2019'
files = glob.glob(path_position + "/position.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


avg_dr_rank = frame.groupby(['Athlete'])['Distance Rate (m/min)'].mean().rank(ascending=False)
avg_sm_rank = frame.groupby(['Athlete'])['Speed Max (km/h)'].mean().rank(ascending=False)
avg_st_rank = frame.groupby(['Athlete'])['Sprints Total (num)'].mean().rank(ascending=False)
avg_hmt_rank = frame.groupby(['Athlete'])['HR Max  Total (bpm)'].mean().rank(ascending=False)
avg_al_rank = frame.groupby(['Athlete'])[' Athlete Load'].mean().rank(ascending=False)
avg_mp_rank = frame.groupby(['Athlete'])['Metabolic PowerPeak'].mean().rank(ascending=False)
avg_hia_rank = frame.groupby(['Athlete'])['Hi Int Acceleration'].mean().rank(ascending=False)
avg_bi_rank = frame.groupby(['Athlete'])['Body Impacts (num)'].mean().rank(ascending=False)
avg_hr_rank = frame.groupby(['Athlete'])['HIE Rate'].mean().rank(ascending=False)
avg_dz1_rank = frame.groupby(['Athlete'])['DistanceSpeed Zone 1  (m)'].mean().rank(ascending=False)
avg_dz2_rank = frame.groupby(['Athlete'])['DistanceSpeed Zone 2  (m)'].mean().rank(ascending=False)
avg_dz3_rank = frame.groupby(['Athlete'])['DistanceSpeed Zone 3  (m)'].mean().rank(ascending=False)
avg_ssz3_rank = frame.groupby(['Athlete'])['SprintsSpeed Zone 3 (num)'].mean().rank(ascending=False)
avg_bit_rank = frame.groupby(['Athlete'])['Body ImpactsBody Impacts Zone Total (num)'].mean().rank(ascending=False)
avg_bi2_rank = frame.groupby(['Athlete'])['Body ImpactsBody Impacts Grade 2 (num)'].mean().rank(ascending=False)


total_rank = (avg_dr_rank+avg_sm_rank+avg_st_rank+avg_hmt_rank+avg_al_rank+avg_mp_rank
      +avg_hia_rank+avg_bi_rank+avg_hr_rank+avg_dz1_rank+avg_dz2_rank+avg_dz3_rank
      +avg_ssz3_rank+avg_bit_rank+avg_bi2_rank).rank()

tr = pd.DataFrame(columns=['Athlete','total_rank'])

tr['Athlete'] = total_rank.index
tr['total_rank'] = total_rank.values

merged_data1 = pd.merge(left=frame,right=tr, how='left', left_on='Athlete', right_on='Athlete')

for file in files:
    df = pd.read_csv(file, index_col=None, header=0)

df['Athlete'] = df[['last', 'first']].apply(lambda x: ', '.join(x), axis=1)

df_position = pd.DataFrame(columns=['Athlete','Position'])
df_position['Athlete'] = df['Athlete']
df_position['Position'] = df['Position ']
print(df_position)

merged_data2 = pd.merge(left=merged_data1,right=df_position, how='left', left_on='Athlete', right_on='Athlete')

#print(merged_data2)

clean_data = merged_data2[-merged_data2['Position'].isna()]

#print(clean_data[clean_data['Position'] == 'Hooker'])

#avg_dr_rank_group = clean_data.groupby(['Position'])['Speed Max (km/h)']
#print(avg_dr_rank_group)
#print(avg_dr_rank_group)
#avg_sm_rank_group = clean_data.groupby(['Athlete','Position'])['Speed Max (km/h)'].mean().rank(ascending=False)
#avg_st_rank_group = clean_data.groupby(['Athlete','Position'])['Sprints Total (num)'].mean().rank(ascending=False)
#avg_hmt_rank_group = clean_data.groupby(['Athlete','Position'])['HR Max  Total (bpm)'].mean().rank(ascending=False)
#avg_al_rank_group = clean_data.groupby(['Athlete','Position'])[' Athlete Load'].mean().rank(ascending=False)
#avg_mp_rank_group = clean_data.groupby(['Athlete','Position'])['Metabolic PowerPeak'].mean().rank(ascending=False)
#avg_hia_rank_group = clean_data.groupby(['Athlete','Position'])['Hi Int Acceleration'].mean().rank(ascending=False)
#avg_bi_rank_group = clean_data.groupby(['Athlete','Position'])['Body Impacts (num)'].mean().rank(ascending=False)
#avg_hr_rank_group = clean_data.groupby(['Athlete','Position'])['HIE Rate'].mean().rank(ascending=False)
#avg_dz1_rank_group = clean_data.groupby(['Athlete','Position'])['DistanceSpeed Zone 1  (m)'].mean().rank(ascending=False)
#avg_dz2_rank_group = clean_data.groupby(['Athlete','Position'])['DistanceSpeed Zone 2  (m)'].mean().rank(ascending=False)
#avg_dz3_rank_group = clean_data.groupby(['Athlete','Position'])['DistanceSpeed Zone 3  (m)'].mean().rank(ascending=False)
#avg_ssz3_rank_group = clean_data.groupby(['Athlete','Position'])['SprintsSpeed Zone 3 (num)'].mean().rank(ascending=False)
#avg_bit_rank_group = clean_data.groupby(['Athlete','Position'])['Body ImpactsBody Impacts Zone Total (num)'].mean().rank(ascending=False)
#avg_bi2_rank_group = clean_data.groupby(['Athlete','Position'])['Body ImpactsBody Impacts Grade 2 (num)'].mean().rank(ascending=False)


#group_rank = (avg_dr_rank_group+avg_sm_rank_group+avg_st_rank_group+avg_hmt_rank_group+avg_al_rank_group+avg_mp_rank_group
#      +avg_hia_rank_group+avg_bi_rank_group+avg_hr_rank_group+avg_dz1_rank_group+avg_dz2_rank_group+avg_dz3_rank_group
#      +avg_ssz3_rank_group+avg_bit_rank_group+avg_bi2_rank_group).rank()

#print(group_rank)

index = [7,8,9,11,12,13,15,20,21,22,23,24,25,26]

#index = [7,8,9,11]

fn_dy = merged_data2.columns[index]

#index = [7,8,9,11,12,13,14]

x = clean_data[clean_data['Position'] == 'Lock'].iloc[:,index]
#print(x)

#x = clean_data.iloc[:,index]

y = clean_data['total_rank'][clean_data['Position'] == 'Lock']
#print(y)

#y = clean_data['total_rank']

su.shuffle(x, y, random_state=7)


train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = x[:train_size], x[train_size:], y[:train_size], y[train_size:]

#RandomForestRegressor
model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
model.fit(train_x, train_y)


#feature importance
fi_dy = model.feature_importances_
#print(fi_dy)


#Test if the model is fitting well
pred_test_y = model.predict(test_x)
print("The r2 score of RandomForestRegresson is :")
print(sm.r2_score(test_y, pred_test_y))


#DecisionTreeRegressor
#4Layers                                  
model2 = st.DecisionTreeRegressor(max_depth=4)
#Train
model2.fit(train_x,train_y)
fi_dt = model2.feature_importances_
#print(fi_dt)
#Predict
pred_test_y2 = model2.predict(test_x)
print("The r2 score of DecisionTreeRegression is :") 
print(sm.r2_score(test_y, pred_test_y2)) 


#AdaBoostRegressor
model3 = se.AdaBoostRegressor(
        st.DecisionTreeRegressor(max_depth=4),
        n_estimators=400,random_state=7)
model3.fit(train_x,train_y)
fi_ab = model3.feature_importances_
#print(fi_ab)
pred_test_y3 = model3.predict(test_x)
print("The r2 score of AdaBoostRegression is :")
print(sm.r2_score(test_y, pred_test_y3)) 



mp.figure('Random Forest', facecolor='lightgray')
mp.subplot(211)
mp.title('Random Forest', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi_dy.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos, fi_dy[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
mp.xticks(pos, fn_dy[sorted_indices],
          rotation=90)
mp.tight_layout()
mp.show()


mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('Decision Tree', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi_dt.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos, fi_dt[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
mp.xticks(pos, fn_dy[sorted_indices],
          rotation=90)
mp.tight_layout()
mp.show()

mp.figure('AdaBoost Feature Importance', facecolor='lightgray')
mp.subplot(212)
mp.title('AdaBoost Decision Tree', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi_ab.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos, fi_ab[sorted_indices], facecolor='indianred', edgecolor='indianred')
mp.xticks(pos, fn_dy[sorted_indices],
          rotation=90)
mp.tight_layout()
mp.show()


