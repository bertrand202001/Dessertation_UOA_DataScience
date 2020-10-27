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

clean_data2 = clean_data.drop(columns=['Duration  Total (min:sec)',
                                       'Duration Speed Hi-Inten (min:sec)',
                                       'Duration HR Hi-Inten (min:sec)',
                                       '% MaxHR',
                                       'Speed Duration  Total (min:sec)',
                                       'HR Duration  Total (min:sec)',
                                       'DurationHR Zone 4 (min:sec)',
                                       'DurationHR Zone 5 (min:sec)',
                                       'Work Recovery Ratio (ratio)'])


clean_data3 = clean_data2.replace('**',0)


#=============================================

#attributs_list = ['HR Max  Total (bpm)', ' Athlete Load', 'Metabolic PowerPeak','Hi Int Acceleration', 'Hi Int Deceleration', 'Impact Rate (Imp/min)',
#       'Body Impacts (num)', 'Hi Intensity Effort', 'HIE Rate',
#       'DistanceSpeed Zone 1  (m)', 'DistanceSpeed Zone 2  (m)',
#       'DistanceSpeed Zone 3  (m)', 'DistanceSpeed Zone 4  (m)',
#       'DistanceSpeed Zone 5  (m)', 'SprintsSpeed Zone 3 (num)',
#       'SprintsSpeed Zone 4 (num)', 'SprintsSpeed Zone 5 (num)',
##       'AccelerationsAcceleration Zone 3 (num)',
#       'AccelerationsAcceleration Zone 4 (num)',
#       'AccelerationsAcceleration Zone 5 (num)',
#       'DecelerationsDeceleration Zone 3 (num)',
#       'DecelerationsDeceleration Zone 4 (num)',
#       'DecelerationsDeceleration Zone 5 (num)',
#       'Body ImpactsBody Impacts Zone Total (num)',
#       'Body ImpactsBody Impacts Grade 1 (num)',
#       'Body ImpactsBody Impacts Grade 2 (num)',
#       'Body ImpactsBody Impacts Grade 3 (num)',
#       'Body ImpactsBody Impacts Grade 4 (num)',
#       'Body ImpactsBody Impacts Grade 5 (num)', 'Duration_Total', 
 #      'Duration_Speed_HiInten', 'Duration_HR_HiInten',
 #      'Speed_Dutation_Total','HR_Dutation_Total','DurationHR_Zone_4',
 #      'MaxHR','Work_Recovery_Ratio']

#for attr in attributs_list:
#    mp.figure(attr, figsize=(18, 12.5))   
#    bplot=sns.stripplot(y=attr, x='Position', 
 #                      data=clean_data3, 
#                      jitter=True, 
#                       marker='o', 
#                       alpha=0.5)
    
 #   sns.boxplot(y=attr, x='Position', 
#                     data=clean_data3, 
#                     color="white")
    
#    attr_name = attr.replace(" ","")
#    
#    if attr_name.find("/"):
#        attr_name = attr.replace("/","_")        
 #   elif attr_name.find("-"):
#        attr_name = attr.replace("-","_")
 #       
 #   print(attr_name)
    
 #   mp.savefig('/Users/yujzhang/Downloads/backup/raw_data_plot/raw_data_'+attr_name+'.png', format='png')

#mp.xticks(rotation=90)

#=============================================
 
 
def draw_plot(title, fn_dy, values, angles,ax):
    
    ax.set_title(title, fontsize=40)
    
    ax.title.set_position([.5, 1.1])
    ax.yaxis.labelpad = 25
    
     
    # Draw one axe per variable + add labels labels yet
    mp.xticks(angles[:-1], fn_dy, color='grey', size=10)
    
    values = list(values)
    values.append(values[0])

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
     
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_rotation(i*90)
    
    ax.set_rmax(2)
    ax.set_rticks([])
    ticks= np.linspace(0,360,46)[:-1] 
    ax.set_xticks(np.deg2rad(ticks))
    
    
    ticklabels = fn_dy
    ax.set_xticklabels(ticklabels, fontsize=15)
    
    mp.gcf().canvas.draw()
    mp.title(title, fontsize=50).set_position([.5, 1.2])
    angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y-.15, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])
    
    
    ax.set_rlabel_position(0)
    y_max = max(values)
    mp.yticks([y_max/4,y_max/2,y_max*3/4], ["{0:.2f}".format(y_max/4),"{0:.2f}".format(y_max/2),"{0:.2f}".format(y_max*3/4)], color="grey", size=7)
    mp.ylim(0,y_max)
    
    

def create_radar_plot(fi_dy,fn_dy):
    
    N = len(fn_dy)

     
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values1=fi_dy
    values1 += values1[:1]
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    values2=fi_dtc
    values2 += values2[:1]
    
    values3=fi_abc
    values3 += values3[:1]
     
    mp.figure('Feature Importance', figsize=(60, 30))
    mp.suptitle('Feature Importance of ' + position, size=56)
    # Initialise the spider plot
    ax1 = mp.subplot(131, polar=True)
    draw_plot('Random Forest', fn_dy, values1, angles, ax1)
    
    ax2 = mp.subplot(132, polar=True)
    draw_plot('Decision Tree', fn_dy, values2, angles, ax2)
    
    ax3 = mp.subplot(133, polar=True)
    draw_plot('ADA Boosting', fn_dy, values3, angles, ax3)
    
    
    mp.savefig('/Users/yujzhang/Downloads/backup/radarplot'+position+'.png', format='png')
    mp.close()


def RFE_Validation(model,rf):
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(train_x,train_y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    
    rf.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #mp.savefig('/Users/yujzhang/Downloads/backup/RFEAUCPlot'+position+'.png', format='png')
    
    count = 0
    rfe_features = []
    for fs in rfecv.support_:
        if fs:
            #print(x.columns[count])
            rfe_features.append(x.columns[count])
            
        count += 1
    
    print("The features selected by RFE are : ")
    print(rfe_features)



for position in positions:
    
    
    group_data = clean_data3[clean_data3['Position'] == position]  
    
    index = list(range(4,41)) + list(range(43,51))
    
    
    fn_dy = group_data.columns[index]
    x = group_data.iloc[:,index]
    y = group_data['Rank']
    
    su.shuffle(x, y, random_state=7)
    
    
    train_size = int(len(x) * 0.7)
    train_x, test_x, train_y, test_y = x[:train_size], x[train_size:], y[:train_size], y[train_size:]
    
    #RandomForest
    model = se.RandomForestClassifier(n_estimators=50, max_depth=6,
                             random_state=5)
    model.fit(train_x, train_y) 
    print(model)
    fi_dy = model.feature_importances_
    print(fi_dy)
    pred_test_y = model.predict(test_x)
    print("The importance of Random Forest classifier is : " + str(fi_dy))
    print(fi_dy)
    print("The F1 score of Random Forest classifier is : " + str(f1_score(test_y, pred_test_y, average=None)))
    print("The accuracy score of Random Forest classifier is : " + str(accuracy_score(test_y, pred_test_y)))
      
    dtc = DecisionTreeClassifier()
    dtc.fit(train_x,train_y)
    fi_dtc = dtc.feature_importances_
    pred_test_y2_dtc = dtc.predict(test_x)
    print("The importance of Decision Tree classifier is : " + str(fi_dtc))
    print("The F1 score of Decision Tree classifier is : " + str(f1_score(test_y, pred_test_y2_dtc, average=None)))
    print("The accuracy score of Decision Tree classifier is : " + str(accuracy_score(test_y, pred_test_y2_dtc)))
      
    abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    model3 = abc.fit(train_x, train_y)
    fi_abc = model3.feature_importances_
    y_pred_abc = model.predict(test_x)    
    print("The importance of ADA Boosting classifier is : " + str(fi_abc))
    print("The F1 score of ADA Boosting classifier is : " + str(f1_score(test_y,y_pred_abc, average=None)))
    print("The accuracy score of ADA Boosting classifier is : " + str(accuracy_score(test_y, y_pred_abc)))
    
    create_radar_plot(fi_dy,fn_dy)
       
    #==================================barplot=====================================
       
    mp.figure('Feature Importance', figsize=(18, 12.5))
    mp.subplot(131)
    mp.title('Random Forest', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_dy.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_dy[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
    mp.xticks(pos, fn_dy[sorted_indices],
              rotation=90,fontsize=12)
    mp.tight_layout()
    
    mp.subplot(132)
    mp.title('Decision Tree', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_dtc.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_dtc[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
    mp.xticks(pos, fn_dy[sorted_indices],
          rotation=90)
    mp.tight_layout()
    
    
    mp.subplot(133)
    mp.title('AdaBoost Decision Tree', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_abc.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_abc[sorted_indices], facecolor='indianred', edgecolor='indianred')
    mp.xticks(pos, fn_dy[sorted_indices],
              rotation=90)
    mp.tight_layout()
    
    mp.savefig('/Users/yujzhang/Downloads/backup/plot'+position+'.png', format='png')
    mp.close()
    
    #======================================barplot=====================================
    
    #======================================trade off===================================
    
    index1 = []
    tmp = 0
    attr_rf = []
    attr_rf2 = []
    attr_rf3 = []
    for id in index:
        index1.append(id)
        fn_dy1 = group_data.columns[index1]
        x1 = group_data.iloc[:,index1]
        y1 = group_data['Rank']
        
        su.shuffle(x1, y1, random_state=7)        
        
        train_size1 = int(len(x1) * 0.9)
        train_x1, test_x1, train_y1, test_y1 = x1[:train_size1], x1[train_size1:], y1[:train_size1], y1[train_size1:]
        
        model1 = se.RandomForestClassifier(n_estimators=50, max_depth=6,
                             random_state=5)
        model1.fit(train_x1, train_y1) 

        pred_test_y1 = model1.predict(test_x1)
        ascore = accuracy_score(test_y1, pred_test_y1)
        
        if ascore >= tmp:
            tmp = ascore
            attr_rf.append(group_data.columns[id])
    
    print("=======================================")
    print("The best factors are : ")
    print(attr_rf)
        
    index2 = []
    tmp2 = 0
    attr_rf2 = []
    for id2 in index:
        index2.append(id2)
        fn_dy2 = group_data.columns[index2]
        x2 = group_data.iloc[:,index2]
        y2 = group_data['Rank']
        
        su.shuffle(x2, y2, random_state=7)        
        
        train_size2 = int(len(x2) * 0.9)
        train_x2, test_x2, train_y2, test_y2 = x2[:train_size2], x2[train_size2:], y2[:train_size2], y2[train_size2:]
               
        model2 = DecisionTreeClassifier()
        model2.fit(train_x2, train_y2) 

        pred_test_y2 = model2.predict(test_x2)
        ascore2 = accuracy_score(test_y2, pred_test_y2)
        
        if ascore2 >= tmp:
            tmp2 = ascore2
            attr_rf.append(group_data.columns[id])
    
    print("=======================================")
    print("The best factors are : ")
    print(attr_rf2)
    
    
    index3 = []
    tmp3 = 0
    attr_rf3 = []
    for id3 in index:
        index3.append(id3)
        fn_dy3 = group_data.columns[index3]
        x3 = group_data.iloc[:,index3]
        y3 = group_data['Rank']
        
        su.shuffle(x3, y3, random_state=7)        
        
        train_size3 = int(len(x3) * 0.9)
        train_x3, test_x3, train_y3, test_y3 = x3[:train_size3], x3[train_size3:], y3[:train_size3], y3[train_size3:]
               
        model3 = DecisionTreeClassifier()
        model3.fit(train_x3, train_y3) 

        pred_test_y3 = model3.predict(test_x3)
        ascore3 = accuracy_score(test_y3, pred_test_y3)
        
        if ascore3 >= tmp:
            tmp3 = ascore3
            attr_rf.append(group_data.columns[id])
    
    print("=======================================")
    print("The best factors are : ")
    print(attr_rf3)
    
    #==========================================trade off==================================
    
    #corrmat = group_data.corr()
    #top_corr_features = corrmat.index
    #mp.figure(figsize=(20,20))
    #g=sns.heatmap(group_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    
    
    #bestfeatures = SelectKBest(score_func=chi2, k=10)
    #fit = bestfeatures.fit(x,y)
    #dfscores = pd.DataFrame(fit.scores_)
    #dfcolumns = pd.DataFrame(x.columns)
    #concat two dataframes for better visualization 
    #featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    #featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(10,'Score'))
    
    #rfe = RFE(model, 10)
    #fit = rfe.fit(x, y)
    #print("Num Features: %d"% fit.n_features_) 
    
    #count = 0
    #for fs in fit.support_:
     #   if fs:
      #      print(x.columns[count])
            
       # count += 1
       
    mp.figure("Accuracy Curve of RFE", figsize=(15, 6))
    #mp.suptitle("Accuracy Curve of RFE ( " + position + " )", y=1.1, fontsize = 18)
    mp.xlabel("Number of features selected")
    mp.ylabel("Cross validation score")
       
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv.fit(train_x,train_y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    #mp.subplot(131)
   # mp.title("Random Forest")
    #mp.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #mp.savefig('/Users/yujzhang/Downloads/backup/RFEAUCPlot'+position+'.png', format='png')
    #mp.tight_layout()
  
    
    count = 0
    rfe_features = []
    for fs in rfecv.support_:
        if fs:
            rfe_features.append(x.columns[count])
            
        count += 1
    
    print("The features selected by RFE are : ")
    print(rfe_features)
    
   #================================================== 
    rfecv1 = RFECV(estimator=dtc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv1.fit(train_x,train_y)
    
    print("Optimal number of features by RFE(Decision Tree) : %d" % rfecv1.n_features_)
    
    #mp.subplot(132)
    #mp.title("Decision Tree")
    #mp.plot(range(1, len(rfecv1.grid_scores_) + 1), rfecv1.grid_scores_)
    #mp.savefig('/Users/yujzhang/Downloads/backup/RFEAUCPlot'+position+'.png', format='png')
    #mp.tight_layout()
  #
    
    count = 0
    rfe_features1 = []
    for fs in rfecv1.support_:
        if fs:
            rfe_features1.append(x.columns[count])
            
        count += 1
    
    print("The features selected by RFE are : ")
    print(rfe_features1)
    
    
    #==================================================
    
    rfecv2 = RFECV(estimator=abc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
    rfecv2.fit(train_x,train_y)
    
    print("Optimal number of features by RFE(ADA Boosting) : %d" % rfecv2.n_features_)
    
    #mp.subplot(133)
    #mp.title("ADA Boosting")
    #mp.plot(range(1, len(rfecv2.grid_scores_) + 1), rfecv2.grid_scores_)
    #mp.savefig('/Users/yujzhang/Downloads/backup/RFEAUCPlot'+position+'.png', format='png')
    #mp.close()
    #mp.show()
    
    count = 0
    rfe_features2 = []
    for fs in rfecv2.support_:
        if fs:
            rfe_features2.append(x.columns[count])
            
        count += 1
    
    print("The features selected by RFE are : ")
    print(rfe_features2)
       
       
       

        