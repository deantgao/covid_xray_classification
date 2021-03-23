#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image 
from sklearn import metrics
import scipy.stats as stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from skimage.io import imread, imshow
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from itertools import combinations, permutations
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc


# In[2]:


covid_path = '/Users/deangao/Desktop/CSM226/covidProject_data/covid/augmented/images/'
norm_path = '/Users/deangao/Desktop/CSM226/covidProject_data/normal/'
pneu_path = '/Users/deangao/Desktop/CSM226/covidProject_data/pneumonia/'


# In[3]:


# covid_imgs = []
# for p in os.listdir(covid_path):
# #     img = Image.open(covid_path + 'images/' + p)
#     covid_imgs.append(p)
# #     print(p)

# c_pixels = []
# for c_img in covid_imgs:
#     print(c_img)
#     data = imread(c_img, as_gray=True)
#     c_pixels.append(data)


# In[4]:


img_train = pd.read_csv('/Users/deangao/Desktop/CSM226/covidProject_data/radiomics/train_covid_normal_pn.csv')
img_test = pd.read_csv('/Users/deangao/Desktop/CSM226/covidProject_data/radiomics/test_covid_normal_pn.csv')
all_data = pd.concat([img_train, img_test], axis=0)

img_train = img_train.drop(columns=['id', 'Entropy', 'Uniformity', 'Energy'])
img_test = img_test.drop(columns=['id', 'Entropy', 'Uniformity', 'Energy'])
all_data = all_data.drop(columns=['id', 'Entropy', 'Uniformity', 'Energy'])


# In[5]:


train_X, train_y = img_train.iloc[:, :-1], img_train.iloc[:, -1]
test_X, test_y = img_test.iloc[:, :-1], img_test.iloc[:, -1]
labels = {0: 'normal lung', 1: 'pneumonia lung', 2: 'covid lung'}

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_X)

scaler2 = MinMaxScaler()
scaled_all = scaler2.fit_transform(all_data.iloc[:, :-1])
scaled_all_data = pd.concat([pd.DataFrame(all_data.iloc[:, -1].reset_index()), pd.DataFrame(scaled_all, columns=train_X.columns)], axis=1)
scaled_all_data = scaled_all_data.drop(['index'], axis=1)


# In[6]:


all_data.shape
all_data.iloc[:, -1]
pd.DataFrame(scaled_all)
# pd.DataFrame(all_data.iloc[:, -1])
# pd.concat([pd.DataFrame(all_data.iloc[:, -1]), pd.DataFrame(scaled_all)], axis=1)
scaled_all_data


# In[7]:


def flag_different_distributions(img_train):
    combos = combinations([0, 1, 2], 2)
    for i in range(img_train.shape[1]-1):
        feature = img_train.iloc[:, [i, -1]]
        for c in combos:
            stat, p = ks_2samp(feature[feature.label==c[0]].iloc[:, i], feature[feature.label==c[1]].iloc[:, i])
            if p < 0.05:
                print(f'There is a significantly different distribution between {labels[c[0]]} and {labels[c[1]]}, on the feature {img_train.iloc[:, i].name}')


# In[8]:


def plot_distributions(img_train):
    '''
    Distributions for each of the features, grouped by normal, pneumonia, and covid
    '''
    for i in range(img_train.shape[1]-1):
        for j in range(3):
            x = img_train[img_train.label==j]
            sns.displot(data = x, x = img_train.iloc[:, i].name)
            plt.title(f'{labels[j]}')


# In[9]:


def calc_ANOVA(df):
    '''
    Uses pairwise tests to test against the null hypothesis that there is NO significant difference between means of groups
    '''
    for i in range(df.shape[1]-1):
        stat, p = stats.f_oneway(df.iloc[:, i][df.iloc[:, -1] == 0], df.iloc[:, i][df.iloc[:, -1] == 1], df.iloc[:, i][df.iloc[:, -1] == 2])
        if p < 0.05/3:
            print(f'There is a statistically significant difference in means between classes on feature {df.iloc[:, i].name}, with p-value = {np.format_float_scientific(p)}')
        else:
            print(f'There is no statistically significant difference in means between classes on feature {df.iloc[:, i].name}, with p-value = {np.format_float_scientific(p)}')


# In[10]:


def calc_PCA(X, n, test_X):
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(X)
    scaled_test = scaler.fit_transform(test_X)
#     idx = X.shape[1]
    pca = PCA(n_components = n)
    pca_scores = pca.fit_transform(scaled_train)
    test_pca = pca.transform(scaled_test)
    pca_df = pd.DataFrame(data = pca_scores, columns = [f'PC{i+1}' for i in range(n)])
    return pca, pca_df, test_pca


# In[11]:


def plot_2d_PCA(pc_df, labels, hue):
    df = pd.concat([pc_df, labels], axis=1)
#     sns.set_palette(sns.color_palette('Paired'))
    sns.scatterplot(data = df, x = df.PC1, y = df.PC2, hue = hue)


# In[12]:


def compute_error(model, X, y, X_test, y_test):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    X = np.array(X)
    y = np.array(y)
    train_errors = 0
    valid_errors = 0
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        mod = model.fit(X_train, y_train)
        train_preds = mod.predict(X_train)
        valid_preds = mod.predict(X_valid)
        train_error = 1 - metrics.accuracy_score(y_train, train_preds, normalize=True)
        train_errors += train_error
        valid_preds = mod.predict(X_valid)
        valid_error = 1 - metrics.accuracy_score(y_valid, valid_preds, normalize=True)
        train_errors += train_error
        valid_errors += valid_error
    y_test_preds = mod.predict(X_test)
    f1 = f1_score(y_test, y_test_preds, average='weighted')
    test_error = 1 - metrics.accuracy_score(y_test, y_test_preds, normalize=True)
    avg_train_error = train_errors/5
    avg_valid_error = valid_errors/5
    return avg_train_error, avg_valid_error, test_error, f1, y_test_preds


# In[13]:


# plot_distributions(img_train)
calc_ANOVA(img_train)
y = []
x = [n for n in range(1, train_X.shape[1])]
for n in range(1, train_X.shape[1]):
    pca, pca_df1, test_X_PCA = calc_PCA(train_X, n, test_X)
    y.append(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(np.cumsum(pca.explained_variance_ratio_))
    
# Plots the first two PCA components    
pc2, pc_df, test_X_PCA2 = calc_PCA(train_X, 2, test_X)
y_labeled = train_y.replace({0: 'normal', 1: 'pneumonia', 2: 'covid'})
plot_2d_PCA(pc_df, pd.DataFrame(y_labeled), 'label')

# Appends the four selected PCA components to the original X dataframe
# pc4, pc_df4 = calc_PCA(scaled_data, 4)
# train_X = pd.concat([train_X, pc_df4], axis=1)
train_X


# In[14]:


# Plots num of PCA components against variance explained
plt.plot(x, y)
plt.xlabel('PCA Components')
plt.ylabel('Variance Explained')
plt.show()


# In[15]:


'''
Covariance and Correlation
'''
cov_matrix = np.cov(train_X.T)
correlation_matrix = train_X.corr()

for i in range(correlation_matrix.values.shape[0]):
    for j in range(correlation_matrix.values.shape[1]):
        if i != j:
            if correlation_matrix.values[i, j] > 0.9:
                print(f'There is a significant correlation ({correlation_matrix.values[i, j]}) between {train_X.columns[j]} and {train_X.columns[i]}')
                


# In[16]:


scaled_all_data.shape


# In[17]:


'''
KMeans
'''
# Fits a KMeans model with three cluster groups and appends the predicted labels to the original labels frame (for comparison)
features_to_use = ['Kurtosis', 'MeanAbsoluteDeviation', 'Mean', 'Range', 'RootMeanSquared', 'Skewness', 'TotalEnergy', 'Variance']
X_to_fit = [idx for idx in range(len(scaled_all_data.columns)) if scaled_all_data.columns[idx] in features_to_use] 
# for i in range(len(X_to_fit)):
#     X_to_fit[i] += 1
X_to_fit = [4, 6, 7, 10, 12, 13, 14, 15]
kmeans = KMeans(n_clusters=3).fit(scaled_all_data.iloc[:, X_to_fit])
df_kmeans = pd.concat([pd.DataFrame(kmeans.labels_, columns=['KMeans']), pd.DataFrame(scaled_all_data)], axis=1)
num_errors = df_kmeans[df_kmeans.label != df_kmeans.KMeans].shape[0]
accuracy = num_errors/df_kmeans.shape[0]
print(f"KMeans model has {accuracy} accuracy")


plot_2d_PCA(pc_df, pd.DataFrame(df_kmeans.KMeans), 'KMeans')

# AUC ROC
# kmeans_dummy = pd.get_dummies(pd.DataFrame(df_kmeans.iloc[:, 0]), columns=[0])
# kmeans_true = df_kmeans.iloc[:, 1]
k_labels = df_kmeans[['KMeans', 'label']]
k_accuracy = metrics.accuracy_score(k_labels.label, k_labels.KMeans, normalize=True)
f1_k = f1_score(k_labels.label, k_labels.KMeans, average='weighted')
print(f'The F1 for kmeans is {f1_k}')
print(f'The accuracy for kmeans is {k_accuracy}')


# In[18]:


zeros = df_kmeans[df_kmeans.label == 0]
ones = df_kmeans[df_kmeans.label == 1]
twos = df_kmeans[df_kmeans.label == 2]

k_tp_norm = zeros[zeros.KMeans==0].shape[0]
k_fn_norm = df_kmeans[(df_kmeans.label==0) & (df_kmeans.KMeans != 0)].shape[0]
sens_k_norm = k_tp_norm/(k_tp_norm+k_fn_norm)

k_tn_norm = df_kmeans[(df_kmeans.label != 0) & (df_kmeans.KMeans != 0)].shape[0]
k_fp_norm = df_kmeans[(df_kmeans.label != 0) & (df_kmeans.KMeans == 0)].shape[0]
spec_k_norm = k_tn_norm/(k_tn_norm + k_fp_norm)
print(f'For NORMAL the sensitivity is {sens_k_norm}, specificity is {spec_k_norm}')


# In[19]:


k_tp_pneu = ones[ones.KMeans==1].shape[0]
k_fn_pneu = df_kmeans[(df_kmeans.label==1) & (df_kmeans.KMeans != 1)].shape[0]
sens_k_pneu = k_tp_pneu/(k_tp_pneu+k_fn_pneu)

k_tn_pneu = df_kmeans[(df_kmeans.label != 1) & (df_kmeans.KMeans != 1)].shape[0]
k_fp_pneu = df_kmeans[(df_kmeans.label != 1) & (df_kmeans.KMeans == 1)].shape[0]
spec_k_pneu = k_tn_pneu/(k_tn_pneu + k_fp_pneu)
spec_k_pneu
print(f'For PNEUMONIA the sensitivity is {sens_k_pneu}, specificity is {spec_k_pneu}')


# In[20]:


k_tp_covid = twos[twos.KMeans==2].shape[0]
k_fn_covid = df_kmeans[(df_kmeans.label==2) & (df_kmeans.KMeans != 2)].shape[0]
sens_k_covid = k_tp_covid/(k_tp_covid+k_fn_covid)
sens_k_covid

k_tn_covid = df_kmeans[(df_kmeans.label != 2) & (df_kmeans.KMeans != 2)].shape[0]
k_fp_covid = df_kmeans[(df_kmeans.label != 2) & (df_kmeans.KMeans == 2)].shape[0]
spec_k_covid = k_tn_covid/(k_tn_covid + k_fp_covid)
print(f'For COVID the sensitivity is {sens_k_covid}, specificity is {spec_k_covid}')
spec_k_covid


# In[ ]:


'''
SVM
'''
# Fit an SVM using all of the features
# X = train_X
# y = train_y
# m1 = SVC(C=1, kernel='linear', decision_function_shape='ovo').fit(X, y)
# m1_train_preds = m1.predict(X)
# m1_test_preds = m1.predict(test_X)
# m1_decision = m1.decision_function(test_X)
# m1_score = m1.score(test_X, test_y)
# m1_f1_train = f1_score(train_y, m1_train_preds, average='weighted')
# m1_f1_test = f1_score(test_y, m1_test_preds, average='weighted') 

# _, X2_train, X2_test = calc_PCA(train_X, 5, test_X)
# X2_train = X2_train
# m2 = SVC(C=1, kernel='linear', decision_function_shape='ovo').fit(X2_train, y)
# m2_train_preds = m2.predict(X2_train)
# m2_test_preds = m2.predict(X2_test)
# m2_decision = m2.decision_function(X2_test)
# m2_score = m2.score(X2_test, test_y)
# m2_f1_train = f1_score(train_y, m2_train_preds, average='weighted')
# m2_f1_test = f1_score(test_y, m2_test_preds, average='weighted')
# svm_labels = pd.concat([test_y, pd.DataFrame(m2_test_preds)], axis=1)
# m2_accuracy = metrics.accuracy_score(svm_labels.label, svm_labels[0], normalize=True)
# print(f'The F1 for SVM is {m2_f1_test}')
# print(f'The accuracy for SVM is {m2_accuracy}')

# y_train = label_binarize(train_y, classes=[0, 1, 2])
# y_test = label_binarize(test_y, classes=[0, 1, 2])
# n_classes = 3

# # # Learn to predict each class against the other
svm_clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))

y_score = svm_clf.fit(train_X, y_train).decision_function(X_test)

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


# In[ ]:





# In[ ]:





# In[22]:


# svm_tp_norm = svm_labels[(svm_labels.label==0) & (svm_labels[0]==0)].shape[0]
# svm_fn_norm = svm_labels[(svm_labels.label==0) & (svm_labels[0] != 0)].shape[0]
# sens_svm_norm = svm_tp_norm/(svm_tp_norm + svm_fn_norm)

# svm_tn_norm = svm_labels[(svm_labels.label != 0) & (svm_labels[0] != 0)].shape[0]
# svm_fp_norm = svm_labels[(svm_labels.label != 0) & (svm_labels[0] == 0)].shape[0]
# spec_svm_norm = svm_tn_norm/(svm_tn_norm + svm_fp_norm)

# print(f'For NORMAL the sensitivity is {sens_svm_norm}, specificity is {spec_svm_norm}')


# In[23]:


# svm_tp_pneu = svm_labels[(svm_labels.label==1) & (svm_labels[0]==1)].shape[0]
# svm_fn_pneu = svm_labels[(svm_labels.label==1) & (svm_labels[0] != 1)].shape[0]
# sens_svm_pneu = svm_tp_pneu/(svm_tp_pneu + svm_fn_pneu)

# svm_tn_pneu = svm_labels[(svm_labels.label != 1) & (svm_labels[0] != 1)].shape[0]
# svm_fp_pneu = svm_labels[(svm_labels.label != 1) & (svm_labels[0] == 1)].shape[0]
# spec_svm_pneu = svm_tn_pneu/(svm_tn_pneu + svm_fp_pneu)

# print(f'For PNEUMONIA the sensitivity is {sens_svm_pneu}, specificity is {spec_svm_pneu}')


# In[24]:


# svm_tp_covid = svm_labels[(svm_labels.label==2) & (svm_labels[0]==2)].shape[0]
# svm_fn_covid = svm_labels[(svm_labels.label==2) & (svm_labels[0] != 2)].shape[0]
# sens_svm_covid = svm_tp_covid/(svm_tp_covid + svm_fn_covid)

# svm_tn_covid = svm_labels[(svm_labels.label != 2) & (svm_labels[0] != 2)].shape[0]
# svm_fp_covid = svm_labels[(svm_labels.label != 2) & (svm_labels[0] == 2)].shape[0]
# spec_svm_covid = svm_tn_covid/(svm_tn_covid + svm_fp_covid)

# print(f'For NORMAL the sensitivity is {sens_svm_covid}, specificity is {spec_svm_covid}')


# In[27]:


'''
Decision Tree Classifier
'''
# depth = [d for d in range(1,16)]
# m3_tr_error = []
# m3_val_error = []
# m3_te_error = []
# m3_f1_scores = []
# for i in range(1, 16):
#     m3 = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
#     m3_train_error, m3_valid_error, m3_test_error, m3_f1, m3_preds = compute_error(m3, img_train.iloc[:, :-1], img_train.iloc[:, -1], test_X, test_y)
#     #     tree_train_preds = pd.DataFrame(m3.predict(train_X.iloc[:, -4:]), columns=['decision_tree'])
#     m3_tr_error.append(m3_train_error)
#     m3_val_error.append(m3_valid_error)
#     m3_te_error.append(m3_test_error)
#     m3_f1_scores.append(m3_f1)
#     if i == 6:
#         m3_tree6 = m3_preds
#         m3_f1_final = m3_f1
#         m3_accuracy = metrics.accuracy_score(test_y, m3_tree6, normalize=True)
    
# # Plot train and valid error against max tree depth
# plt.figure()
# plt.plot(depth, m3_tr_error, label='Training Error')
# plt.plot(depth, m3_val_error, label='Validation Error')
# plt.plot(depth, m3_te_error, label='Test Error')
# plt.legend()
# plt.xlabel('Tree Depth')
# plt.ylabel('Average Error')
# plt.title('Decision Tree Classifier (with original features)')
# plt.show()

# plt.plot(depth, m3_f1_scores)
# plt.xlabel('Tree Depth')
# plt.ylabel('Accuracy (F1 Score)')
# plt.title('Decision Tree Classifier (with original features)')
# plt.show()

# # _, test_X_PCA = calc_PCA(test_X, 4)
# # test_X = pd.concat([test_X, test_X_PCA], axis=1)
# test_y = pd.DataFrame(test_y)
# m4_f1_scores = []
# for comp in range(4, 9):
#     m4_tr_error = []
#     m4_val_error = []
#     m4_te_error = []
#     pca_f1_scores = []
#     _, train_X_PCA, test_X_PCA = calc_PCA(train_X, comp, test_X)
# #     test_X_PCA = pca_gen.transform(test_X)
#     for i in range(1, 16):
#         m4 = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
#         m4_train_error, m4_valid_error, m4_test_error, m4_f1, m4_preds = compute_error(m4, train_X_PCA, train_y, test_X_PCA, test_y)
#         m4_tr_error.append(m4_train_error)
#         m4_val_error.append(m4_valid_error)
#         m4_te_error.append(m4_test_error)
#         pca_f1_scores.append(m4_f1)
#     m4_f1_scores.append([pca_f1_scores])
#     # Plot train and valid error against max tree depth
#     plt.figure()
#     plt.plot(depth, m4_tr_error, label='Training Error')
#     plt.plot(depth, m4_val_error, label='Validation Error')
#     plt.plot(depth, m4_te_error, label='Test Error')
#     plt.legend()
#     plt.xlabel('Tree Depth')
#     plt.ylabel('Average Error')
#     plt.title(f'Decision Tree Classifier (with {comp} PCA components)')
#     plt.show()
#     if comp == 5:
#         plt.figure()
#         plt.plot(depth, pca_f1_scores)
#         plt.xlabel('Tree Depth')
#         plt.ylabel('Accuracy (F1 Score)')
#         plt.title('Decision Tree Classifier (with 5 PCA components)')
#         plt.show()
    
# plt.figure()
# for i in range(len(m4_f1_scores)):
#     plt.plot(depth, m4_f1_scores[i][0], label=f'n={i+4}')
# plt.legend()
# plt.xlabel('Tree Depth')
# plt.ylabel('Accuracy (F1 Score)')
# plt.title(f'Decision Tree Classifier (with n PCA components)')
# plt.show()

# print(f'The F1 for Decision Tree is {m3_f1_final}')
# print(f'The accuracy for kmeans is {m3_accuracy}')

y_train = label_binarize(train_y, classes=[0, 1, 2])
y_test = label_binarize(test_y, classes=[0, 1, 2])
n_classes = 3

# # # Learn to predict each class against the other
dt_clf = OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=6))

y_score = dt_clf.fit(train_X, y_train).predict_proba(test_X)

# # Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Decision Tree')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


tree_labels = pd.concat([test_y, pd.DataFrame(m3_tree6)], axis=1)
tree_tp_norm = tree_labels[(tree_labels.label==0) & (tree_labels[0]==0)].shape[0]
tree_fn_norm = tree_labels[(tree_labels.label==0) & (tree_labels[0] != 0)].shape[0]
sens_tree_norm = tree_tp_norm/(tree_tp_norm + tree_fn_norm)

tree_tn_norm = tree_labels[(tree_labels.label != 0) & (tree_labels[0] != 0)].shape[0]
tree_fp_norm = tree_labels[(tree_labels.label != 0) & (tree_labels[0] == 0)].shape[0]
spec_tree_norm = tree_tn_norm/(tree_tn_norm + tree_fp_norm)

print(f'For NORMAL the sensitivity is {sens_tree_norm}, specificity is {spec_tree_norm}')


# In[ ]:


tree_tp_pneu = tree_labels[(tree_labels.label==1) & (tree_labels[0]==1)].shape[0]
tree_fn_pneu = tree_labels[(tree_labels.label==1) & (tree_labels[0] != 1)].shape[0]
sens_tree_pneu = tree_tp_pneu/(tree_tp_pneu + tree_fn_pneu)

tree_tn_pneu = tree_labels[(tree_labels.label != 1) & (tree_labels[0] != 1)].shape[0]
tree_fp_pneu = tree_labels[(tree_labels.label != 1) & (tree_labels[0] == 1)].shape[0]
spec_tree_pneu = tree_tn_pneu/(tree_tn_pneu + tree_fp_pneu)

print(f'For NORMAL the sensitivity is {sens_tree_pneu}, specificity is {spec_tree_pneu}')


# In[ ]:


tree_tp_covid = tree_labels[(tree_labels.label==2) & (tree_labels[0]==2)].shape[0]
tree_fn_covid = tree_labels[(tree_labels.label==2) & (tree_labels[0] != 2)].shape[0]
sens_tree_covid = tree_tp_covid/(tree_tp_covid + tree_fn_covid)

tree_tn_covid = tree_labels[(tree_labels.label != 2) & (tree_labels[0] != 2)].shape[0]
tree_fp_covid = tree_labels[(tree_labels.label != 2) & (tree_labels[0] == 2)].shape[0]
spec_tree_covid = tree_tn_covid/(tree_tn_covid + tree_fp_covid)

print(f'For NORMAL the sensitivity is {sens_tree_covid}, specificity is {spec_tree_covid}')


# In[ ]:


sns.heatmap(correlation_matrix)
correlation_matrix


# In[ ]:


# decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)


# kf = KFold(n_splits=5)
# kf.get_n_splits(train_X)
# X = np.array(train_X)
# y = np.array(train_y)
# for train_index, valid_index in kf.split(X):
#     X_train, X_valid = X[train_index], X[valid_index]
#     y_train, y_valid = y[train_index], y[valid_index]
#     decision_tree = decision_tree.fit(X_train, y_train)
# plt.figure()
# plot_roc_curve(decision_tree, test_X, test_y)
# plt.show()

# # plt.figure()
# # plot_roc_curve()
# # plt.show()


# In[ ]:


test = label_binarize(test_y, classes=[0, 1, 2])
test

