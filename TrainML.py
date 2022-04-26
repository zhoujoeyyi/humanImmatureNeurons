cd D:\Users\zeiss\Documents\filesForYr0

#%% Import packages ----
from __future__ import division
import pandas as pd
import numpy as np
from backspinpy import SPIN, backSPIN, fit_CV, feature_selection, CEF_obj, Cef_tools, cef2df
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from matplotlib import pyplot as mp
import pickle
import scipy
from scipy.io import mmread
import seaborn as sns

#%% Define functions ----
#Filter for Development genes in filesForYr0 dataset
#thrs is defined later.
#Plotting function
def plot_cvmean():
    mp.figure()
    mp.scatter(np.log2(mu),np.log2(cv), marker='o', edgecolor ='none',alpha=0.1, s=5)
    mu_sorted = mu[np.argsort(score)[::-1]]
    cv_sorted = cv[np.argsort(score)[::-1]]
    mp.scatter(np.log2(mu_sorted[:thrs]),np.log2(cv_sorted[:thrs]), marker='o', edgecolor ='none',alpha=0.15, s=8, c='r')
    mp.plot(mu_linspace, cv_fit,'-k', linewidth=1, label='$Fit$')
    mp.plot(np.linspace(-9,7), -0.5*np.linspace(-9,7), '-r', label='$Poisson$')
    mp.ylabel('log2 CV')
    mp.xlabel('log2 mean')
    mp.grid(alpha=0.3)
    mp.xlim(-8.6,6.5)
    mp.ylim(-2,6.5)
    mp.legend(loc=1, fontsize='small')
    mp.gca().set_aspect(1.2)


# Wheel/Polygonal plot
def polygonalPlot(data, scaling=True,start_angle=90, rotate_labels=True, labels=('one','two','three'),\
                sides=3, label_offset=0.10, edge_args={'color':'black','linewidth':2},\
                fig_args = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},):

    basis = np.array([[np.cos(2*i*np.pi/sides + start_angle*np.pi/180),
                    np.sin(2*i*np.pi/sides + start_angle*np.pi/180)] for i in range(sides)])
    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data
        newdata = np.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data,basis)
    fig = mp.figure(**fig_args)
    ax = fig.add_subplot(111)
    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/np.pi + 90
            if angle > 90 and angle <= 270:
                angle = np.mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + label_offset),
                y*(1 + label_offset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )
    # Clear normal matplotlib axes graphics
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)
    # Plot borders
    ax.plot([basis[_,0] for _ in list(range(sides)) + [0,]],
            [basis[_,1] for _ in list(range(sides)) + [0,]],
            **edge_args)
    return newdata,ax



#%% Load filesForYr0 data ----
df_filesForYr0 = scipy.io.mmread("filesForYr0.mtx")
#df = pd.DataFrame(df.toarray()) <- works too
df_filesForYr0 = pd.DataFrame(df_filesForYr0.todense())

filesForYr0row = pd.read_csv("filesForYr0row.txt", sep='\t')
df_filesForYr0.index = list(filesForYr0row.iloc[:,0])
filesForYr0col = pd.read_csv("filesForYr0col.txt", sep='\t')
df_filesForYr0.columns = list(filesForYr0col.iloc[:,0])
df_filesForYr0.iloc[0:3, 0:3]

#Load rows/cols_annotation
#Need to add "hi" at the beginning and end of the "xxxrowmxt.txt"
filesForYr0rowmxt = pd.read_csv("filesForYr0rowmxt.txt", sep='"\t"')

#The line above leads to a warning, but still works. Copy/paste till here to avoid error
filesForYr0rowmxt.index = ['Gene']
filesForYr0rowmxt = filesForYr0rowmxt.drop('hi"', 1)
filesForYr0rowmxt = filesForYr0rowmxt.drop('"hi', 1)
filesForYr0rows_annot = filesForYr0rowmxt.copy()

filesForYr0colannott = pd.read_csv("filesForYr0colannott.csv")
filesForYr0colannott.index = ['Cell_ID', 'Cell_type']
filesForYr0colannott = filesForYr0colannott.drop('Unnamed: 0', 1)
filesForYr0colannott.iloc[0:2, 0:5]

filesForYr0colannott1 = filesForYr0colannott
filesForYr0colannott1 = filesForYr0colannott1.T
filesForYr0colannott1.head()

dcxcl_filesForYr0colannott = pd.read_csv("D:/Users/zeiss/Documents/filesForYr0_cleanedUp1.csv")

dcxcl_filesForYr0colannott.index = ['Cell_ID', 'Cell_type']
dcxcl_filesForYr0colannott = dcxcl_filesForYr0colannott.drop('Unnamed: 0', 1)
dcxcl_filesForYr0colannott.iloc[0:2, 0:5]
dcxcl_filesForYr0colannott1 = dcxcl_filesForYr0colannott
dcxcl_filesForYr0colannott1 = dcxcl_filesForYr0colannott1.T
dcxcl_filesForYr0colannott1.head()

filesForYr0cols_annot1 = pd.merge(filesForYr0colannott1, dcxcl_filesForYr0colannott1, how = 'outer', on= 'Cell_ID', suffixes= ('_m', '_n'), indicator= True)
filesForYr0cols_annot1.iloc[0:5,:]

filesForYr0cols_annot1['Cell_type'] = filesForYr0cols_annot1['Cell_type_m'].where(filesForYr0cols_annot1['Cell_type_n'].isnull(), filesForYr0cols_annot1['Cell_type_n'])
filesForYr0cols_annot1[['Cell_ID', 'Cell_type', 'Cell_type_m', 'Cell_type_n', '_merge']]
filesForYr0cols_annot1.drop(['Cell_type_m', 'Cell_type_n', '_merge'], axis= 1, inplace= True)
filesForYr0cols_annot1.head()

filesForYr0cols_annot = filesForYr0cols_annot1.T
filesForYr0cols_annot.columns = filesForYr0cols_annot.iloc[0,:]
filesForYr0cols_annot.iloc[0:2, 0:5]
#del filesForYr0cols_annot1.index.name
#filesForYr0cols_annot1.rename_axis(None)
#filesForYr0cols_annot1.index = ['Cell_ID', 'Cell_type']
#filesForYr0cols_annot1.index
filesForYr0cols_annotdup0 = filesForYr0cols_annot.copy() #added



#Get rid of blood, MT genes, etc. ----
blood_genes = ['HBG1','HBA1','HBA2','HBE1']
mito_genes = open('D:/Users/zeiss/Documents/MitoGenes.txt').read().split('"\t"')
del mito_genes[0]
del mito_genes[-1]

df_filesForYr0 = df_filesForYr0.loc[~np.in1d(df_filesForYr0.index, blood_genes)&\
                   ~np.in1d(df_filesForYr0.index, mito_genes),:]

df_filesForYr0dup0 = df_filesForYr0.copy()

#%% Annotations of filesForYr0 ----
#Describe the prototypic cell types
proto = pd.Series({'neuron':'none', 'astro':'none', 'gn':'none', 'moli':'none', 'opc':'none', 'microglia':'none', 'other':'none','x':'none',
                   'sn':'newsn', 'sd':'newsd', 'smo':'newsmo', 'sopc':'newsopc', 'sa':'newsa'})

protocol = {'none': (190,10,10), 'neuron':(30,133,180), 'astro':(240,255,250), 'moli':(240,255,250),'gn':(240,255,250),'microglia':(50,80,80), 'other':(50,80,80),'opc':(250,80,80),
            'sn':(50,254,180), 'sa':(220,180,180),'sd':(220,180,180),'sopc':(30,133,180), 'smo':(50,254,180),'smg':(30,133,180),'si':(50,80,80), '5':(220,180,180),
            'snd':(50,254,180), 'sad':(220,180,180),'sdd':(150,200,50),'sopcd':(30,133,180), 'smod':(50,0,180),'smgd':(50,254,180),
            'newsn': (190,10,10), 'newsa': (190,10,10), 'newsd':(255,95,105), 'endo':(255,95,105), 'newsopc':(50,180,180),'newsmo':(50,180,180), 'newsmg':(50,180,180)}

cols_annot_all = filesForYr0cols_annot
df_dev = df_filesForYr0.copy()

ct_dev = cols_annot_all[df_dev.columns].loc['Cell_type']
protogruop = proto.loc[ct_dev].values
df_dev = df_dev.loc[:,protogruop != 'none']

#%% Pre-filtering and select genes for filesForYr0 ----

df_f = df_dev.copy()
df_f.shape

df_f = df_f.loc[:,(df_f>0).sum(0)>0] #>= x genes is expressed in this cell
#df_f = df_f.loc[:,(df_f>0).sum(0)<=5000] #<= x genes is expressed in this cell

df_f = df_f.loc[(df_f>0).sum(1)>0,:] #>= x cells express this gene

df_dev = df_f.iloc[np.argsort(score)[::-1],:]
del df_f


#%%Enrichment: Select for cell type positive markers (optional) ----
df_fall = df_dev
cell_types = cols_annot_all[df_fall.columns].loc['Cell_type'].values
df_means = df_fall.mean(1) + 1e-5
df_bin = df_fall>0
df_fold = pd.DataFrame()
df_avgpos = pd.DataFrame()


# for some single cell types
enrichment_order = ['sn', 'sa', 'sd', 'smo', 'sopc']

for ct in enrichment_order:
    df_fold[ct] = df_fall.loc[:,cell_types == ct].mean(1) / df_means
    df_avgpos[ct] = df_bin.loc[:,cell_types == ct].mean(1)


score00 = df_fold
score05 = df_fold * df_avgpos**0.5
score10 = df_fold * df_avgpos

ix00 = np.argsort( score00 , 0)
ix05 = np.argsort( score05 , 0)
ix10 = np.argsort( score10 , 0)

markers = defaultdict(set)

N = 100

for ct in df_fold.columns:
    markers[ct] |= set( df_fold.index[ix00.loc[:,ct][::-1]][:N] )
    markers[ct] |= set( df_fold.index[ix05.loc[:,ct][::-1]][:N] )
    markers[ct] |= set( df_fold.index[ix10.loc[:,ct][::-1]][:N] )


for ct in df_fold.columns:
    for mk in markers[ct]:
        for ct2 in list( set(df_fold.columns) - set([ct])):
            if score10.loc[mk,ct] >= score10.loc[mk,ct2]:
                markers[ct2] -= set([mk])
    for mk in list(markers[ct]):
        if df_avgpos.loc[mk,ct] < 0.15:
            markers[ct] -= set([mk])


#N = 145 or 1000 or others;  if df_avgpos.ix[mk,ct] < 0.15:

list_genes = ([list(markers[ct]) for ct in df_fold.columns])
#Out: 23
list_genes_out_of_nest = list()
for x in list_genes:
    for a in x:
        list_genes_out_of_nest.append(a)


len(list_genes_out_of_nest)
df_dev = df_dev.loc[list(set(list_genes_out_of_nest)),:]


#%% Train model for filesForYr0 and dup0 ----
#Prepare the reference dataset
# Log normalization
df_dev_log = np.log2(df_dev + 1)

# the same as seurat NormalizeData
#df_f = df_dev.copy()
#df_dev_log = np.log1p(df_f/df_f.sum(0)*10000)
#df_dev_log = df_dev

# Check if this is repeated
ct_dev = cols_annot_all[df_dev.columns].loc['Cell_type']
protogruop = proto.loc[ct_dev].values
bool1 = protogruop != 'none'
classes_names, classes_index = np.unique(protogruop[bool1], return_inverse=True, return_counts=False)
train_index = classes_index
df_train_set = df_dev_log.loc[:,bool1].copy()
df_train_set.shape

#Regularization Path
normalizer = 0.9*df_train_set.values.max(1)[:,np.newaxis]

LR = LogisticRegressionCV(Cs=np.logspace(-4.25,1,30), refit=True, penalty='l2',
                          solver='newton-cg', fit_intercept=False, multi_class='multinomial',class_weight='balanced',
                          cv=StratifiedShuffleSplit(n_splits=35,test_size=0.15, random_state=123)) # n_jobs = -1,

LR.fit((df_train_set.values/normalizer).T, train_index) # <-- Taking a long time and computing power, execute till here



mp.figure(figsize=(6,3))
chos = .75

mp.subplot(122)
CMA = np.array([abs(LR.coefs_paths_[i]) for i in range(len(set(classes_index)))])
val_ = np.array( [[sum( sum( CMA[:,i,j,:],0) ,0) for i in range(35)] for j in range(30) ] )
mp.plot( np.log10(LR.Cs_),\
     val_.mean(1), c='k' , lw=2 )
mp.xlabel('-log( Regularization strength )')
mp.ylabel('# Sum abs(coeffs)')
mp.xlim(-4.25,1) #Was: (-3.25,0.8)
mp.axvline( np.log10(chos) )

mp.subplot(121)
mp.plot(np.log10(LR.Cs_), np.mean([LR.scores_[i].mean(0) for i in range(len(set(classes_index)))],0), c='k', lw=2 )
mp.plot(np.log10(LR.Cs_), np.percentile( LR.scores_[1],97.5,0), c='r')
mp.plot(np.log10(LR.Cs_), np.percentile( LR.scores_[1],2.5,0), c='r')
mp.axvline( np.log10(chos) )
mp.ylabel('Accuracy Score')
mp.xlabel('-log( Regularization strength )')
mp.xlim(-4.25,1) #Was: (-3.25,0.8)
mp.ylim(0.85,1) #Was: (0.70,0.95)

mp.tight_layout()
mp.savefig('RegPathfilesForYr0.pdf')



#%% Final model for filesForYr0 picked cells only ----
# Normalized by the max
LR = LogisticRegression(C=chos, penalty='l2', solver='newton-cg', fit_intercept=False, 
                        multi_class='multinomial',class_weight='balanced',random_state=123)#, n_jobs = -1)

normalizer = 0.9*df_train_set.values.max(1)[:,np.newaxis]
LR.fit((df_train_set.values / normalizer).T, train_index)



# Score with the model for filesForYr0 picked cells only ---- 
hist_order = ['newsn', 'newsa', 'newsd', 'newsmo', 'newsopc']
hist_ixes = [list(classes_names).index(i) for i in hist_order]

mp.figure(figsize=(10,4))
pobs_list = []
for z,i in enumerate(hist_ixes):
    mp.subplot(len(set(classes_index))/1,1,z+1)
    prob = LR.predict_proba((df_train_set.values[:,classes_index==i]/ normalizer).T)[:,i]
    amounts,_,_ = mp.hist(prob, color=np.array(protocol[classes_names[i]])/255.,bins=np.linspace(0,1,25) )
    pobs_list.append([ classes_names[i], np.mean(prob)])
    
    if z == len(hist_ixes)-1:
        mp.tick_params('both',which='both', right='on',left='off',top='off',
                    labelleft='off',labelbottom='on',labelright='on')
    mp.axvline(0.5)
    mp.xlim(0.,1)
    mp.ylabel(classes_names[i],rotation='horizontal',horizontalalignment = 'right')
    mp.yticks(np.linspace(0,max(amounts), 4 )[1:],['','','%d' % ( max(amounts) )])
    mp.ylim(0,1.2*max(amounts))
    mp.tick_params('both',which='both', right='on',left='off',top='off',
                    labelleft='off',labelbottom='off',labelright='on')


mp.tight_layout(pad=1,h_pad=0,w_pad=1)
mp.savefig('ScoreModelfilesForYr0.pdf')
#Lots warnings, but worked.



#%% Wheel/Polygonal plot for filesForYr0 picked cells only ----
#Plot wheel plot
wanted_order = ['newsn', 'newsa', 'newsd', 'newsmo', 'newsopc']

reorder_ix = [list(classes_names).index(i) for i in wanted_order]
bool00 = np.in1d( classes_names[classes_index],  wanted_order )
color_dict = pd.Series({'sn':(20,20,20),'astro': (200,200,200),
                        'sa':(54,27,0),'microglia':(255,200,150),
                        'sd':(255,0,0),'gn':(200,255,0),
                        'sopc':(76,0,150),'opc': (227,159,246),
                        'smo':(0,0,200),'moli':(137,207,240),
                        'sglut':(0,0,0),'sbv':(0,0,0),'seuratglut':(137,7,240),'endo':(137,7,240),
                        'sgaba':(0,0,0),'seuratgaba':(137,107,240),
                        'neuron':(255, 153, 153)})
color_dict = color_dict.map(lambda x: list(map(lambda y: y/255., x)))

mp.rcParams['savefig.dpi'] = 90
newcolors = np.array(list(color_dict[cols_annot_all.loc[:,df_train_set.columns].loc['Cell_type']].values))

newdata,ax = polygonalPlot(LR.predict_proba((df_train_set.values/ df_train_set.values.max(1)[:,np.newaxis]).T)[:,reorder_ix],\
                         scaling=False, sides=len(reorder_ix), labels=classes_names[reorder_ix])

ax.scatter( newdata[bool00,0]*0.99, newdata[bool00,1]*0.99, alpha=0.8,\
           c=newcolors[bool00,],\
           s = 5, lw=0.2)

mp.savefig('PlotwheelplotfilesForYr0.pdf')



#%% Viualize the top positive coefficients (heatmap) for filesForYr0 picked cells only ----
sel_class = 'newsd' #edit
nonzero_coef_bool = LR.coef_[np.where( classes_names == sel_class )[0][0], :] > 0
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ][nonzero_coef_bool]
names_nonzero_coef= df_dev_log.index[nonzero_coef_bool]
ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]
names_sorted_coefsd = names_sorted_coef #edit


sel_class = 'newsn' #edit
nonzero_coef_bool = LR.coef_[np.where( classes_names == sel_class )[0][0], :] > 0
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ][nonzero_coef_bool]
names_nonzero_coef= df_dev_log.index[nonzero_coef_bool]
ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]
names_sorted_coefsn = names_sorted_coef #edit


sel_class = 'newsopc' #edit
nonzero_coef_bool = LR.coef_[np.where( classes_names == sel_class )[0][0], :] > 0
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ][nonzero_coef_bool]
names_nonzero_coef= df_dev_log.index[nonzero_coef_bool]
ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]
names_sorted_coefsopc = names_sorted_coef #edit


sel_class = 'newsmo' #edit
nonzero_coef_bool = LR.coef_[np.where( classes_names == sel_class )[0][0], :] > 0
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ][nonzero_coef_bool]
names_nonzero_coef= df_dev_log.index[nonzero_coef_bool]
ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]
names_sorted_coefsmo = names_sorted_coef #edit


sel_class = 'newsa' #edit
nonzero_coef_bool = LR.coef_[np.where( classes_names == sel_class )[0][0], :] > 0
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ][nonzero_coef_bool]
names_nonzero_coef= df_dev_log.index[nonzero_coef_bool]
ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]
names_sorted_coefsa = names_sorted_coef #edit

#Get top cells ----
res = LR.predict_proba((df_train_set.values / normalizer).T)
hist_ixes = [list(classes_names).index(i) for i in hist_order]
resdf = pd.DataFrame(res)
resdf.columns = classes_names
resdf.index = df_train_set.columns

top20cellssn = resdf.sort_values('newsn', ascending=False)[:10].index
top20cellssa = resdf.sort_values('newsa', ascending=False)[:10].index
top20cellssd = resdf.sort_values('newsd', ascending=False)[:10].index
top20cellssopc = resdf.sort_values('newsopc', ascending=False)[:10].index
top20cellssmo = resdf.sort_values('newsmo', ascending=False)[:10].index

top30genessn = names_sorted_coefsn[:10]
top30genessa = names_sorted_coefsa[:10]
top30genessd = names_sorted_coefsd[:10]
top30genessopc = names_sorted_coefsopc[:10]
top30genessmo = names_sorted_coefsmo[:10]

'''
topcellsheatmap = df_train_set.loc[top30genessh.tolist() + top30genessf.tolist() +top30genesGN.tolist() + top30genesRGL.tolist() +
                                   ['NES', 'VIM', 'FABP7', 'SLC1A3', 'MFGE8', 'SOX2', 'PAX6', 'ALDOC', 'S100B', 'HOPX', 'SOX9', 'HES6', 'NEUROG2', 'PTPRZ1', 'LIFR', 'OTX2', 'GFAP', 'GJA1', 'IL33', 'AQP4', 'PDGFD', 
                                    'MKI67', 'PCNA', 'EOMES', 'DCX', 'STMN1', 'NEUROD1', 'NFIA', 'ZBTB20', 'TUBB3', 'NCAM1', 'PROX1', 'RBFOX3', 'CALB1', 'CALB2', 'DPF1', 'NEUROD6', 'ELAVL2', 'SATB2', 'STMN2', 'CA2'],
                                   top20cellssh.tolist() + top20cellssf.tolist() + top20cellsGN.tolist() + top20cellsRGL.tolist()]

topcellsheatmap = df_train_set.loc[top30genessn.tolist() + top30genessd.tolist() +
                                   ['NES', 'VIM', 'FABP7', 'SLC1A3', 'MFGE8', 'SOX2', 'PAX6', 'ALDOC', 'S100B', 'HOPX', 'SOX9', 'HES6', 'NEUROG2', 'PTPRZ1', 'LIFR', 'OTX2', 'GFAP', 'GJA1', 'IL33', 'AQP4', 'PDGFD', 
                                    'MKI67', 'PCNA', 'EOMES', 'DCX', 'STMN1', 'NEUROD1', 'NFIA', 'ZBTB20', 'TUBB3', 'NCAM1', 'PROX1', 'RBFOX3', 'CALB1', 'CALB2', 'DPF1', 'NEUROD6', 'ELAVL2', 'SATB2', 'STMN2', 'CA2'],
                                   top20cellssn.tolist() + top20cellssd.tolist()]

topcellsheatmap = df_train_set.loc[top30genessb.tolist() + top30genes6IN.tolist() + top30geness0.tolist() + top30genesGN.tolist() +
                                   ['MKI67', 'PCNA', 'EOMES', 'DCX', 'STMN1', 'NEUROD1', 'NFIA', 'ZBTB20', 'TUBB3', 'NCAM1', 'PROX1', 'RBFOX3', 'CALB1',
                                    'CALB2', 'DPF1', 'NEUROD6', 'ELAVL2', 'SATB2', 'STMN2'],
                                   top20cellssb.tolist() + top20cells6IN.tolist() + top20cellss0.tolist() + top20cellsGN.tolist()]

#+ top30genessopc.tolist()   top20cellssopc.tolist()  top30genessmo.tolist() + + top20cellssmo.tolist()
'''

topcellsheatmap = df_train_set.loc[top30genessd.tolist() + top30genessn.tolist() + top30genessopc.tolist() + top30genessmo.tolist() + top30genessa.tolist(),
                                   top20cellssd.tolist() + top20cellssn.tolist() + top20cellssopc.tolist() + top20cellssmo.tolist() + top20cellssa.tolist()]

#Can write a line to make nes genes has nes cells > dcx cells

"PTPRZ1" in names_sorted_coefsn
names_sorted_coefsn.tolist()[0:250]

mp.rcParams['savefig.dpi'] = 100
mp.figure(figsize=(25,25))
sns.set(font_scale=1) #All: 0.3; just dcx cluster: 0.8
sns.heatmap(topcellsheatmap, cmap = 'Blues', linewidths = 0.2, annot=False)
mp.savefig('TopcellsheatmapfilesForYr0.pdf')



#%% Viualize the top positive coefficients (excel csv list) for filesForYr0 picked cells only ----
sel_class = 'newsd'
nonzero_coef_bool = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ]
values_nonzero_coef = LR.coef_[ np.where( classes_names == sel_class )[0][0],: ]
names_nonzero_coef= df_dev_log.index

ix_nonzero_coef = np.argsort(values_nonzero_coef)[::-1]
values_sorted_coef = values_nonzero_coef[ix_nonzero_coef]
names_sorted_coef = names_nonzero_coef[ix_nonzero_coef]

avg_expr = df_dev.loc[names_sorted_coef, protogruop == sel_class].mean(1)
std_expr = df_dev.loc[names_sorted_coef, protogruop == sel_class].std(1)

# Plot in a matplotlib table
mp.rcParams['savefig.dpi'] = 160
colLabels=("Gene", "Weight", 'Avg %s' % sel_class, 'Std.Dev.')
#Changes made below
an = 1000
rows_list = zip(list(names_sorted_coef)[:an],\
                list(map(lambda x: '%.3f' % x, values_sorted_coef[:an])),\
                list(map(lambda x: '%.3f' % x, avg_expr[:an])),\
                list(map(lambda x: '%.3f' % x, std_expr[:an])))
rows_listshow = list(rows_list)

#shape = len(rows_listshow)
myexcel = pd.DataFrame(np.array(rows_listshow).reshape(-1,4), columns = colLabels)
#writer = pd.ExcelWriter('Vistopgenes.xlsx', engine='xlsxwriter')
#myexcel.to_excel(writer, sheet_name='sheet1')
#writer.save()
#myexcel.to_csv(r'D:\Users\zeiss\Documents\VistopgenesfilesForYr0_sdd_1.csv', index=None, header=True)
myexcel.to_csv(r'VistopgenesfilesForYr0_sdd_1.csv', index=None, header=True)



#%% Unpicked filesForYr0 data (dup0) ----
#Trying to unify the two dataset cluster names
protodup0 = pd.Series({'neuron':'none', 'astro':'newsn', 'gn':'newsd', 'moli':'newsmo', 'opc':'newsopc', 'microglia':'newsa', 'other':'none','x':'none',
                       'sn':'newsn', 'sd':'newsd', 'smo':'newsmo', 'sopc':'newsopc', 'sa':'newsa'})

protocoldup0 = {'none': (190,10,10), 'neuron':(30,133,180), 'astro':(240,255,250), 'moli':(240,255,250),'gn':(240,255,250),'microglia':(50,80,80), 'other':(50,80,80),'opc':(250,80,80),
            'sn':(50,254,180), 'sa':(220,180,180),'sd':(220,180,180),'sopc':(30,133,180), 'smo':(50,254,180),'smg':(30,133,180),'si':(50,80,80), '5':(220,180,180), '7':(220,180,180),'8':(220,180,180),
            'snd':(50,254,180), 'sad':(220,180,180),'sdd':(150,200,50),'sopcd':(30,133,180), 'smod':(50,0,180),'smgd':(50,254,180),
            'newsn': (190,10,10), 'newsa': (190,10,10), 'newsd':(255,95,105), 'endo':(255,95,105), 'newsopc':(50,180,180),'newsmo':(50,180,180), 'newsmg':(50,180,180)}

cols_annot_alldup0 = filesForYr0cols_annotdup0
df_devdup0 = df_filesForYr0dup0.copy()

ct_devdup0 = cols_annot_alldup0[df_devdup0.columns].loc['Cell_type']
protogruopdup0 = protodup0.loc[ct_devdup0].values
df_devdup0 = df_devdup0.loc[:,protogruopdup0 != 'none']

#Pre-filtering
df_f = df_devdup0.copy()
df_f.shape

df_f = df_f.loc[:,(df_f>0).sum(0)>=0] #>= x genes is expressed in this cell
#df_f = df_f.loc[:,(df_f>0).sum(0)<=5000] #<= x genes is expressed in this cell

df_f = df_f.loc[(df_f>0).sum(1)>0,:] #>= x cells express this gene

df_devdup0 = df_f.copy()
del df_f

#Notice this is the list from filesForYr0
df_devdup0 = df_devdup0.loc[list(set(list_genes_out_of_nest)),:]

#Train model ----
#Prepare the reference dataset
# Log normalization
df_dev_logdup0 = np.log2(df_devdup0 + 1)
# Check if this is repeated
ct_devdup0 = cols_annot_alldup0[df_devdup0.columns].loc['Cell_type']
protogruopdup0 = protodup0.loc[ct_devdup0].values
bool1dup0 = protogruopdup0 != 'none'
classes_namesdup0, classes_indexdup0 = np.unique(protogruopdup0[bool1dup0], return_inverse=True, return_counts=False)
train_indexdup0 = classes_indexdup0
df_train_setdup0 = df_dev_logdup0.loc[:,bool1dup0].copy() # <- IndexError: indices are out-of-bounds
df_train_setdup0.shape



#%% Final model for unpicked dup0 ----

'''# Normalized by the max
LR = LogisticRegression(C=chos, penalty='l2', solver='newton-cg', fit_intercept=False,
                        multi_class='multinomial',class_weight='balanced',random_state=1)

LR.fit((df_train_set.values / normalizer).T, train_index)
LR.coef_.shape
#LR.intercept_
'''

normalizerdup0 = 0.9*df_train_setdup0.values.max(1)[:,np.newaxis]

unseendup0 = (df_train_setdup0.values / normalizerdup0).T

where_nan = np.isnan(unseendup0)
unseendup0[where_nan] = 0

#check on NaN and infinity, for it to pass the test, it should be: false, then true
print(np.any(np.isnan(unseendup0)))
print(np.all(np.isfinite(unseendup0)))



#%% Score with the model for unpicked dup0 ---- 
#hist_order = ['1', '5', '9', '26', '27']
#hist_order = ['sn', 'filesForYr0_3RGL', 'sd', 'filesForYr0_2GN', 'smo', 'filesForYr0_O', 'sopc', 'filesForYr0_8OPC']
hist_order = ['newsn','newsd', 'newsa',  'newsmo', 'newsopc']
#, 'filesForYr0_RGL', 'filesForYr0_O','filesForYr0_GN', 'si', 'sca'

hist_ixes = [list(classes_names).index(i) for i in hist_order]

mp.figure(figsize=(10,4))
pobs_listdup0 = []
for z,i in enumerate(hist_ixes):
    mp.subplot(len(set(classes_index))/1,1,z+1)
    unseendata = (df_train_setdup0.values[:,classes_indexdup0==i]/ normalizerdup0).T
    datawhere_nandup0 = np.isnan(unseendata)
    unseendata[datawhere_nandup0] = 0
    probdup0 = LR.predict_proba(unseendata)[:,i]
    amounts,_,_ = mp.hist(probdup0, color=np.array(protocol[classes_names[i]])/255.,bins=np.linspace(0.2,1,100) )
    pobs_listdup0.append([ classes_names[i], np.mean(probdup0)])
    
    if z == len(hist_ixes)-1:
        mp.tick_params('both', which='both', right='on',left='off',top='off', labelleft='off',labelbottom='on',labelright='on')
    
    mp.axvline(0.5)
    mp.xlim(0,1)
    mp.ylabel(classes_names[i],rotation='horizontal', horizontalalignment = 'right')
    mp.yticks(np.linspace(0, max(amounts), 4 )[1:],['','','%d' % ( max(amounts) )])
    mp.ylim(0,1.2*max(amounts))
    mp.tick_params('both', which='both', right='on',left='off',top='off', labelleft='off',labelbottom='off',labelright='on')


#mp.tight_layout(pad=1,h_pad=0,w_pad=1)
mp.savefig('ScoreModeldup0.pdf')
#Lots warnings, but worked.
#Notice here joeyunix16 is edited, compared to joeyunix15 or before



#%% Wheel/Polygonal plot for unpicked dup0 ----
#Plot wheel plot (with filesForYr0 dataset)
wanted_orderdup0 = ['newsd', 'newsn', 'newsopc',  'newsmo','newsa']
reorder_ixdup0 = [list(classes_namesdup0).index(i) for i in wanted_orderdup0]
bool00dup0 = np.in1d( classes_namesdup0[classes_indexdup0],  wanted_orderdup0 )
color_dictdup0 = pd.Series({
                        'sn':(20,20,20),'astro': (200,200,200),
                        'sa':(54,27,0),'microglia':(255,200,150),
                        'sd':(255,0,0),'gn':(200,255,0),
                        'sopc':(76,0,150),'opc': (227,159,246),
                        'smo':(0,0,200),'moli':(137,207,240),
                        'sglut':(0,0,0),'sbv':(0,0,0),'seuratglut':(137,7,240),'endo':(137,7,240),
                        'sgaba':(0,0,0),'seuratgaba':(137,107,240),
                        'neuron':(255, 153, 153)})

color_dictdup0 = color_dictdup0.map(lambda x: list(map(lambda y: y/255., x)))

mp.rcParams['savefig.dpi'] = 90
newcolorsdup0 = np.array(list(color_dictdup0[ct_devdup0].values))

newdatadup0,axdup0 = polygonalPlot(LR.predict_proba(unseendup0)[:,reorder_ixdup0],\
                         scaling=False, sides=len(reorder_ixdup0), labels=classes_namesdup0[reorder_ixdup0])

axdup0.scatter( newdatadup0[bool00dup0,0]*0.99, newdatadup0[bool00dup0,1]*0.99, alpha=0.6,\
           c=newcolorsdup0[bool00dup0,],\
           s = 25, lw=0.5)


mp.savefig('Plotwheelplotdup0.pdf')



#%% Run this if you don't want to do round 2:

list_genes_out_of_nestdup02 = list_genes_out_of_nest
classes_namesdup02 = classes_namesdup0
classes_indexdup02 = classes_indexdup0
LRdup02 = LR
protocoldup02 = protocoldup0
hist_orderdup02 = ['newsn', 'newsa', 'newsd', 'newsmo', 'newsopc']

