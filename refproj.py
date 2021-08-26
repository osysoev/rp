
import pandas as pd
import numpy as np
import scanpy as sc
from numpy.random import choice
import scvi
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import linear_model
import math
from statsmodels.distributions.empirical_distribution import ECDF


def denoise(ccData, refData, batch_key=None, modelFolder=None, saveModel=True, use_cuda=True, trueLabels=None):
    fullData=pd.concat([ccData,refData], join='inner', sort=False)
    nR=refData.shape[0]
    nS=ccData.shape[0]
    p=fullData.shape[1]
    
    ccData0=sc.AnnData(fullData.to_numpy()[:nS,:])
    
    if modelFolder==None:
        ind=choice(list(range(nS)), nS, replace=False)
        ccData2=sc.AnnData(fullData.to_numpy()[ind,:])
        scvi.data.setup_anndata(ccData2, batch_key=batch_key)
        vae = scvi.model.SCVI(ccData2, n_latent=10, n_layers=3, n_hidden=64)
        vae.train()
        if saveModel:
            vae.save("./vae",save_anndata=True)            
    else:
        vae=scvi.model.SCVI.load("./vae", use_cuda=use_cuda) 
    
    latent=vae.get_latent_representation(ccData0)
    

    
    
    denoised=vae.get_normalized_expression(ccData0, library_size="latent")
    
    return vae,latent,denoised


def create_clusters(latent, resolution):
    ad=sc.AnnData(latent)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad, min_dist=0.1)
    sc.tl.leiden(ad, key_added="leiden_scvi", resolution=resolution)

    
    return ad.obs["leiden_scvi"]
    

from scipy.stats import spearmanr

def corr_spear(A,B):
    rho1,pi=spearmanr(A, B, axis=1)
    sA=A.shape[0]
    return rho1[:sA,sA:]


def project_ref(ccData, ref, denoised, CellLabel):
    fullData=pd.concat([ccData,ref], join='inner', sort=False)
    nR=ref.shape[0]
    nS=ccData.shape[0]
    p=fullData.shape[1] 
    
    ccData1=fullData.iloc[:nS,:]
    ref1=fullData.iloc[nS:,:]
    
    rhoR=corr_spear(ccData1.to_numpy(), ref1.to_numpy())
    rhoR2=corr_spear(denoised.to_numpy(), ref1.to_numpy())

    refn=np.zeros(shape=ref1.shape)
    
    win=np.argmax(rhoR, axis=1)
    Label=CellLabel[win]
    filtered=np.where(np.isin(CellLabel, Label))[0]


    for i in filtered:
        Cand=rhoR2[:,i]
        index=Cand.argmax()
        y=ccData1.iloc[index,:].to_numpy()
        yhat=ref1.iloc[i,].to_numpy()
        
        yhat[np.argsort(yhat)]=np.round(y[np.argsort(y)])
    
        refn[i,:]=yhat
    
    return filtered, refn



def rp(ccData, ref, refLabels, refProj="project_ref+scvi", clusterLabels="create_cluster", resolution=0.5, ccDataDenoised="scvi", 
                   ccDataLatent="scvi", 
                    batch_key=None, plot=True, modelFolder=None, saveModel=True, savePlots=False,
                   use_cuda=True, trueLabels=None):
    """Performs cell annotation with RP method.


    Parameters
    ----------
    ccData 
        single cell data set matrix
    ref
        bulk reference matrix
    refLabels
        labels for reference observations
    refProj
        how to project bulk data into single cell
    clusterLabels
        clustering approach to be used
    resolution
        resolution for Lovain clustering
    ccDataDenoised
        'scvi' leads to automatic denoising by scvi, alternatively provide your denoised single cell data
    ccDataLatent
        'scvi' leads to automatic denoising by scvi, alternatively provide your latent single cell data
    batch_key
        a vector showing batches of single cell data
    plot
        shall the plots be produced
    modelFolder
        If none, scVI is computed from scratch. If True, the scVI model is reloaded from the saved location (assumes that it was saved there earlier)
    saveModel
        If True, scVI model is saved to a predefined folder
    save 
        If True, plots are saved to a predefined folder
    
    use_cuda
        if CUDA shall be used
    trueLabels
        If a vector of true cell types provided it enables plotting these labels in the latent dimensions
    
    ------
    Returns
    
    Labs, CellT, Cell
        Cell type annotations, cell types per cluster and probabilities per cluster
    """    

    if ccDataDenoised=="scvi" and ccDataLatent=="scvi":
        vae, latent, denoised= denoise(ccData, ref, batch_key, modelFolder, saveModel, use_cuda, trueLabels)
    else:
        latent=ccDataLatent
        denoised=ccDataDenoised
    
    if refProj=="project_ref+scvi":
        filtered, refn=project_ref(ccData, ref, denoised, refLabels)
        RefLabel=refLabels[filtered]
        refn2=refn[filtered,:]
        refP=vae.get_normalized_expression(sc.AnnData(refn2))
        refL=vae.get_latent_representation(sc.AnnData(refn2))

    else:
        filtered=list(range(refLabels.size))
        RefLabel=refLabels
    
    if clusterLabels=="create_cluster":
        ClusterID=create_clusters(latent, resolution)
    else:
        ClusterID=clusterLabels
    
    fullData=pd.concat([ccData,ref], join='inner', sort=False)
    nR=ref.shape[0]
    nS=ccData.shape[0]
    p=fullData.shape[1] 
    
    ccData1=fullData.iloc[:nS,:]
    ref1=fullData.iloc[nS:,:]
    
    ad=sc.AnnData(np.concatenate((latent,refL)))
    sc.pp.pca(ad)
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.umap(ad, min_dist=0.1)
    ad.obs["CellTypes_References"]=["Single Cell"]*latent.shape[0]+RefLabel.tolist()

    sc.pl.umap(ad, color=["CellTypes_References"], size=[12]*nS+[100]*RefLabel.size, save="Projected.pdf" if savePlots else False)
     
    
    Xtr=np.log(denoised.to_numpy())
    Ytr=np.array(ClusterID)
    Xte=np.log(refP.to_numpy())#
    

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr=scaler.transform(Xtr)
    Xte=scaler.transform(Xte)
    

    reg=linear_model.LogisticRegressionCV(penalty='l2', multi_class="multinomial")
    reg.fit(Xtr, Ytr) 
    
    nC=np.unique(Ytr)
    CellT={}
    CellP={}

    CellType=pd.DataFrame({"First":np.array(nC), "Second":np.array(nC)})
    ProbType=pd.DataFrame(np.empty(shape=(len(nC),2)))
    WeightType=pd.DataFrame(np.empty(shape=(len(nC),2)))

    for i in range(nC.size):
        obsI=np.where(np.array(ClusterID)==nC[i])[0]
        scores=np.matmul(Xtr[obsI,:], np.transpose(reg.coef_[i,:]))

        ecdf=ECDF(scores)
        scores1=np.matmul(Xte, np.transpose(reg.coef_[i,:]))
        dd=pd.DataFrame()

        weights=ecdf(scores1)+1e-12


        scores2=weights*1/(1+np.exp(-scores1+ reg.intercept_[i]))
        dd["num"]=scores2
        dd["label"]=RefLabel
        dd["weight"]=weights
        ddS=dd.sort_values(by="num",ascending=False)

        ddS1=dd.groupby("label").agg("max")
        ddS1=ddS1.sort_values(by="num",ascending=False)
        ddS1["num"]=np.round(ddS1["num"]/ddS1["num"].sum()*100)
        ddS1=ddS1.iloc[:2,:]
        ddS2=ddS1.sort_values(by="num",ascending=False)

        CellType.iloc[i,0]=ddS2.index[0]
        CellType.iloc[i,1]=ddS2.index[1]

        ProbType.iloc[i,0]=ddS2["num"][0]
        ProbType.iloc[i,1]=ddS2["num"][1]

        WeightType.iloc[i,0]=ddS2["weight"][0]
        WeightType.iloc[i,1]=ddS2["weight"][1]
        print(ddS2)
        CellT[nC[i]]=str(CellType.iloc[i,0])
        CellP[nC[i]]=str(ProbType.iloc[i,0])
    
    
    Labs=ClusterID.to_list()
    LabsP=ClusterID.to_list()
    Tlab=ClusterID.to_list()
    for i in range(len(Tlab)):
        LabsP[i]=CellT[Tlab[i]]
        Labs[i]=CellT[Tlab[i]]
    
    if plot:
        if refProj=="project_ref+scvi":
            ad=sc.AnnData(np.concatenate((latent,refL)))
        else:
            ad=sc.AnnData(latent)

        sc.pp.pca(ad)
        sc.pp.neighbors(ad, n_neighbors=15)
        sc.tl.umap(ad, min_dist=0.1)
        sc.pl.umap(ad[:nS,:],save="latent.pdf" if savePlots else False)
        
        if trueLabels is not None:
            ad.obs["TrueCellTypes"]=trueLabels.tolist()+['1']*filtered.size
            sc.pl.umap(ad[:nS,:], color=["TrueCellTypes"], save="TrueCellTypes.pdf" if savePlots else False)  
            
        ad.obs["Clusters"]=ClusterID.to_list()+['1']*filtered.size
        sc.pl.umap(ad[:nS,:], color=["Clusters"], save="Clusters.pdf" if savePlots else False)
        
        ad.obs["CellTypes"]=LabsP+['1']*filtered.size
        sc.pl.umap(ad[:nS,:], color=["CellTypes"], save="Decision.pdf" if savePlots else False)
        
        
        if refProj=="project_ref+scvi":
            ad.obs["CellTypes_References"]=["Single Cell"]*latent.shape[0]+RefLabel.tolist()

            sc.pl.umap(ad, color=["CellTypes_References"], size=[12]*nS+[100]*RefLabel.size, save="Projected.pdf" if savePlots else False)
    
    return Labs, CellT, CellP


    


