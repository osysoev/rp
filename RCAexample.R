
# A function to construct references from external reference data set
# Based on the RCA codes, correlation computations were changed to increase the performance
featureConstructRef=function(data_obj,refData, power=4){

  
  data=data_obj$fpkm_transformed
  data1=refData
  
  commongenes=intersect(rownames(data), rownames(data1))
  data1=data1[commongenes,]
  data=data[commongenes,]

  data3=cor(data1, data, method="pearson")
  data4=as.data.frame(data3)
  data5=abs(data4)^power*sign(data4)
  data6=scale(data5, center = T, scale = T)
  data_obj$fpkm_for_clust = as.data.frame(data6)
  
  return(data_obj)
  
}

#Cell annotation function, based on the principle described in (Aran et al 2019, Supplementary Materials)
labelCells <- function(data_obj){
  data=data_obj$fpkm_for_clust
  Labels=data_obj$group_labels_color[["groupLabel"]]
  Types=rownames(data)
  TypePerCell=apply(data, MARGIN = 2, FUN= function(x) Types[which.max(x)])
  data1=as.data.frame(t(data))
  confus=as.data.frame.matrix(table(Labels, TypePerCell))
  nm=colnames(confus)
  print(confus)
  LabelMap=nm[apply(confus, 1, which.max)]
  CellType=LabelMap[Labels+1]
  data_obj$CellType=CellType
  return(data_obj)
  
  
}

d=read.csv("data/ReferenceData.csv", header = T, stringsAsFactors = F)
data_cellline =  read.table("data/scdata.txt", header = T, row.names = 1, stringsAsFactors = F)


library(dplyr)
library(RCA)


d$MainCellType=c()
d$FineCellType=as.factor(d$FineCellType)
d1=aggregate(d[,-1],by=list(CT=d$FineCellType), FUN=mean)
rownames(d1)<-d1$CT
d1=as.data.frame(d1)
d1$CT=c()
d1=as.matrix(d1)
d1=t(d1)

TrueCellTypes=c(rep('CD4+',2500), rep('Monocyte',2500), rep('CD8+',2500), rep('B-cell',2500), rep('NK-cell',2500))
refData=d1

fpkm_data = data_cellline
data_obj = dataConstruct(fpkm_data);
data_obj = geneFilt(obj_in = data_obj);
data_obj = cellNormalize(data_obj);
data_obj = dataTransform(data_obj,"log10");
data_obj = featureConstructRef(data_obj, refData);
data_obj = cellClust(data_obj)
data_obj = labelCells(data_obj)

table(TrueCellTypes, data_obj$CellType)
