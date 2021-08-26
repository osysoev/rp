
d=read.csv("data/ReferenceData.csv", header = T, stringsAsFactors = F)
data=  read.table("data/scdata.txt", header = T, row.names = 1, stringsAsFactors = F)

data=as.data.frame(t(data))
data1=data
d1=d
Labels=d1$FineCellType
d1$MainCellType=c()
d1$FineCellType=c()
refData=t(as.matrix(d1))
ccData=t(as.matrix(data1))

TrueCellTypes=c(rep('CD4+',2500), rep('Monocyte',2500), rep('CD8+',2500), rep('B-cell',2500), rep('NK-cell',2500))




library(SingleR)

pred=SingleR(ccData, refData, labels=Labels)      
table(truth=TrueCellTypes, predicted=pred$labels)
