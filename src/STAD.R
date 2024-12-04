
setwd("TCGA ESTIMATE")  
library(utils) 
install.packages("estimate", repos=rforge, dependencies=TRUE)
library(estimate)
library(tidyverse)

expr <- read.table("STAD_fpkm_mRNA_01A.txt",sep = "\t",row.names = 1,check.names = F,stringsAsFactors = F,header = T)

# Calculate immune scores
filterCommonGenes(input.f = "STAD_fpkm_mRNA_01A.txt", 
                  output.f = "STAD_fpkm_mRNA_01A.gct",
                  id = "GeneSymbol")   

estimateScore("STAD_fpkm_mRNA_01A.gct",
              "STAD_fpkm_mRNA_01A_estimate_score.txt", 
              platform="affymetrix") 

# 3. Output scores for each sample
result <- read.table("STAD_fpkm_mRNA_01A_estimate_score.txt", sep = "\t", row.names = 1, check.names = F, stringsAsFactors = F, header = T)
result <- result[,-1]   
colnames(result) <- result[1,]   
result <- as.data.frame(t(result[-1,]))
rownames(result) <- colnames(expr)
write.table(result, file = "STAD_fpkm_mRNA_01A_estimate_score.txt", sep = "\t", row.names = T, col.names = NA, quote = F) 

setwd("survival")
surv <- read.table("exp_sur_01A.txt", sep = "\t", row.names = 1, check.names = F, stringsAsFactors = F, header = T)

surv$OS.time <- surv$OS.time * 12

# Median
median(surv$SPP1)

surv$group <- ifelse(surv$SPP1 > median(surv$SPP1), "High", "Low")
surv$group <- factor(surv$group, levels = c("Low", "High")) 
class(surv$group)
table(surv$group)

library(survival)
fitd <- survdiff(Surv(OS.time, OS) ~ group,
                 data      = surv,
                 na.action = na.exclude)

pValue <- 1 - pchisq(fitd$chisq, length(fitd$n) - 1)

# 2.2 Fit the survival curve
fit <- survfit(Surv(OS.time, OS) ~ group, data = surv)

plot(fit, conf.int = T,
     col = c("blue", "red"),
     lwd = 2,
     xlab = "Time(Months)",
     ylab = "Survival probability(%)"
)

legend("topright",
       title = "Group",
       c("Low", "High"),
       lwd = 2, lty = 1,
       col = c("blue", "red"))

p.lab <- paste0("P", ifelse(pValue < 0.001, " < 0.001", paste0(" = ", round(pValue, 3))))

text(25, 0.2, p.lab)

dev.off()

setwd("_fuji")
BiocManager::install('clusterProfiler')
library(tidyverse)
library("BiocManager")
library(org.Hs.eg.db)
library(clusterProfiler)

# 3. Visualization
## 3.1 Bar plot
barplot(ego, showCategory = 20, color = "pvalue")

## 3.2 Bubble plot
dotplot(ego, showCategory = 20) # Draw scatter plot

## 3.3 Classification display
barplot(ego, drop = TRUE, showCategory = 10, split = "ONTOLOGY") + 
  facet_grid(ONTOLOGY ~ ., scale = 'free')

dotplot(ego, showCategory = 10, split = "ONTOLOGY") + 
  facet_grid(ONTOLOGY ~ ., scale = 'free')

# ssGSEA
setwd("xena") # Expression of immune cells
BiocManager::install('GSVA')
library(tidyverse)
library(data.table)
library(GSVA)

# 1.2 Prepare cell markers
cellMarker <- data.table::fread("cellMarker.csv", data.table = F)
colnames(cellMarker)[2] <- "celltype"
type <- split(cellMarker, cellMarker$celltype)

cellMarker <- lapply(type, function(x) {
  dd = x$Metagene
  unique(dd)
})

save(cellMarker, file = "cellMarker_ssGSEA.Rdata")

expr <- data.table::fread("STAD_fpkm_mRNA_01A.txt", data.table = F) # Load expression file
rownames(expr) <- expr[,1]   
expr <- expr[,-1]   
expr <- as.matrix(expr)   

# 2. Quantify immune infiltration using ssGSEA
gsva_data <- gsva(expr,cellMarker, method = "ssgsea")
a <- gsva_data %>% t() %>% as.data.frame()
identical(rownames(a),rownames(group))
a$group <- group$group
a <- a %>% rownames_to_column("sample")
write.table(a,"ssGSEA.txt",sep = "\t",row.names = T,col.names = NA,quote = F)
library(ggsci)
library(tidyr)
library(ggpubr)
b <- gather(a,key=ssGSEA,value = Expression,-c(group,sample))
ggboxplot(b, x = "ssGSEA", y = "Expression",
          fill = "group", palette = "lancet")+
  stat_compare_means(aes(group = group),
                     method = "wilcox.test",
                     label = "p.signif",
                     symnum.args=list(cutpoints = c(0, 0.001, 0.01, 0.05, 1),
                                      symbols = c("***", "**", "*", "ns")))+
  theme(text = element_text(size=10),
        axis.text.x = element_text(angle=45, hjust=1)) 
dev.off()


### X cell ###
chooseBioCmirror()
devtools::install_github('dviraran/xCell')
library(xCell)
library(ggpubr)
library(tidyverse)
setwd("Xcell")
exp <- read.table("STAD_fpkm_mRNA_01A.txt",sep = "\t",row.names = 1,check.names = F,stringsAsFactors = F,header = T)

celltypeuse<-xCell.data$spill$K
rs<-xCellAnalysis(exp,parallel.sz=10) 

gene <- "PDCD1"
med=median(as.numeric(exp[gene,]))
expgene <- exp[gene,]
expgene <- expgene %>% t() %>% as.data.frame()
expgene$group <- ifelse(expgene$PDCD1>med,"High","Low")
rs <- as.data.frame(rs)
rs <- rs %>% t() %>% as.data.frame()
comname <- intersect(rownames(rs),rownames(expgene)) 
rs <- rs[comname,]
expgene <- expgene[comname,]
identical(rownames(rs),rownames(expgene))
rs$group <- as.factor(expgene$group)
class(rs$group)
rs <- rs %>% rownames_to_column("sample")
a <- rs
library(ggsci)
library(tidyr)
library(ggpubr)
b <- gather(a,key=xCell,value = Expression,-c(group,sample))
ggboxplot(b, x = "xCell", y = "Expression",
          fill = "group", palette = "lancet")+
  stat_compare_means(aes(group = group),
                     method = "wilcox.test",
                     label = "p.signif",
                     symnum.args=list(cutpoints = c(0, 0.001, 0.01, 0.05, 1),
                                      symbols = c("***", "**", "*", "ns")))+
  theme(text = element_text(size=10),
        axis.text.x = element_text(angle=45, hjust=1)) 
dev.off()

#### Volcano Plot for TCGA Differential Expression Analysis ####
setwd("xena")
library(tidyverse)
exp <- read.table("STAD_fpkm_mRNA_all.txt", sep = "\t", row.names = 1, check.names = F, stringsAsFactors = F, header = T)
DEG <- as.data.frame(res) %>% 
  arrange(padj) %>% 
  dplyr::filter(abs(log2FoldChange) > 0, padj < 0.05)

logFC_cutoff <- 1
type1 = (DEG$padj < 0.05) & (DEG$log2FoldChange < -logFC_cutoff)
type2 = (DEG$padj < 0.05) & (DEG$log2FoldChange > logFC_cutoff)
DEG$change = ifelse(type1, "DOWN", ifelse(type2, "UP", "NOT"))

table(DEG$change)

install.packages("ggpubr")
install.packages("ggthemes")
library(ggpubr)
library(ggthemes)

DEG$logP <- -log10(DEG$padj)
ggscatter(DEG,
          x = "log2FoldChange", y = "logP") +
  theme_base()

# Add up-regulated and down-regulated gene information
ggscatter(DEG, x = "log2FoldChange", y = "logP",
          color = "change",
          palette = c("blue", "black", "red"),
          size = 1) +
  theme_base()

ggscatter(DEG, x = "log2FoldChange", y = "logP", xlab = "log2FoldChange",
          ylab = "-log10(Adjust P-value)",
          color = "change",
          palette = c("blue", "black", "red"),
          size = 1) +
  theme_base() +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed")
dev.off()

# Add gene label information
DEG$Label = ""   
DEG <- DEG[order(DEG$padj), ]
DEG$Gene <- rownames(DEG)

up.genes <- head(DEG$Gene[which(DEG$change == "UP")], 5)
down.genes <- head(DEG$Gene[which(DEG$change == "DOWN")], 5)
DEG.top5.genes <- c(as.character(up.genes), as.character(down.genes))
DEG$Label[match(DEG.top5.genes, DEG$Gene)] <- DEG.top5.genes
match("a", c("a", "b", "c"))

ggscatter(DEG, x = "log2FoldChange", y = "logP",
          color = "change",
          palette = c("blue", "black", "red"),
          size = 1,
          label = DEG$Label,
          font.label = 8,
          repel = T,
          xlab = "log2FoldChange",
          ylab = "-log10(Adjust P-value)") +
  theme_base() +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed")
dev.off()


#### Cox Regression Analysis ####
setwd("cox")
install.packages("survival")
install.packages("forestplot")
library(survival)
library(forestplot)
library(tidyverse)

surv = read.table(file = 'TCGA-STAD.survival.tsv', sep = '\t', header = TRUE)

# Organize survival information data
surv$sample <- gsub("-", ".", surv$sample)
rownames(surv) <- surv$sample
surv <- surv[,-1]
surv <- surv[,-2]

# Read expression data
expr <- read.table("STAD_fpkm_mRNA_all.txt", sep = "\t", row.names = 1, check.names = F, stringsAsFactors = F, header = T)
comgene <- intersect(colnames(expr), rownames(surv))
table(substr(comgene, 14, 16))
expr <- expr[, comgene]
surv <- surv[comgene, ]
res_deseq2 <- as.data.frame(res) %>% 
  arrange(padj) %>% 
  dplyr::filter(abs(log2FoldChange) > 2, padj < 0.05)

# Integrate data
deg_expr <- expr[rownames(res_deseq2), ] %>% t() %>% as.data.frame()
surv.expr <- cbind(surv, deg_expr)

# Cox analysis
Coxoutput <- NULL 
for(i in 3:ncol(surv.expr)){
  g <- colnames(surv.expr)[i]
  cox <- coxph(Surv(OS.time, OS) ~ surv.expr[, i], data = surv.expr) 
  coxSummary = summary(cox)
  
  Coxoutput <- rbind.data.frame(Coxoutput,
                                data.frame(gene = g,
                                           HR = as.numeric(coxSummary$coefficients[,"exp(coef)"])[1],
                                           z = as.numeric(coxSummary$coefficients[,"z"])[1],
                                           pvalue = as.numeric(coxSummary$coefficients[,"Pr(>|z|)"])[1],
                                           lower = as.numeric(coxSummary$conf.int[, 3][1]),
                                           upper = as.numeric(coxSummary$conf.int[, 4][1]),
                                           stringsAsFactors = F),
                                stringsAsFactors = F)
}

write.table(Coxoutput, file = "cox results.txt", sep = "\t", row.names = F, col.names = T, quote = F)

### Select top genes
pcutoff <- 0.001
topgene <- Coxoutput[which(Coxoutput$pvalue < pcutoff), ] # Extract genes with p-values less than the threshold
topgene <- topgene[1:10, ]

#### 3. Draw forest plot ####
## 3.1 Create input table
tabletext <- cbind(c("Gene", topgene$gene),
                   c("HR", format(round(as.numeric(topgene$HR), 3), nsmall = 3)),
                   c("lower 95%CI", format(round(as.numeric(topgene$lower), 3), nsmall = 3)),
                   c("upper 95%CI", format(round(as.numeric(topgene$upper), 3), nsmall = 3)),
                   c("pvalue", format(round(as.numeric(topgene$pvalue), 3), nsmall = 3)))

##3.2 Draw forest plot
forestplot(labeltext=tabletext,
           mean=c(NA,as.numeric(topgene$HR)),
           lower=c(NA,as.numeric(topgene$lower)), 
           upper=c(NA,as.numeric(topgene$upper)),
           graph.pos=5,
           graphwidth = unit(.25,"npc"),
           fn.ci_norm="fpDrawDiamondCI",
           col=fpColors(box="#00A896", lines="#02C39A", zero = "black"),
           
           boxsize=0.4,
           lwd.ci=1,
           ci.vertices.height = 0.1,ci.vertices=T,
           zero=1,
           lwd.zero=1.5,
           xticks = c(0.5,1,1.5),
           lwd.xaxis=2,
           xlab="Hazard ratios",
           txt_gp=fpTxtGp(label=gpar(cex=1.2),
                          ticks=gpar(cex=0.85),
                          xlab=gpar(cex=1),
                          title=gpar(cex=1.5)),
           hrzl_lines=list("1" = gpar(lwd=2, col="black"),   
                           "2" = gpar(lwd=1.5, col="black"), 
                           "12" = gpar(lwd=2, col="black")), 
           lineheight = unit(.75,"cm"),
           colgap = unit(0.3,"cm"),
           mar=unit(rep(1.5, times = 4), "cm"),
           new_page = F
)
dev.off()