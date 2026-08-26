#!/usr/bin/env Rapp
#| name: prepare_data
#| description: Preprocess BASSGWAS input data

#| description: Variant presence absence tab-delimited file. Each row must correspond to a variant. First column will be used as variant ID. Column names must match sample names.
#| required: true
vfile <- NULL

#| description: Tree file in newick format. Tip names must match sample names.
#| required: true
tfile <- NULL

#| description: Output directory.
odir <- "."

#| description: SVD truncation threshold
svd_threshold <- 0.99
#| description: Allele frequency filter threshold
af_threshold <- 0.0

#| description: Text file with isolate identifiers of isolates to drop
#| required: false
to_drop <- NA_character_

suppressPackageStartupMessages(library(tidyverse))
library(ape)

{
vars <- vroom::vroom(vfile)
phylo <- read.tree(tfile)

colnames(vars) <- c("variant_id", colnames(vars)[-1])
vids <- vars[[1]]

stopifnot("Variant identifiers must be unique!"=length(unique(vids))==length(vids))
stopifnot("Some samples present in the phylogeny are missing from the variant file!"=all(phylo$tip.label %in% colnames(vars)))
stopifnot("Some samples present in the variant file are missing from the phylogeny!"=all(colnames(vars)[-1] %in% phylo$tip.label))

#drop bad isolates
if (!is.na(to_drop))
{
    drop <- readLines(to_drop)
    phylo <- drop.tip(phylo, drop)
}

vars <- vars %>%
    select(all_of(c("variant_id",phylo$tip.label))) %>%
    relocate(all_of(c("variant_id",phylo$tip.label)))

stopifnot(all(colnames(vars)[-1] == phylo$tip.label))

patts <- t(as.matrix(vars[,-1]))

#Convert to minor allele frequency
n <- nrow(patts)
signs <- rep(1L, ncol(patts))
for (i in 1:ncol(patts))
{
    v <- patts[, i]
    sv <- sum(v)
    af <- sv/n
    if(af < af_threshold || af > (1-af_threshold))
    {
        patts[, i] <- 0L
    }
    else if (sv > n/2)
    {
	    signs[i] <- -1L
        patts[, i] <- 1L-v
    }
}

pattern_ids <- apply(patts, 2, paste0, collapse="")
patts_minor <- as_tibble(t(patts)) %>% 
    mutate(pattern = pattern_ids, 
        variant = vids,
	    direction = signs)

patts_minor <- patts_minor %>% 
    filter(!if_all(!all_of(c("pattern", "variant","direction")), ~ .x == 0)) %>%
    mutate(pattern = as.integer(factor(pattern))) 

seqs <- patts_minor %>% 
    select(c("pattern", "variant","direction"))
patts_minor <- patts_minor %>% 
    select(!all_of(c("variant","direction"))) %>% 
    distinct(.keep_all=T)

patts_minor <- patts_minor %>% 
    mutate(pattern_new = 1:nrow(patts_minor))
seqs <- seqs %>% 
    left_join(patts_minor %>% select(c("pattern", "pattern_new")), by="pattern") 

variants_out <- seqs %>% select(c("pattern_new", "variant","direction")) %>% 
    rename(pattern_id=pattern_new)

patterns_out <- patts_minor %>% 
    select(!all_of(c("pattern")))%>%
    rename(pattern_id=pattern_new) %>% 
    relocate("pattern_id") %>%
    as.matrix() %>% t()

patterns_out <- patts_minor %>% 
    select(!all_of(c("pattern")))%>%#,"pattern_new"))) %>%
    rename(pattern_id=pattern_new) %>% 
    relocate("pattern_id") %>%
    as.matrix() %>% t()

colnames(patterns_out) <- paste0(patterns_out[1, ])
patterns_out <- patterns_out[-1, ]
patterns_ctr <- patterns_out

for (i in 1:ncol(patterns_ctr))
{
	patterns_ctr[,i] <- patterns_ctr[,i] - mean(patterns_ctr[,i])
}

popVCV <- vcv(phylo, corr=T)

SV <- svd(popVCV)
ssum <- sum(SV$d)
truncpt <- which(cumsum(SV$d)/ssum > svd_threshold)[1]

U = SV$u[,1:truncpt]
S = SV$d[1:truncpt]

write.csv(patterns_out, paste0(odir,"/patterns.csv"))
write.csv(patterns_ctr, paste0(odir,"/patterns_centred.csv"))
write.csv(variants_out, paste0(odir,"/vars2patts.csv"),row.names = FALSE)
write.csv(U, paste0(odir,"/U.csv"),row.names = FALSE)
write.csv(S, paste0(odir,"/S.csv"),row.names = FALSE)
}

