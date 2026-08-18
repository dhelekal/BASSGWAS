#!/usr/bin/env Rapp
#| name: annotate_all_pyseer
#| description: annotate all unitig patterns using annotate_hits_pyseer. Requires a pyseer installation.

#| description: vars2patts file as returned by prepare_data.R
#| required: true
vfile <- ""

#| description: reference list as required by annotate_hits_pyseer
#| required: true
rfile <- ""

#| description: output file name
#| required: true
out <- ""

#| description: working directory
dir <- "."

library(tidyverse)

seqs <- read_csv(vfile)

ln1 <- c("")
lns <- sapply(seqs[["variant"]], paste0)
lns <- unique(unname(lns))

writeLines(c(ln1,lns), paste0(dir,"/hits_tmp.txt"))
system(paste0("annotate_hits_pyseer ", dir, "/hits_tmp.txt ", rfile, " ", dir, "/annotated.txt"))
anno <- read_delim(paste0(dir,"/annotated.txt"),col_names=c("sequence","ann"),delim="\t")
anno <- anno %>% mutate(ann = str_extract_all(anno$ann, "[^;,]+:\\d+-\\d+[^,]+")) %>% unnest_longer(ann)
anno <- anno %>% mutate(contig = str_extract(ann, "[^:]+(?=:)"), range = str_extract(ann, "\\d+-\\d+"), ann = str_extract_all(ann,"(?<=((;)(\")?))[\\w\\.]*(?=(\")?($|[;,]))")) %>%
	unnest_wider(ann, names_sep ="_") %>%
    pivot_longer(c(ann_1,ann_2,ann_3),values_to="locus") %>% 
    select(!c(name)) %>% 
    distinct() %>%
    mutate(locus = ifelse(is.na(locus), "unannotated", locus)) %>%
    mutate(locus = ifelse(locus=="", "unannotated", locus)) %>%
	mutate(locus = ifelse(is.na(contig), "unmapped", locus)) %>%
    distinct() %>%
    mutate(down=as.integer(str_extract(range, "^\\d+")), 
        up=as.integer(str_extract(range, "\\d+$"))) %>%
	select(!range) 

anno %>% write_csv(out)