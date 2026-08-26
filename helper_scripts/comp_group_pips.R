#!/usr/bin/env Rapp
#| name: comp_group_pips
#| description: compute variant group PIPs

#| description: variant groups file. Must have variant column, and a group column
#| required: true
gfile <- NULL

#| description: output file name
#| required: true
out <- NULL

#| description: par_draws file
#| required: true
par_draws <- NULL

#| description: gammas_draws file
#| required: true
gammas_draws <- NULL

#| description: vars2patts file
#| required: true
vars2patts <- NULL

suppressPackageStartupMessages(library(tidyverse))
Sys.setenv(VROOM_CONNECTION_SIZE = 1500072)

pars <- vroom::vroom(par_draws, show_col_types = FALSE, progress = FALSE, altrep = FALSE)
pars <- pars %>% mutate(Z=Z/sum(Z),iteration=row_number())
Z <- pars$Z
nr <- nrow(pars)

gammas <- vroom::vroom(paste0(gammas_draws), show_col_types = FALSE, progress = FALSE, altrep = FALSE)
gammas <- gammas[,3:ncol(gammas)] > 0 
stopifnot(nrow(gammas)==nr)

variants <-vroom::vroom(vars2patts,show_col_types = FALSE, progress = FALSE, altrep = FALSE)
groups <- vroom::vroom(gfile, col_types = c(variant="c", group="c"), 
	show_col_types = FALSE, progress = FALSE, altrep = FALSE) %>% 
	left_join(variants, by="variant") %>%
	select(pattern_id, group) %>%
	group_by(pattern_id, group) %>%
	distinct %>%
	ungroup()

group_pips <- groups %>% 
	    group_by(group) %>%
   	    summarise(PIP = sum(pars$Z*apply(gammas[,pattern_id,drop=F],1,any))) %>%
        ungroup()

write_csv(group_pips, out)

