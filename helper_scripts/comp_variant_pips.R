#!/usr/bin/env Rapp
#| name: comp_group_pips
#| description: compute variant group PIPs

#| description: output file name
#| required: true
out <- NULL

#| description: par_draws file
#| required: true
par_draws <- NULL

#| description: pip_draws file
#| required: true
pip_draws <- NULL

#| description: vars2patts file
#| required: true
vars2patts <- NULL

library(tidyverse)
Sys.setenv(VROOM_CONNECTION_SIZE = 1500072)

pars <- vroom::vroom(par_draws, show_col_types = FALSE, progress = FALSE, altrep = FALSE)
pars <- pars %>% mutate(Z=Z/sum(Z),iteration=row_number())
Z <- pars$Z
nr <- nrow(pars)

pip_draws <- vroom::vroom(pip_draws, show_col_types = FALSE, progress = FALSE, altrep = FALSE)
pips <- apply(pip_draws[,3:ncol(pip_draws)]*Z,2,sum)
  
pips_df<-data.frame(pattern_id=1:length(pips), PIP=pips)
variants <-vroom::vroom(vars2patts,show_col_types = FALSE, progress = FALSE, altrep = FALSE)
pips_df %>% 
	left_join(variants,by="pattern_id") %>%
	write_csv(out)
