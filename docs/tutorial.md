-   [Introduction](#introduction)
-   [Input Data](#input-data)
    -   [Note on Input Types](#note-on-input-types)
    -   [Inspecting the input data](#inspecting-the-input-data)
    -   [Preparing the inputs](#preparing-the-inputs)
    -   [Observations file](#observations-file)
-   [GWAS](#gwas)
    -   [`bassgwass-run.jl` usage](#bassgwass-run.jl-usage)
    -   [Running GWAS on a toy dataset](#running-gwas-on-a-toy-dataset)
    -   [Identifying associations](#identifying-associations)
-   [Adaptive experimental design](#adaptive-experimental-design)
    -   [Setup](#setup)
    -   [Adaptive design](#adaptive-design)
    -   [Additional functionality](#additional-functionality)
-   [Tips](#tips)

First we load the necessary libraries used in this tutorial

    library(ape)
    library(tidyverse)
    library(ggplot2)

    set.seed(1)

**NOTICE: RE-RUNNING THIS TUTORIAL REQUIRES ~16GB OF RAM AND 4 CPU
CORES**

**EXPECTED EXECUTION TIME: ~25-45MIN**

# Introduction

This tutorial assumes that you have previously downloaded the files in
this repository. The first section of the tutorial covers the general
usage of the GWAS model. The second section covers adaptive experimental
design.

Throughout this tutorial we will be using the toy data found in the
`sample_data` folder. This is a version of the ciprofloxacin resistance
benchmark dataset used in the manuscript modified to contain only 20,000
unique variants. The toy dataset uses *unitigs* to characterise genetic
variation.

# Input Data

Using BASS-GWAS requires two main data files: a variant presence-absence
file, and a phylogeny file in the newick format. Both the phylogeny and
variant presence-absence should be constructed for the entire collection
at hand. The first column of the variant file should contain unique
variant identifiers. The remaining columns should each correspond to an
isolate, and the first row contain unique isolate identifiers:

<table>
<thead>
<tr>
<th>variant</th>
<th>isolate_1</th>
<th>isolate_2</th>
<th>isolate_3</th>
<th>…</th>
<th>isolate_n</th>
</tr>
</thead>
<tbody>
<tr>
<td>variant_1</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>…</td>
<td>1</td>
</tr>
<tr>
<td>variant_2</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>…</td>
<td>1</td>
</tr>
<tr>
<td>variant_3</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>…</td>
<td>0</td>
</tr>
<tr>
<td>…</td>
<td>…</td>
<td>…</td>
<td>…</td>
<td>…</td>
<td>…</td>
</tr>
<tr>
<td>variant_n</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>…</td>
<td>0</td>
</tr>
</tbody>
</table>

The isolate identifiers must be shared between the variant file and the
phylogeny. If using unitigs,
[unitig-caller](https://github.com/bacpop/unitig-caller) can create this
file for you.

## Note on Input Types

Representing genetic variation in bacterial populations is an active
area of research. There are many distinct approaches, for example using
VCFs, unitigs, kmers, gene presence-absence, etc. To ensure proper
functionality of BASS-GWAS, the representation of genetic variation
should be both *comprehensive*, and *precise*. This is especially
important when using adaptive sequential sampling. Unitigs are usually a
good choice.

**DO NOT use adaptive sequential sampling with highly redundant
representations such as kmers**

**DO NOT use adaptive sequential sampling with gene presence-absence
data alone**

## Inspecting the input data

The variant file in the toy dataset uses unitigs as variant identifiers
and `sra` accession numbers as isolate identifiers.

    vroom::vroom("../sample_data/variants.rtab",
        altrep=F, progress=F, show_col_types=F) %>%   
        print(n = 5, max_extra_cols = 0,width=60)

    ## # A tibble: 26,286 × 801
    ##   Unitig_sequence        SRR11698195 SRR11698198 SRR11698231
    ##   <chr>                        <dbl>       <dbl>       <dbl>
    ## 1 CATACCGGCGGGGGGGGGGGG…           0           0           0
    ## 2 CTTCAAGCAACGTAAACAAAG…           0           0           0
    ## 3 ACTTCTCTTCTCTTCTCTTCT…           0           0           0
    ## 4 CTAGATTCCCGCTTTCGCGGG…           0           0           0
    ## 5 AGAATGCCCTCTCCCCGGCCC…           0           1           1
    ## # ℹ 26,281 more rows

## Preparing the inputs

To generate the input files needed for use BASS-GWAS, we will need to
use the data preprocessing script. First we will create a directory for
storing the generated input files.

    mkdir gwas_inputs

Next we will use the data preprocessing script `prepare_data.R` to
process the variant presence-absence file and the phylogeny into
BASS-GWAS inputs. As a part of preprocessing, variant presence-absence
patterns will be standardised, centred, and deduplicated. The phylogeny
is used to compute a factorised, low-rank similarity matrix.

`prepare_data.R` usage:

    Rapp ../helper_scripts/prepare_data.R --help

    ## Usage: prepare_data [OPTIONS] <VFILE> <TFILE>
    ## 
    ## Preprocess BASSGWAS input data
    ## 
    ## Options:
    ##   --odir <ODIR>                   Output directory.
    ##                                   [default: "."] [type: string]
    ##   --svd-threshold <SVD-THRESHOLD>  SVD truncation threshold
    ##                                   [default: 0.99] [type: float]
    ##   --af-threshold <AF-THRESHOLD>   Allele frequency filter threshold
    ##                                   [default: 0] [type: float]
    ##   --to-drop <TO-DROP>             Text file with isolate identifiers of
    ##                                   isolates to drop
    ##                                   [default: "NA"] [type: string]
    ## 
    ## Arguments:
    ##   <VFILE>  Variant presence absence tab-delimited file. Each row must
    ##            correspond to a variant. First column will be used as variant ID.
    ##            Column names must match sample names.
    ##   <TFILE>  Tree file in newick format. Tip names must match sample names.

-   `--af-threshold` can be used to set an optional allele frequency
    filter. Variants with frequency lower than this value will be
    discarded. Default: 0.
-   `--svd_threshold` controls the fraction of variance of the original
    phylogenetic correlation matrix that will be retained. Default:
    0.99.

We will now use the script to preprocess the toy data. To speed up
inference for the purposes of this tutorial, we will retain only 90% of
the variance.

    Rapp ../helper_scripts/prepare_data.R ../sample_data/variants.rtab ../sample_data/tree.tre --odir gwas_inputs --svd_threshold 0.9

This will generate the following input files in a `gwas_inputs`
subdirectory.

The following two files are subsequently used with BASS-GWAS:

-   `patterns_centred.csv`: a centred, minorized, and deduplicated
    variant presence-absence file for use with BASS-GWAS.

-   `U.csv` & `S.csv`: a low-rank factorisation of the correlation
    matrix implied by the phylogeny for use with BASS-GWAS.

The `vars2patts.csv` file maps between the original variant identifiers
and the deduplicated variants that form `patterns_centred.csv`:

        vroom::vroom("gwas_inputs/vars2patts.csv",
        altrep=F, progress=F, show_col_types=F) %>%   
        print(unitigs, n = 4, max_extra_cols = 0,width=60)

    ## # A tibble: 26,286 × 3
    ##   pattern_id variant                         direction
    ##        <dbl> <chr>                               <dbl>
    ## 1          1 CATACCGGCGGGGGGGGGGGGGGGGGGGCGC         1
    ## 2          2 CTTCAAGCAACGTAAACAAAGAAATCCAAGA         1
    ## 3          3 ACTTCTCTTCTCTTCTCTTCTCTTCCCTTCC         1
    ## 4          4 CTAGATTCCCGCTTTCGCGGGAATGACGAAA         1
    ## # ℹ 26,282 more rows

-   `pattern_id` contains the column index in `patterns_centred.csv`.

-   `variant` contains the original variant identifier.

-   `direction` is equal to
    −1
    if the original variant was complemented as
    1 − *v*
    .

The `patterns.csv` file is a `0/1` version of the pattern
presence-absence file.

## Observations file

The observation file should be a table in the `.csv` format with two
columns and a header. The first column must contain strain identifiers.
The second column must contain binary phenotype entries (either `0/1` or
`TRUE/FALSE`). An example is shown below:

<table>
<thead>
<tr>
<th>strain</th>
<th>phenotype</th>
</tr>
</thead>
<tbody>
<tr>
<td>isolate_1</td>
<td>0</td>
</tr>
<tr>
<td>isolate_2</td>
<td>0</td>
</tr>
<tr>
<td>isolate_3</td>
<td>1</td>
</tr>
<tr>
<td>isolate_4</td>
<td>0</td>
</tr>
<tr>
<td>isolate_10</td>
<td>1</td>
</tr>
<tr>
<td>…</td>
<td>…</td>
</tr>
<tr>
<td>isolate_n</td>
<td>0</td>
</tr>
</tbody>
</table>

The toy dataset contains an observation file in the correct format:

    vroom::vroom("../sample_data/data_full.csv",
        altrep=F, progress=F, show_col_types=F) %>%   
        print(n = 5, max_extra_cols = 0,width=60)

    ## # A tibble: 800 × 2
    ##   wgs_id      cip  
    ##   <chr>       <lgl>
    ## 1 SRR11698258 FALSE
    ## 2 SRR16644552 FALSE
    ## 3 SRR11698444 FALSE
    ## 4 SRR11699051 FALSE
    ## 5 SRR11699151 FALSE
    ## # ℹ 795 more rows

# GWAS

## `bassgwass-run.jl` usage

    bassgwas-run.jl --pfile PFILE --ufile UFILE --sfile SFILE
                    --obsfile OBSFILE --odir ODIR --batchsz BATCHSZ
                    [--inclnext INCLNEXT] [--seed SEED]
                    [--ndraws NDRAWS] [--nthin NTHIN]
                    [--nchains NCHAINS] [--p_mean P_MEAN]
                    [--pr_0 PR_0] [--randdes] [-h]

-   `--pfile` `patterns_centred.csv` variant file (generated by
    preprocessing script)

-   `--ufile` `U.csv` left singular vectors for population structure
    correction (generated by preprocessing script)

-   `--sfile` `S.csv` singular values for population structure
    correction (generated by preprocessing script)

-   `--obsfile` observations file

-   `--odir` output directory

-   `--batchsz` batch size for experimental design. If set to 0,
    experimental design will be disabled

-   `--inclnext` path to a file listing isolate identifiers to include
    in the next experimental designs, one per line. If omitted, all
    isolates for next experimental design will be selected
    algorithmically. See [Additional
    functionality](#Additional-functionality)

-   `--seed` random seed. Default: `0`

-   `--n_draws` number of draws per chain. Default: `1000`

-   `--n_thin` thinning factor per draw. Default: `500`

-   `--n_chains` number of parallel chains to run. Default: `4`

-   `--p_mean` prior mean number of causal variants conditional on at
    least one effect. Default: `5`

-   `--pr_0` prior probability of 0 effects. Default: `0.25`

-   `--randdes` flag indicated a random design should be returned rather
    than a design based on expected information gained. You probably
    don’t want to use this.

-   `-h --help` print help and exit.

With the inputs ready, we can now run the GWAS model. BASS-GWAS uses
efficient MCMC importance sampling and will create multiple files with
MCMC draws when finished, together with a file specifying a batch of
isolates for subsequent phenotyping if requested.

**NOTE the generated files are large. For 100k variants, 4000 draws
require approximately 60GB of storage**

## Running GWAS on a toy dataset

We will now demonstrate GWAS functionality on the toy dataset. This
dataset was constructed using data from (*Reimche 2023*). The trait in
this dataset is high-level (
 ≥ 1*μ**g* *m**L*<sup>−1</sup>
) ciprofloxacin resistance in *N. gonorrhoeae*. To attain this level of
resistance, at least one substitution in each of *gyrA* and *parC* genes
is known to be required.

The polygenic nature of this trait makes it challenging for GWAS with
binary phenotypes. To speed things up, we will downsample the full
dataset to 200 isolates.

    dat_small <- read_csv("../sample_data/data_full.csv", progress=F, show_col_types=F) %>% 
        slice_sample(n = 200)
    dat_small %>% group_by(cip) %>% 
        summarise(count=n())

    ## # A tibble: 2 × 2
    ##   cip   count
    ##   <lgl> <int>
    ## 1 FALSE   179
    ## 2 TRUE     21

    dat_small %>% write_csv("gwas_inputs/data200.csv")

We’re now ready to run BASS-GWAS sampling. We will run 2 chains in
parallel each for a total of
250 × 500
iterations. Moreover we will parallelize each chain across 2 CPU cores
using `julia -t2`. As the toy dataset only contains 20000 variants, this
should only take a few minutes.

    mkdir gwas_outputs
    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile gwas_inputs/data200.csv --odir gwas_outputs --batchsz 0 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=3.364711920865157
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Missing
    ## 
    ##            n    7.1630    4.6519    0.2682   303.0600   430.8142    1.0023       missing
    ##        sigma    1.1539    1.9776    0.0641   926.7644   945.1891    0.9996       missing
    ##            r    0.8467    1.4288    0.0574   633.3964   938.1596    1.0014       missing
    ##            c   12.4318   13.9325    0.5153   490.7548   903.9422    1.0014       missing
    ##        icept   -5.1459    2.2802    0.1441   262.3256   523.0442    1.0011       missing
    ##            Z    0.0698    0.0356    0.0012   838.9135   908.4580    1.0003       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.36017609834671 minutes."

When using BASS-GWAS with real data, using up to 4 threads per chain is
recommended. We recommend running chains for 500 × 1000 iterations and
running 4 chains in parallel.

## Identifying associations

### Background

BASS-GWAS uses a Bayesian sparse whole-genome regression model. The
combined effect
*ψ*
of all variants on the *j*-th strain is modelled as:

*ψ*<sub>*j*</sub> = *X*<sub>*j*</sub>**β** + *u*<sub>*j*</sub> + *μ*

Here
**β**
is a vector of variant effects, *u*<sub>*j*</sub> is a random effect
that accounts for population structure, and *μ* is an intercept.
Bayesian sparse models ensure that all but a few of the entries of
**β**
are zero, i.e. that all but a few variants have no effect.

To identify association we will use *posterior inclusion probabilties*
(PIPs). A PIP is the probability that a variant has a non-zero effect
given the observed data.

PIPs can be extended to *groups of variants*, for which the PIP is the
probability that the specified group contains *at least* one variant
with an effect given the observed data. This can be useful for example
if we group variants by genomic position, or by gene annotation.

We can interpret PIPs as the probability of a variant, or a group of
variants, containing a causal effect.

### Computing PIPs from BASS-GWAS outputs

BASS-GWAS saves the posterior draws into the `gwas_outputs` directory:

    ls gwas_outputs

    ## b_draws.csv
    ## gamma_draws.csv
    ## par_draws.csv
    ## pip_draws.csv
    ## receipt.txt
    ## u_draws.csv
    ## Y_draws.csv

-   `b_draws.csv` variant coefficient draws

-   `gamma_draws.csv` variant inclusion indicator draws

-   `par_draws.csv` hyperparameter, summary statistic, and importance
    weight draws

-   `pip_draws.csv` variant PIP draws

-   `u_draws.csv` lineage effect draws

-   `Y_draws.csv` latent utility draws for observed isolates

We will use the `comp_group_pips.R` script to identify associated loci
using user-defined variant groups.

`comp_group_pips.R` usage:

    Rapp ../helper_scripts/comp_group_pips.R --help

    ## Usage: comp_group_pips <GFILE> <OUT> <PAR-DRAWS> <GAMMAS-DRAWS> <VARS2PATTS>
    ## 
    ## compute variant group PIPs
    ## 
    ## Arguments:
    ##   <GFILE>         variant groups file. Must have variant column, and a group
    ##                   column
    ##   <OUT>           output file name
    ##   <PAR-DRAWS>     par_draws file
    ##   <GAMMAS-DRAWS>  gammas_draws file
    ##   <VARS2PATTS>    vars2patts file

`comp_group_pips.R` computes PIPs for groups of variants specified in
the variant group file. The variant group file should list group-variant
pairs:

<table>
<thead>
<tr>
<th>variant</th>
<th>group</th>
</tr>
</thead>
<tbody>
<tr>
<td>variant_1</td>
<td>group1</td>
</tr>
<tr>
<td>variant_1</td>
<td>group2</td>
</tr>
<tr>
<td>variant_2</td>
<td>group4</td>
</tr>
<tr>
<td>variant_3</td>
<td>group2</td>
</tr>
<tr>
<td>variant_3</td>
<td>group3</td>
</tr>
<tr>
<td>…</td>
<td>…</td>
</tr>
<tr>
<td>variant_n</td>
<td>group_k</td>
</tr>
</tbody>
</table>

The toy dataset contains two variant grouping files:

-   `locus_groups.R` groups unitigs by annotations of the loci they map
    to.
-   `position_groups.R` groups unitigs by genomic position, using 250bp
    overlapping windows spaced 125bp apart.

<!-- -->

    vroom::vroom("../sample_data/position_groups.csv",
        altrep=F, progress=F, show_col_types=F) %>%   
        print(n = 5, max_extra_cols = 0, width=60)

    ## # A tibble: 58,479 × 2
    ##   variant                           group
    ##   <chr>                             <dbl>
    ## 1 CTTCAAGCAACGTAAACAAAGAAATCCAAGA 1727250
    ## 2 CTTCAAGCAACGTAAACAAAGAAATCCAAGA 1727375
    ## 3 CTTCAAGCAACGTAAACAAAGAAATCCAAGA 1729625
    ## 4 CTTCAAGCAACGTAAACAAAGAAATCCAAGA 1729750
    ## 5 CTTCAAGCAACGTAAACAAAGAAATCCAAGA 1735375
    ## # ℹ 58,474 more rows

### Using Manhattan plots

We can use grouping by genomic position, together with the posterior
draws, to calculate the PIPs for individual genomic position windows
along the chromosome.

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv gwas_outputs/position_pips.csv gwas_outputs/par_draws.csv gwas_outputs/gamma_draws.csv gwas_inputs/vars2patts.csv

In turn, position PIPs can be visualised as a Manhattan plot:

    parC_begin <- 194768
    parC_end <- 197071
    parC_pos <- parC_begin+(parC_end-parC_begin)/2

    gyrA_begin <- 996474
    gyrA_end <- 999224
    gyrA_pos <- gyrA_begin+(gyrA_end-gyrA_begin)/2

    read_csv("gwas_outputs/position_pips.csv", progress=F, show_col_types=F) %>% 
        ggplot(aes(x=group/1e3, y=PIP)) + 
            geom_col(fill="gray25",color="gray25") +
            geom_vline(data=data.frame(x=c(parC_pos,gyrA_pos)/1e3), aes(xintercept=x), 
                color="darkred", linetype="dashed", alpha=.5, linewidth=1) +
            ylim(0,1)+
            labs(y="PIP", x="Position (KB)")

![](tutorial_files/figure-markdown_strict/unnamed-chunk-14-1.png)

### Using variant annotations

Alternatively, we can use the locus grouping to identify associated
annotated loci while limiting reference bias by mapping each unitig to
the annotated de novo assembly from its own parent isolate.

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/locus_groups.csv gwas_outputs/locus_pips.csv gwas_outputs/par_draws.csv gwas_outputs/gamma_draws.csv gwas_inputs/vars2patts.csv

    read_csv("gwas_outputs/locus_pips.csv", progress=F, show_col_types=F) %>% 
        arrange(-PIP) %>% 
        print(n = 5, max_extra_cols = 0, width=60)

    ## # A tibble: 3,600 × 2
    ##   group       PIP
    ##   <chr>     <dbl>
    ## 1 gyrA     0.895 
    ## 2 parC     0.302 
    ## 3 tbpB_2   0.0994
    ## 4 WHO_N.55 0.0972
    ## 5 piiC_3   0.0926
    ## # ℹ 3,595 more rows

Based on these two summaries we can see that there are two peaks at the
*gyrA* and *parC* loci, substitutions in both of which are required to
attain high-level ciprofloxacin resistance.

While *gyrA* was identified with a relatively high level of confidence
(PIP
 &gt; 0.7
), the confidence for the *parC* hit remains low (PIP
 &lt; 0.45
). This is a consequence of random sampling.

Relying on hits with low PIPs can still lead to successful
identification of causative variants but it will increase false positive
rates, in turn leading to failed validation experiments.

Moreover, with PIPs this low we cannot discern that **both** *gyrA* and
*parC* are required.

### Refining associated variants

We can further *fine-map* the *gyrA* hit to identify which variants are
most likely to be causal. First we will compute the per-unitig PIPs.

    Rapp ../helper_scripts/comp_variant_pips.R gwas_outputs/variant_pips.csv gwas_outputs/par_draws.csv gwas_outputs/pip_draws.csv gwas_inputs/vars2patts.csv

Then we will use the variant annotation file to identify variants
mapping to *gyrA* and display the top 5.

    parC_hits <- read_csv("../sample_data/annotated_hits.csv", progress=F, show_col_types=F) %>% 
        filter(locus=="gyrA") %>%
        left_join(
            read_csv("gwas_outputs/variant_pips.csv", progress=F, show_col_types=F),
            by="variant")
            
    parC_hits %>% 
        arrange(-PIP) %>% 
        print(n = 10, max_extra_cols = 0,width=60)

    ## # A tibble: 217 × 8
    ##    variant     contig locus   down     up pattern_id     PIP
    ##    <chr>       <chr>  <chr>  <dbl>  <dbl>      <dbl>   <dbl>
    ##  1 TCATCGGTAA… WHO_N  gyrA  996715 996748       5753 0.396  
    ##  2 TCATCGGTAA… WHO_N  gyrA  996715 996756       2055 0.386  
    ##  3 CGGTAAATAC… WHO_N  gyrA  996719 996756       8603 0.0688 
    ##  4 TACGACACCA… WHO_G  gyrA  995790 995822       4659 0.0210 
    ##  5 GACACCATCG… WHO_N  gyrA  996756 996787       1507 0.0110 
    ##  6 ACCACCCCCA… WHO_N  gyrA  996727 996775       2687 0.00606
    ##  7 CCTTCCCCGC… gnl|X… gyrA     200    231       6762 0.00426
    ##  8 AACCCTTCCC… gnl|X… gyrA     197    229       7377 0.00337
    ##  9 AATTTGCAGC… gnl|X… gyrA   44252  44290       9690 0.00181
    ## 10 ATTAACGGCA… gnl|X… gyrA      93    123       1068 0.00174
    ## # ℹ 207 more rows

# Adaptive experimental design

As we saw above, even with 200 samples chosen at random, we were not
able to identify *parC* with reasonable confidence. Instead, we will now
conduct a synthetic experiment using the adaptive sampling functionality

## Setup

We will begin by subsetting the 200 isolates we used in the previous
example to just 4, keeping at least one susceptible and one resistant
isolate.

    mkdir round0

    pheno_small <- dat_small$cip
    idx1 <- sample(which(pheno_small), 1) 
    idx0 <- sample(which(!pheno_small), 1) 
    rem <- sample(c(1:length(pheno_small))[-c(idx1,idx0)], 4-2,replace=F)

    to_keep <- c(idx0, idx1, rem)
    write_csv(dat_small[to_keep,], "round0/data.csv")

## Adaptive design

We will run BASS-GWAS adaptive design using a batch size of 16 for 4
rounds.

### Round 0

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round0/data.csv --odir round0 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=3.5294800314215062
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Missing
    ## 
    ##            n    7.2150    6.7510    0.2883    646.2945   509.6093    1.0098       missing
    ##        sigma    0.9838    0.9233    0.0274   1125.8978   836.7644    0.9983       missing
    ##            r    2.7430    5.6440    0.1789   1047.7588   915.9081    0.9989       missing
    ##            c    8.9786   11.3439    0.3578   1013.3785   808.4946    1.0023       missing
    ##        icept   -2.0196    2.5075    0.0778   1050.7900   975.3741    0.9989       missing
    ##            Z    0.0734    0.0316    0.0011    660.5971   554.6163    1.0038       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...
    ## Design objective value: 4.25132152967327
    ## Writing design ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.14785154660543 minutes."

### Round 1

Create an output directory for round 1 and append the entries for the
selected isolates

    mkdir round1
    cp round0/data.csv round1/data.csv
    grep -wf round0/design.txt ../sample_data/data_full.csv >> round1/data.csv

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round1/data.csv --odir round1 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=2.9206738241148793
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Missing
    ## 
    ##            n    6.4620    6.5737    0.3002    635.8612   563.6441    1.0024       missing
    ##        sigma    1.1854    2.5929    0.0867   1015.2237   938.5374    0.9996       missing
    ##            r    5.6091    8.6183    0.3051    822.4060   873.2982    0.9983       missing
    ##            c   12.3616   27.0914    0.8555    991.9257   984.6863    0.9983       missing
    ##        icept   -2.7995    2.3430    0.0824    857.2771   896.9564    0.9992       missing
    ##            Z    0.0803    0.0380    0.0013    776.1739   662.4620    1.0010       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...
    ## Design objective value: 4.012187140556623
    ## Writing design ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.12229596376419 minutes."

Compute position PIPs

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv round1/position_pips.csv round1/par_draws.csv round1/gamma_draws.csv gwas_inputs/vars2patts.csv

Manhattan plot for round 1:

    read_csv("round1/position_pips.csv", progress=F, show_col_types=F) %>% 
        ggplot(aes(x=group/1e3, y=PIP)) + 
            geom_col(fill="gray25",color="gray25") +
            geom_vline(data=data.frame(x=c(parC_pos,gyrA_pos)/1e3), aes(xintercept=x), 
                color="darkred", linetype="dashed", alpha=.5, linewidth=1) +
            ylim(0,1)+
            labs(y="PIP", x="Position (KB)")

![](tutorial_files/figure-markdown_strict/unnamed-chunk-29-1.png)

### Round 2

Create an output directory for round 2 and append the entries for the
selected isolates

    mkdir round2
    cp round1/data.csv round2/data.csv
    grep -wf round1/design.txt ../sample_data/data_full.csv >> round2/data.csv

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round2/data.csv --odir round2 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=2.9990586787385
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Missing
    ## 
    ##            n    6.2920    4.6982    0.1729    801.1437    660.3938    0.9994       missing
    ##        sigma    1.0996    1.5087    0.0488    954.3218    915.9004    1.0013       missing
    ##            r    1.1706    2.0789    0.0663    938.0808    984.8871    1.0015       missing
    ##            c   12.7951   15.7394    0.5086    912.9581    832.9100    1.0029       missing
    ##        icept   -3.7080    2.2189    0.0726   1001.7765    977.3993    0.9990       missing
    ##            Z    0.0774    0.0352    0.0012    912.9056   1026.6981    1.0000       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...
    ## Design objective value: 3.1106827540585407
    ## Writing design ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.09530641635259 minutes."

Compute position PIPs

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv round2/position_pips.csv round2/par_draws.csv round2/gamma_draws.csv gwas_inputs/vars2patts.csv

Manhattan plot for round 2:

    read_csv("round2/position_pips.csv", progress=F, show_col_types=F) %>% 
        ggplot(aes(x=group/1e3, y=PIP)) + 
            geom_col(fill="gray25",color="gray25") +
            geom_vline(data=data.frame(x=c(parC_pos,gyrA_pos)/1e3), aes(xintercept=x), 
                color="darkred", linetype="dashed", alpha=.5, linewidth=1) +
            ylim(0,1)+
            labs(y="PIP", x="Position (KB)")

![](tutorial_files/figure-markdown_strict/unnamed-chunk-35-1.png)

### Round 3

Create an output directory for round 3 and append the entries for the
selected isolates

    mkdir round3
    cp round2/data.csv round3/data.csv
    grep -wf round2/design.txt ../sample_data/data_full.csv >> round3/data.csv

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round3/data.csv --odir round3 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=3.9304697256891674
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Missing
    ## 
    ##            n    8.2330    5.3512    0.2545    547.7509   479.8402    1.0034       missing
    ##        sigma    1.2494    2.6663    0.0859    936.5412   898.6857    1.0012       missing
    ##            r    0.9001    1.6459    0.0536   1005.5432   940.8297    1.0013       missing
    ##            c   13.6990   14.3108    0.4554    899.9172   944.7027    0.9998       missing
    ##        icept   -6.1240    3.2857    0.1616    363.3375   797.3027    1.0032       missing
    ##            Z    0.0641    0.0283    0.0010    721.0292   986.4167    1.0006       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...
    ## Design objective value: 3.1092790995851622
    ## Writing design ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.17374399900436 minutes."

Compute position PIPs

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv round3/position_pips.csv round3/par_draws.csv round3/gamma_draws.csv gwas_inputs/vars2patts.csv

Manhattan plot for round 3:

    read_csv("round3/position_pips.csv", progress=F, show_col_types=F) %>% 
        ggplot(aes(x=group/1e3, y=PIP)) + 
            geom_col(fill="gray25",color="gray25") +
            geom_vline(data=data.frame(x=c(parC_pos,gyrA_pos)/1e3), aes(xintercept=x), 
                color="darkred", linetype="dashed", alpha=.5, linewidth=1) +
            ylim(0,1)+
            labs(y="PIP", x="Position (KB)")

![](tutorial_files/figure-markdown_strict/unnamed-chunk-41-1.png)

### Round 4

Create an output directory for round 4 and append the entries for the
selected isolates

    mkdir round4
    cp round3/data.csv round4/data.csv
    grep -wf round3/design.txt ../sample_data/data_full.csv >> round4/data.csv

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round4/data.csv --odir round4 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.13.3+0.aarch64.apple.darwin14. You currently have `1.13.0+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.13.3+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=3.5486680271107076
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse   ess_bulk    ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64    Float64     Float64   Float64       Missing
    ## 
    ##            n    7.0380    3.7157    0.1428   705.4269    715.2635    1.0004       missing
    ##        sigma    1.6449    3.9371    0.1237   911.1994    986.8508    1.0010       missing
    ##            r    0.8485    2.1559    0.0676   980.9906   1029.6174    1.0016       missing
    ##            c   16.3885   16.4174    0.5262   962.7673    875.4053    1.0041       missing
    ##        icept   -7.7263    3.2957    0.1158   810.6976    907.3478    1.0006       missing
    ##            Z    0.0664    0.0305    0.0011   796.8973    933.4946    1.0008       missing
    ## 
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...
    ## Design objective value: 1.4231022987344013
    ## Writing design ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 2.15266360044479 minutes."

Compute position PIPs

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv round4/position_pips.csv round4/par_draws.csv round4/gamma_draws.csv gwas_inputs/vars2patts.csv

Manhattan plot for round 4:

    read_csv("round4/position_pips.csv", progress=F, show_col_types=F) %>% 
        ggplot(aes(x=group/1e3, y=PIP)) + 
            geom_col(fill="gray25",color="gray25") +
            geom_vline(data=data.frame(x=c(parC_pos,gyrA_pos)/1e3), aes(xintercept=x), 
                color="darkred", linetype="dashed", alpha=.5, linewidth=1) +
            ylim(0,1)+
            labs(y="PIP", x="Position (KB)")

![](tutorial_files/figure-markdown_strict/unnamed-chunk-47-1.png) After
round 4, representing a total of 68 (64 + 4) isolates both causal loci,
parC and gyrA, were identified with high confidence (PIP
 &gt; 0.9
). This makes it clear that variants in both loci contribute to
high-level resistance. Further inspection of the predictor (beyond the
scope of this tutorial) reveals the relationship between variants with
high PIPs: in this case, that variants in both loci are required to
achieve high-level resistance.

Let’s see if the unitig-level PIPs are now more precise.

    Rapp ../helper_scripts/comp_variant_pips.R round4/variant_pips.csv round4/par_draws.csv round4/pip_draws.csv gwas_inputs/vars2patts.csv

Inspect *parC* variants

    read_csv("../sample_data/annotated_hits.csv", progress=F, show_col_types=F) %>% 
        filter(locus=="parC" ) %>%
        left_join(
            read_csv("round4/variant_pips.csv", progress=F, show_col_types=F),
            by="variant")%>% 
        arrange(-PIP) %>% 
        print(n = 5, max_extra_cols = 0,width=60)

    ## # A tibble: 137 × 8
    ##   variant      contig locus   down     up pattern_id     PIP
    ##   <chr>        <chr>  <chr>  <dbl>  <dbl>      <dbl>   <dbl>
    ## 1 ACCATCCGCAC… WHO_N  parC  1.97e5 1.97e5       5780 0.988  
    ## 2 TTTTGGGTAAA… WHO_N  parC  1.97e5 1.97e5       3554 0.0115 
    ## 3 GGCGGCGATGC… WHO_N  parC  1.97e5 1.97e5       1425 0.00582
    ## 4 ACGCAGCCGTG… WHO_N  parC  1.95e5 1.95e5       5229 0.00175
    ## 5 CAATAGCAATT… WHO_F  parC  1.31e6 1.31e6       7243 0.00172
    ## # ℹ 132 more rows

The variants mapping to *parC* are dominated by a single unitig mapping
to the QRDR region. This is in contrast to random sampling which failed
to identify any *parC* variants with high PIP.

Inspect *gyrA* variants

    read_csv("../sample_data/annotated_hits.csv", progress=F, show_col_types=F) %>% 
        filter(locus=="gyrA" ) %>%
        left_join(
            read_csv("round4/variant_pips.csv", progress=F, show_col_types=F),
            by="variant")%>% 
        arrange(-PIP) %>% 
        print(n = 5, max_extra_cols = 0,width=60)

    ## # A tibble: 217 × 8
    ##   variant      contig locus   down     up pattern_id     PIP
    ##   <chr>        <chr>  <chr>  <dbl>  <dbl>      <dbl>   <dbl>
    ## 1 TCATCGGTAAA… WHO_N  gyrA  996715 996756       2055 0.727  
    ## 2 TCATCGGTAAA… WHO_N  gyrA  996715 996748       5753 0.145  
    ## 3 CGGTAAATACC… WHO_N  gyrA  996719 996756       8603 0.121  
    ## 4 ACCACCCCCAC… WHO_N  gyrA  996727 996782       9393 0.00335
    ## 5 TCATCAAAGAA… WHO_N  gyrA  997687 997747       4666 0.00302
    ## # ℹ 212 more rows

The variants mapping to *gyrA* are dominated by a handful of strongly
associated variants all mapping to the QRDR region. The relatively low
PIPs for each of these individual variants reflects the unitig
representation for this highly diverse causal locus within *gyrA*, which
is split over several different unitigs.

In both cases adaptive design greatly improved efficiency over random
sampling.

## Additional functionality

BASS-GWAS provides additional functionality that may be useful in
real-world experimental applications.

### Re-using draws to redesign an experiment

`bassgwas-design.jl` allows for experimental design to be optimized
using a previous MCMC run.

    bassgwas-design.jl --pfile PFILE --ufile UFILE --sfile SFILE
                            --obsfile OBSFILE --b_draws B_DRAWS
                            --u_draws U_DRAWS --par_draws PAR_DRAWS
                            --ofile OFILE --batchsz BATCHSZ
                            [--inclnext INCLNEXT]

-   `--pfile` `patterns_centred.csv` variant file

-   `--ufile` `U.csv` left singular vectors for population structure
    correction

-   `--sfile` `S.csv` singular values for population structure
    correction

-   `--obsfile` observations file

-   `--b_draws` `b_draws.csv` variant coefficient draws

-   `--u_draws` `u_draws.csv` lineage effect draws

-   `--par_draws` `par_draws.csv` hyperparameter, summary statistic, and
    importance weight draws

-   `--ofile` output file name

-   `--batchsz` batch size for experimental design. If set to 0,
    experimental design will be disabled

-   `--inclnext` path to a file listing isolate identifiers to include
    in the next experimental designs, one per line

-   `-h --help` print help and exit.

### Including fixed isolates in design

The `--inclnext` argument can be uses to pass a text file listing
isolate identifiers to include in the design. One per line.

### Removing problematic isolates from the collection

During experimental design, problematic or non-viable isolates may be
encountered. To drop problematic isolates, we can generate new BASS-GWAS
inputs using the original variant and tree file, and pass a list of
isolate identifiers to exclude to `prepare_data.R`, one identifier per
line.

    prepare_data.R --to_drop identifiers_to_drop.txt

# Tips

-   Ensure that only high-quality isolates are included in experimental
    design collections. Mislabeled or contaminated isolates may severely
    compromise BASS-GWAS functionality.

-   Use experimental controls to reduce phenotyping noise.

-   A high proportion of experiments returning no phenotype may
    compromise experimental design efficiency.

-   Set the allele frequency filter to a reasonable value. This should
    be organism specific. A frequency of
    5/*N*
    or
    10/*N*
    where N is the collection size is a good starting value.

-   When applying BASS-GWAS to a new collection or a new organism, run
    synthetic experiments (as in this tutorial) using known
    genotype/phenotype relationships to ensure proper data preparation.

-   Carefully consider the batch-size for your experimental setup.
    Synthetic experiments can help probe batch-size efficiency
    trade-offs.

-   Check the tempered distribution diagnostics. The `ess_bulk` should
    ideally be more than 1000 for each of the parameters, and more than
    200 per chain. If the `ess_bulk` is low, increase `--n_thin` to a
    higher value.
