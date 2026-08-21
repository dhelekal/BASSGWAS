-   [Introduction](#introduction)
-   [Input Data](#input-data)
    -   [Note on Input Types](#note-on-input-types)
    -   [Inspecting the input data](#inspecting-the-input-data)
    -   [Preparing the inputs](#preparing-the-inputs)
    -   [GWAS](#gwas)
        -   [`bassgwass-run.jl` usage](#bassgwass-run.jl-usage)
        -   [](#section)
        -   [Running GWAS on the toy
            dataset](#running-gwas-on-the-toy-dataset)
        -   [Inspecting](#inspecting)
    -   [Adaptive experimental design](#adaptive-experimental-design)

First we load the necessary libraries used in this tutorial

    library(ape)
    library(ggtree)
    library(tidyverse)
    library(ggplot2)

    set.seed(1)

    rm -r round1
    rm -r round2
    rm -r round3
    rm -r gwas_inputs
    rm -r gwas_outputs

    ## rm: round1: No such file or directory
    ## rm: round2: No such file or directory
    ## rm: round3: No such file or directory

# Introduction

This tutorial assumes that you have previously downloaded the helper &
launch scripts in this repository. The first section of the tutorial
covers the general usage of the GWAS model. The second section covers
adaptive experimental design.

Throughout this tutorial we will be using the toy data found in the
`sample_data` folder. This is a version of the ciprofloxacin resistance
benchmark datasets used in the manuscript modified to contain only
20,000 unique variants. The toy dataset uses *unitigs* to characterise
genetic variation.

# Input Data

Using BASSGWAS requires two main data files: a variant presence-absence
file, and a phylogeny file in the newick formati. Both the phylogeny and
variant presence-absence should be constructed for the entire collection
at hand. The first column of the variant file should contain unique
variant identifiers. The remaining columns should each correspond to an
isolate, and the first row contain unique isolate identifiers:

<table>
<thead>
<tr>
<th>variant_id</th>
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

The isolate indentifiers must be shared between the variant file and the
phylogeny.

## Note on Input Types

Representing genetic variation in bacterial populations is an active
area of research. There are many distinct approaches, for example using
VCFs, unitigs, kmers, gene presence absence, etc. To ensure proper
functionality of BASSGWAS, the representation of genetic variation
should be both *comprehensive*, and *precise*. This is especially
important when using adaptive sequential sampling. Unitigs are usually a
good choice.

**DO NOT use adaptive sequential sampling with highly redundant
representations such as kmers**

**DO NOT use adaptive sequential sampling with gene presence absence
data alone**

## Inspecting the input data

The variant file in the toy dataset uses unitigs as variant identifiers
and `sra` accession numbers as isolate identifiers.

    vroom::vroom("../sample_data/variants.rtab",
        altrep=F, progress=F, show_col_types = FALSE) %>%   
        print(unitigs, n = 5, max_extra_cols = 0,width=60)

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

To generate the input files needed for use BASSGWAS, we will need to use
the data preprocessing script. First we will create a directory for
storing the generated input files.

    mkdir gwas_inputs

Next we will use the data preprocessing script `prepare_data.R` to
process the variant presence-absence file and the phylogeny into
BASSGWAS inputs. As a part of preprocessing, variant presence absence
patterns will be standardised, centred, and deduplicated. The phylogeny
is used to compute a factorised, low rank, similarity matrix.

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

The `--af-threshold` can be used to set an optional allele frequency
filter.

We will now use the script to preprocess the toy data.

    Rapp ../helper_scripts/prepare_data.R ../sample_data/variants.rtab ../sample_data/tree.tre --odir gwas_inputs

This generated input files in the `gwas_inputs` directory. The following
two files are subsequently used with BASSGWASS:

-   `patterns_centred.csv` a centred, minorized, and deduplicated
    variant presence absence file for use with BASSGWAS.

-   `U.csv` & `S.csv` a low factorisation of the correlation matrix
    implied by the phylogeny for use with BASSGWAS.

The `vars2patts.csv` file maps between the original variant identifiers
and the deduplicated variants that form `patterns_centred.csv`:

    vroom::vroom("gwas_inputs/vars2patts.csv",
        altrep=F, progress=F, show_col_types = FALSE) %>%   
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

-   `direction` is equal to −1 if the original variant was complemented
    as 1 − *v*.

Additionally an uncentred, binary version of the pattern file was
generated as `patterns.csv`.

## GWAS

### `bassgwass-run.jl` usage

    bassgwas-run.jl --pfile PFILE --ufile UFILE --sfile SFILE
                    --obsfile OBSFILE --odir ODIR --batchsz BATCHSZ
                    [--inclnext INCLNEXT] [--seed SEED]
                    [--ndraws NDRAWS] [--nthin NTHIN]
                    [--nchains NCHAINS] [--p_mean P_MEAN]
                    [--pr_0 PR_0] [--randdes] [-h]

### 

-   `--pfile` `patterns_centred.csv` variant file

-   `--ufile` `U.csv` left singular vectors for population structure
    correction

-   `--sfile` `S.csv` singular values for population structure
    correction

-   `--odir` output directory

-   `--batchsz` batch size for experimental design. If set to 0,
    experimental design will be disabled

-   `--inclnext` path to a file listing isolate identifiers to include
    in the next experimental designs, one per line

-   `--seed` random seed. Default: `0`

-   `--n_draws` number of draws per chain. Default: `1000`

-   `--n_thin` thinning factor per draw. Default: `500`

-   `--n_chains` number of parallel chains to run. Default: `4`

-   `--p_mean` prior mean number of causal variants conditional on at
    least one effect. Default: `5`

-   `--pr_0` prior probability of 0 effects. Default: `0.25`

-   `--randdes` flag indicated a random design should be returned. You
    probably don’t want to use this.

-   `-h --help` print help and exit.

With the inputs ready, we can now run the GWAS model. BASSGWAS uses
efficient MCMC sampling and will create multiple files with MCMC draws
when finished, along with a file specifying a batch of isolates for
subsequent phenotyping if requested.

**NOTE the generated files are large. For 100k variants, 4000 draws
require approximately 60GB of storage**

### Running GWAS on the toy dataset

We will now demonstrate GWAS functionality on the toy dataset. We will
run 2 chains in parallel each for a total of 250 × 500 iterations.
Moreover we will parallelize each chain across 2 CPU cores by setting
`JULIA_NUM_THREADS=2`. To speed things up, we will downsample the full
dataset to 200 isolates.

    dat_small <- read_csv("../sample_data/data_bin.csv") %>% 
        slice_sample(n = 200)
    dat_small %>% group_by(cip) %>% 
        summarise(count=n())

    ## # A tibble: 2 × 2
    ##   cip   count
    ##   <lgl> <int>
    ## 1 FALSE   179
    ## 2 TRUE     21

    dat_small %>% write_csv("gwas_inputs/data200.csv")

We’re now ready to run BASSGWAS. As the toy dataset only contains 10000
variants, this should only take a few minutes.

    mkdir gwas_outputs
    export JULIA_NUM_THREADS=2
    julia ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile gwas_inputs/data200.csv --odir gwas_outputs --batchsz 0 --ndraws 500 --nthin 250 --nchains 2

    ## The latest version of Julia in the `release` channel is 1.12.7+0.aarch64.apple.darwin14. You currently have `1.11.6+0.aarch64.apple.darwin14` installed. Run:
    ## 
    ##   juliaup update
    ## 
    ## in your terminal shell to install Julia 1.12.7+0.aarch64.apple.darwin14 and update the `release` channel to that version.
    ## Adding worker processes ...
    ## Done
    ## Adapting xi ...
    ## Found xi=4.7486072835145885
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## Summary Statistics
    ##   parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec
    ##       Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Missing
    ## 
    ##            n    7.0030    4.4320    0.2233   402.7138   648.5411    1.0038       missing
    ##        sigma    1.0907    1.4596    0.0480   965.4971   896.8256    1.0025       missing
    ##            r    0.8447    1.4355    0.0606   649.4256   749.4683    1.0037       missing
    ##            c   13.7560   18.8572    0.7545   497.6810   679.2789    1.0052       missing
    ##        icept   -5.2629    2.5462    0.1339   383.5858   459.3622    1.0075       missing
    ##            Z    0.0646    0.0295    0.0010   702.6339   924.1404    0.9987       missing
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...

### Inspecting

When using BASSGWAS with real data using up to 4 threads per chain is
recommended.

## Adaptive experimental design
