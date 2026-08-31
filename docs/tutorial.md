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

The isolate indentifiers must be shared between the variant file and the
phylogeny. If using unitigs,
[unitig-caller](https://github.com/bacpop/unitig-caller) can create this
file for you.

## Note on Input Types

Representing genetic variation in bacterial populations is an active
area of research. There are many distinct approaches, for example using
VCFs, unitigs, kmers, gene presence absence, etc. To ensure proper
functionality of BASS-GWAS, the representation of genetic variation
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
BASS-GWAS inputs. As a part of preprocessing, variant presence absence
patterns will be standardised, centred, and deduplicated. The phylogeny
is used to compute a factorised, low rank, similarity matrix.

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
the variance

    Rapp ../helper_scripts/prepare_data.R ../sample_data/variants.rtab ../sample_data/tree.tre --odir gwas_inputs --svd_threshold 0.99

This will generate the following input files in an gwas\_inputs
subdirectory.

The following two files are subsequently used with BASS-GWAS

-   `patterns_centred.csv` a centred, minorized, and deduplicated
    variant presence absence file for use with BASS-GWAS.

-   `U.csv` & `S.csv` a low-rank factorisation of the correlation matrix
    implied by the phylogeny for use with BASS-GWAS.

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
binary phenotypes.

We will run 2 chains in parallel each for a total of
250 × 500
iterations. Moreover we will parallelize each chain across 2 CPU cores
using `julia -t2`. To speed things up, we will downsample the full
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

We’re now ready to run BASSGWAS sampling. As the toy dataset only
contains 20000 variants, this should only take a few minutes.

    mkdir gwas_outputs
    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile gwas_inputs/data200.csv --odir gwas_outputs --batchsz 0 --ndraws 500 --nthin 250 --nchains 1

    ## mkdir: gwas_outputs: File exists
    ## Adding worker processes ...
    ## Done
    ## Precompiling packages...
    ##    1881.4 ms  ✓ BASSGWAS
    ##   1 dependency successfully precompiled in 2 seconds. 101 already precompiled.
    ##   73 dependencies precompiled but different versions are currently loaded (Accessors, Accessors → LinearAlgebraExt, Adapt, Adapt → AdaptSparseArraysExt, AliasTables, ArrayInterface, ArrayInterface → ArrayInterfaceFillArraysExt, ArrayInterface → ArrayInterfaceSparseArraysExt, Base64, CommonSolve, Compat, Compat → CompatLinearAlgebraExt, CompilerSupportLibraries_jll, CompositionsBase, CompositionsBase → CompositionsBaseInverseFunctionsExt, ConstructionBase, ConstructionBase → ConstructionBaseLinearAlgebraExt, DataAPI, DataStructures, Dates, Distributions, DocStringExtensions, FillArrays, FillArrays → FillArraysPDMatsExt, FillArrays → FillArraysSparseArraysExt, FillArrays → FillArraysStatisticsExt, Gamma, HypergeometricFunctions, InteractiveUtils, InverseFunctions, InverseFunctions → InverseFunctionsDatesExt, IrrationalConstants, JLLWrappers, JuliaSyntaxHighlighting, LogExpFunctions, LogExpFunctions → LogExpFunctionsInverseFunctionsExt, Logging, MacroTools, Markdown, Missings, OffsetArrays, OffsetArrays → OffsetArraysAdaptExt, OpenLibm_jll, OpenSpecFun_jll, OrderedCollections, PDMats, PDMats → StatsBaseExt, PrecompileTools, Preferences, Printf, PtrArrays, QuadGK, Reexport, Rmath, Rmath_jll, Roots, Serialization, SortingAlgorithms, SparseArrays, SpecialFunctions, Statistics, Statistics → SparseArraysExt, StatsAPI, StatsBase, StatsFuns, StatsFuns → StatsFunsInverseFunctionsExt, StyledStrings, SuiteSparse, SuiteSparse_jll, TOML, UUIDs, UnPack and Unicode). Restart julia to access the new versions. Otherwise, 24 dependents of these packages may trigger further precompilation to work with the unexpected versions.
    ## Adapting xi ...
    ## Found xi=4.748607283514716
    ## Running parallel chains ...
    ## Done
    ## Processing draws ...
    ## time upstate.up.sqx0 : 1.1003997678142489e-5
    ## time upstate.up.down2 : 8.478938991192937e-6
    ## time flip : 0.0004937890114080491
    ## time upstate.up.fx0 : 1.6392609047237676e-5
    ## time upstate.llr : 4.496530176140953e-5
    ## time betas : 0.00022541295831955018
    ## time upstate.up.qx0 : 0.0018901281887109577
    ## time upstate.up.ssq : 2.3229399519614267e-7
    ## time flip.remove : 0.0002965284148868683
    ## time ggibs.sc : 0.0005942066728415477
    ## time upstate.up : 0.0027483317074459557
    ## time gibbs : 0.0010373446218987757
    ## time nextstep : 0.003544230292153724
    ## time upstate : 0.002915139945636489
    ## time upstate.up.up2 : 0.0008061849481184921
    ## time upstate.rate : 7.723486725380345e-5
    ## time flip.add : 0.0006043073566905861
    ## Tempered distribution diagnostics:
    ## Summary Statistics
    ## 
    ##   parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   e ⋯
    ##       Symbol   Float64   Float64   Float64    Float64    Float64   Float64     ⋯
    ## 
    ##            n    6.9020    4.2209    0.3088   178.5867   266.3518    1.0134     ⋯
    ##        sigma    1.0376    1.0656    0.0539   421.4509   407.3927    1.0037     ⋯
    ##            r    0.8098    1.4799    0.1075   216.8281   309.2494    1.0119     ⋯
    ##            c   14.0406   15.5849    0.9285   233.6896   331.3647    1.0112     ⋯
    ##        icept   -5.2739    2.4683    0.2223   125.8603   231.6260    1.0239     ⋯
    ##            Z    0.0645    0.0294    0.0015   348.5206   461.6350    0.9988     ⋯
    ## 
    ##                                                                 1 column omitted
    ## Saving draws ...
    ## Done
    ## Designing next experiment ...

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 10.3994185487429 minutes."

When using BASSGWAS with real data using up to 4 threads per chain is
recommended. We recommend running chains for 500 × 1000 iterations.

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
    ## locus_pips.csv
    ## par_draws.csv
    ## pip_draws.csv
    ## position_pips.csv
    ## receipt.txt
    ## u_draws.csv
    ## variant_pips.csv
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

-   `locus_groups.R` which groups unitigs by annotations of the loci
    they map to
-   `position_groups.R` which groups unitigs by genomic position using
    250bp overlapping windows spaced 125bp apart.

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

We can use the position grouping, together with the posterior draws to
calculate the PIPs for individual genomic positions along the
chromosome.

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/position_groups.csv gwas_outputs/position_pips.csv gwas_outputs/par_draws.csv gwas_outputs/gamma_draws.csv gwas_inputs/vars2patts.csv

In turn, position PIPs can be visualised as a manhattan plot:

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

Alternatively we can use the locus grouping to identify associated loci
while limiting reference bias

    Rapp ../helper_scripts/comp_group_pips.R ../sample_data/locus_groups.csv gwas_outputs/locus_pips.csv gwas_outputs/par_draws.csv gwas_outputs/gamma_draws.csv gwas_inputs/vars2patts.csv

    read_csv("gwas_outputs/locus_pips.csv", progress=F, show_col_types=F) %>% 
        arrange(-PIP) %>% 
        print(n = 5, max_extra_cols = 0, width=60)

    ## # A tibble: 3,600 × 2
    ##   group               PIP
    ##   <chr>             <dbl>
    ## 1 gyrA              0.676
    ## 2 parC              0.511
    ## 3 SRR16999462_00492 0.175
    ## 4 piiC_3            0.122
    ## 5 WHO_N.55          0.119
    ## # ℹ 3,595 more rows

Based on these two summaries we can see that there are two peaks at the
*gyrA* and *parC* loci, substitutions in both of which are required to
attain high-level ciprofloxacin resistance.

While *gyrA* was identified with a relatively high-level of confidence
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
    ##  1 TCATCGGTAA… WHO_N  gyrA  996715 996756       2055 0.274  
    ##  2 TCATCGGTAA… WHO_N  gyrA  996715 996748       5753 0.258  
    ##  3 ACCACCCCCA… WHO_N  gyrA  996727 996775       2687 0.0410 
    ##  4 CGGTAAATAC… WHO_N  gyrA  996719 996756       8603 0.0363 
    ##  5 GACACCATCG… WHO_N  gyrA  996756 996787       1507 0.0266 
    ##  6 TACGACACCA… WHO_G  gyrA  995790 995822       4659 0.0248 
    ##  7 AAAAATACAA… gnl|X… gyrA     139    169       2216 0.0106 
    ##  8 TAACAGGCGC… WHO_N  gyrA  997475 997535       9351 0.00637
    ##  9 CTATATATAG… WHO_N  gyrA  999299 999332        149 0.00347
    ## 10 TTCAAAGGAA… gnl|X… gyrA    5758   5790       9112 0.00266
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

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 4.14689381917318e-05 minutes."

### Round 1

Create an output directory for round 1 and append the entries for the
selected isolates

    mkdir round1
    cp round0/data.csv round1/data.csv
    grep -wf round0/design.txt ../sample_data/data_full.csv >> round1/data.csv

Run BASS-GWAS

    julia -t2 ../bassgwas-run.jl --pfile gwas_inputs/patterns_centred.csv --ufile gwas_inputs/U.csv --sfile gwas_inputs/S.csv --obsfile round1/data.csv --odir round1 --batchsz 16 --ndraws 500 --nthin 250 --nchains 2

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 3.74158223470052e-05 minutes."

Compute position PIPS

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

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 4.01139259338379e-05 minutes."

Compute position PIPS

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

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 4.18980916341146e-05 minutes."

Compute position PIPS

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

    paste0("Run time: ", difftime(Sys.time(),tbegin,units='mins'), " minutes.")

    ## [1] "Run time: 3.93827756245931e-05 minutes."

Compute position PIPS

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
*parC* and *gyrA*, were identified with high confidence (PIP
 &gt; 0.9
). This makes it clear that variants in both loci are required to
achieve high-level resistance.

Let’s see if the the unitig-level PIPs are now more precise.

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
    ## 1 ACCATCCGCAC… WHO_N  parC  196786 196830       5780 0.986  
    ## 2 TTTTGGGTAAA… WHO_N  parC  196801 196842       3554 0.00676
    ## 3 GGCGGCGATGC… WHO_N  parC  196659 196709       1425 0.00212
    ## 4 GCCGTGCCTGA… WHO_N  parC  197135 197165       2133 0.00167
    ## 5 GGTGATGACCG… WHO_N  parC  195082 195134       1281 0.00156
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
    ## 1 TCATCGGTAAA… WHO_N  gyrA  996715 996756       2055 0.656  
    ## 2 TCATCGGTAAA… WHO_N  gyrA  996715 996748       5753 0.162  
    ## 3 CGGTAAATACC… WHO_N  gyrA  996719 996756       8603 0.155  
    ## 4 ACCACCCCCAC… WHO_N  gyrA  996727 996782       9393 0.0127 
    ## 5 GACACCATCGT… WHO_N  gyrA  996756 996787       1507 0.00920
    ## # ℹ 212 more rows

The variants mapping to *gyrA* are dominated by a handful of strongly
associated variants all mapping to the QRDR region. The relatively low
PIPs for each of these individual variants reflects the unitig
representation for this highly diverse causal locus within gyrA, which
is split over several different unitigs.

In both cases adaptive design greatly improved efficiency over random
sampling.

## Additional functionality

BASS-GWAS provides additional functionality that may be useful in
real-world experimental applications

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

-   Ensure that only high quality isolates are included in experimental
    design collections. Mislabeled or contaminated isolates may severely
    compromise BASS-GWAS functionality.

-   Use experimental controls to reduce phenotyping noise

-   Set the allele frequency filter to a reasonable value. This should
    be organism specific. A frequency of
    5/*N*
    or
    10/*N*
    where N is the collection size is a good starting value.

-   When applying BASS-GWAS to a new collection or a new organism, run
    synthetic experiments using known genotype/phenotype relationships
    to ensure proper data preparation.

-   Check the tempered distribution diagnostics. The `ess_bulk` should
    ideally be more than 1000 for each of the parameters, and more than
    200 per chain. If the `ess_bulk` is low, increase `--n_thin` to a
    higher value.
