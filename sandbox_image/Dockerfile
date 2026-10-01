# BixBench's notebook environment, rebuilt for amd64.
#
# This is data-analysis-crow v1.5.0's src/fhda/Dockerfile.pinned -- the environment
# BixBench's published agents ran their notebooks in (BixBench pins fhda @ v1.5.0) --
# verbatim but for one line. The published image, futurehouse/bixbench:aviary-notebook-env,
# exists only for arm64, so it is rebuilt here from its own pinned recipe. The one line
# removed installs the agent framework (ldp, fhaviary, futurehouse-client); nothing in
# the notebook kernel imports it, and our agent loop runs outside the container.
# Three more changes, each marked where it is made: kernel_requirements.txt's exact
# pins are installed with pip because the mamba solve no longer terminates; one
# library link that today's solve breaks is restored; a cleanup step that only
# saves disk space is allowed to fail.

# DANGER: Beware of changing this dockerfile, orchestrating the versioning in these R/python packages was very challenging
FROM continuumio/miniconda3:24.9.2-0

RUN mkdir /workspace && \
    mkdir /envs
WORKDIR /envs

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        wget \
        gpg \
        software-properties-common \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN conda install mamba=2.0.5 -c conda-forge -y

# Install R packages from conda-forge
RUN mamba install -c conda-forge -y \
    r-base=4.3.3 \
    r-recommended=4.3 \
    r-irkernel=1.3.2 \
    r-factominer=2.11 \
    r-rcolorbrewer=1.1_3 \
    r-devtools=2.4.5 \
    r-broom=1.0.7 \
    r-data.table=1.15.4 \
    r-enrichr=3.2 \
    r-factoextra=1.0.7 \
    r-ggnewscale=0.5.0 \
    r-ggrepel=0.9.6 \
    r-ggpubr=0.6.0 \
    r-ggvenn=0.1.10 \
    r-janitor=2.2.1 \
    r-multcomp=1.4_26 \
    r-matrix=1.6_5 \
    r-pheatmap=1.0.12 \
    r-tidyverse=2.0.0 \
    r-readxl=1.4.3 \
    r-reshape=0.8.9 \
    r-rstatix=0.7.2 \
    r-viridis=0.6.5 \
    udocker=1.3.17 \
    imbalanced-learn=0.13.0 \
    ipykernel=6.29.5 \
    sqlite=3.47.2

RUN python -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"
RUN R -e 'IRkernel::installspec(name = "R", displayname = "R (4.3.3)")'

# I separate these because not all packages need both channels, additionally,
# creating multiple layers makes caching easier
RUN mamba install -c conda-forge -c bioconda -y \
    biokit=0.5.0 \
    gseapy=1.1.4 \
    blast=2.16.0 \
    clipkit=2.3.0 \
    fastqc=0.12.1 \
    iqtree=2.3.6 \
    mafft=7.526 \
    metaeuk=7.bba0d80 \
    mygene=3.2.2 \
    perl=5.32.1 \
    phykit=2.0.1 \
    pydeseq2=0.4.12 \
    spades=4.0.0 \
    trim-galore=0.6.10 \
    bioconductor-enhancedvolcano=1.20.0 \
    bioconductor-deseq2=1.42.0 \
    bioconductor-clusterprofiler=4.10.0 \
    bioconductor-org.hs.eg.db=3.18.0 \
    bioconductor-genomicranges=1.54.1 \
    bioconductor-summarizedexperiment=1.32.0 \
    bioconductor-apeglm=1.24.0


COPY kernel_requirements.txt .

# Install conda packages first
# (Upstream: RUN mamba install -c conda-forge --file kernel_requirements.txt -y.
#  Against today's conda-forge that solve ran 45 minutes at full CPU without
#  finishing, so the same pinned list is installed with pip -- identical versions,
#  a different installer. See the header.)
RUN pip install --no-cache-dir -r kernel_requirements.txt

# (Not upstream. Solved today, the bioconda layer downgrades `sqlite` to a 2020
#  build whose libsqlite3.so.0 symlink shadows conda-forge's `libsqlite`, and
#  Python 3.12's sqlite3 -- which rpy2 imports -- then fails to load. Reinstalling
#  libsqlite alone restores the link; nothing else changes.)
RUN mamba install -y -c conda-forge --no-deps --force-reinstall libsqlite=3.53.4 && \
    python -c "import sqlite3, rpy2.robjects"

# (the agent-framework pip install is omitted; see the header)

# Certain tools are not easily installable via conda. A common practice for
# bioinformaticians is to use udocker to run certain heavy duty omics processing
# tools in an isolated environment
# RUN udocker --allow-root install && \
#     udocker --allow-root pull ezlabgva/busco:v5.8.0_cv1

WORKDIR /workspace

# (Upstream chains these with &&. After the pip layer, conda itself errors on
#  `clean`, which only saves disk space, so its failure is tolerated here.)
RUN mamba clean --all -f -y; \
    conda clean --all -f -y || true; \
    rm -rf /root/.cache/pip
