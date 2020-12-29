# Generated by: Neurodocker version 0.7.0
# Latest release: Neurodocker version 0.7.0
# Timestamp: 2020/12/29 15:21:22 UTC
# 
# Thank you for using Neurodocker. If you discover any issues
# or ways to improve this software, please submit an issue or
# pull request on our GitHub repository:
# 
#     https://github.com/ReproNim/neurodocker

Bootstrap: docker
From: nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

%post
su - root

export ND_ENTRYPOINT="/neurodocker/startup.sh"
apt-get update -qq
apt-get install -y -q --no-install-recommends \
    apt-utils \
    bzip2 \
    ca-certificates \
    curl \
    locales \
    unzip
apt-get clean
rm -rf /var/lib/apt/lists/*
sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
dpkg-reconfigure --frontend=noninteractive locales
update-locale LANG="en_US.UTF-8"
chmod 777 /opt && chmod a+s /opt
mkdir -p /neurodocker
if [ ! -f "$ND_ENTRYPOINT" ]; then
  echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT"
  echo 'set -e' >> "$ND_ENTRYPOINT"
  echo 'export USER="${USER:=`whoami`}"' >> "$ND_ENTRYPOINT"
  echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT";
fi
chmod -R 777 /neurodocker && chmod a+s /neurodocker

export PATH="/opt/miniconda-latest/bin:$PATH"
echo "Downloading Miniconda installer ..."
conda_installer="/tmp/miniconda.sh"
curl -fsSL --retry 5 -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash "$conda_installer" -b -p /opt/miniconda-latest
rm -f "$conda_installer"
conda update -yq -nbase conda
conda config --system --prepend channels conda-forge
conda config --system --set auto_update_conda false
conda config --system --set show_channel_urls true
sync && conda clean -y --all && sync
conda create -y -q --name fit
conda install -y -q --name fit \
    "python=3.7" \
    "astra-toolbox" \
    "-c" \
    "astra-toolbox/label/dev" \
    "pytorch" \
    "torchvision" \
    "torchaudio" \
    "cudatoolkit=10.2" \
    "-c" \
    "pytorch"
sync && conda clean -y --all && sync
bash -c "source activate fit
  pip install --no-cache-dir  \
      "/fourier_image_transformers-0.1.0-py3-none-any.whl""
rm -rf ~/.cache/pip/*
sync
sed -i '$isource activate fit' $ND_ENTRYPOINT


echo '{
\n  "pkg_manager": "apt",
\n  "instructions": [
\n    [
\n      "base",
\n      "nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
\n    ],
\n    [
\n      "user",
\n      "root"
\n    ],
\n    [
\n      "_header",
\n      {
\n        "version": "generic",
\n        "method": "custom"
\n      }
\n    ],
\n    [
\n      "copy",
\n      [
\n        "/home/tibuch/Gitrepos/FourierImageTransformer/dist/fourier_image_transformers-0.1.0-py3-none-any.whl",
\n        "/fourier_image_transformers-0.1.0-py3-none-any.whl"
\n      ]
\n    ],
\n    [
\n      "miniconda",
\n      {
\n        "create_env": "fit",
\n        "conda_install": [
\n          "python=3.7",
\n          "astra-toolbox",
\n          "-c",
\n          "astra-toolbox/label/dev",
\n          "pytorch",
\n          "torchvision",
\n          "torchaudio",
\n          "cudatoolkit=10.2",
\n          "-c",
\n          "pytorch"
\n        ],
\n        "pip_install": [
\n          "/fourier_image_transformers-0.1.0-py3-none-any.whl"
\n        ],
\n        "activate": true
\n      }
\n    ],
\n    [
\n      "entrypoint",
\n      "/neurodocker/startup.sh python"
\n    ]
\n  ]
\n}' > /neurodocker/neurodocker_specs.json

%environment
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export ND_ENTRYPOINT="/neurodocker/startup.sh"
export CONDA_DIR="/opt/miniconda-latest"
export PATH="/opt/miniconda-latest/bin:$PATH"

%files
/home/tibuch/Gitrepos/FourierImageTransformer/dist/fourier_image_transformers-0.1.0-py3-none-any.whl /fourier_image_transformers-0.1.0-py3-none-any.whl

%runscript
/neurodocker/startup.sh python
