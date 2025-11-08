# InversePep

_Diffusion-Driven Structure-Based Inverse Folding for Functional Peptides._

![Model Architecture](fig/pipeline.png)

## Installation

Run the following command to create the conda environment

```python
conda env create -f environment.yaml
```

Run the following command to activate the conda environment

```python
conda activate inversepep
```

Installing Additional Softwares

```python
conda install -y -c conda-forge libgfortran

pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu116

pip install torch_cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu116.html

pip install torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu116.html

pip install torch_geometric==2.3.1

pip install transformers

pip install fair-esm
```

Make the libgfortran available ( Needed only once for every fresh start )

```python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
## Usage

Run the following command to run the example for sequence generation:

```python
python inf.py --pdb_file example/1DTC.pdb
```
