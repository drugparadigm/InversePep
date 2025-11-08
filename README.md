# InversePep

_Diffusion-Driven Structure-Based Inverse Folding for Functional Peptides._

![Model Architecture](fig/pipeline.png)

## Installation

Run the following command to create the conda environment ( Use Linux to avoid compatibility issues )

```python
conda env create -f environment.yaml
```

Run the following command to activate the conda environment

```python
conda activate inversepep
```

Installing Additional Softwares

Make the script executable (only needed once, Linux/Mac)

```python
chmod +x additional_software.sh
```

Run the installation script

```python
./additional_software.sh
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
