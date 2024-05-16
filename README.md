# Hop-Decorate (HopDec) O4739

High throughput molecular dynamics workflow for generating atomistic databases of defect transport in chemically complex materials.

## Table of Contents

- [Hop-Decorate (HopDec)](#hop-decorate-hopdec)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Bug Reports and Discussions](#bug-reports-and-discussions)
  - [Code limitations](#code-limitations)
  - [License & Copyright](#license--copyright)

## Description

The Hop-Decorate code has 2 main functionalities:

1.	Given an atomic configuration representing some material at the atomic scale with some defect present, Hop-Decorate automatically either runs molecular dynamics on that atomic configuration at a user defined temperature in order to discover defect transitions or conducts a Dimer search [1] for a new transition. The energy barrier of the discovered transition is then calculated with an implementation of the Nudged Elastic Band (NEB) method [2]. This searching and NEB calculation process is completed iteratively to generate a database of possible transitions that the defect can make in the material which can be returned to the user.

2.	The other functionality requires the code to know about a defect transition from one stable configuration to another, either as an input by the user or discovered by the previously described routine. Regardless of the source of the transition between atomic configurations, the code calculates how the energy barrier of the transition changes when the atoms are assigned to be different atomic species with concentrations provided by the user. For example, the user may input that they are interested in a 50:50 Cu:Ni ratio alloy. The code would then randomly assign atoms in the atomic configurations as Cu or Ni at a ratio of 50:50 and then recompute the energy barrier of the transition of interest. This would generate a distribution of energy barriers and other statistics which are returned to the user.

These two functions are used together in the code to automatically generate transitions between defect configurations and calculate the distribution of energy barriers of a given transition and user defined alloy composition.


[1] Henkelman, Graeme, and Hannes Jónsson. "A dimer method for finding saddle points on high dimensional potential surfaces using only first derivatives." The Journal of chemical physics 111.15 (1999): 7010-7022.

[2] Henkelman, Graeme, Blas P. Uberuaga, and Hannes Jónsson. "A climbing image nudged elastic band method for finding saddle points and minimum energy paths." The Journal of chemical physics 113.22 (2000): 9901-9904.


<!-- ## Features

List out the key features of your project here. What makes it unique or useful? This can be a bullet-point list or a table. -->

## Installation

### Requirements

List of third-party python libraries that this code requires to run:

* [LAMMPS](https://www.lammps.org)
* [numPy](http://www.numpy.org/)
* [sciPy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [networkx](https://networkx.org)
* [ASE](https://wiki.fysik.dtu.dk/ase/)
* [pandas](https://pandas.pydata.org)
* [openMPI](https://www.open-mpi.org)
* [mpi4py](https://pypi.org/project/mpi4py/)

It is recommended that you install these in a conda requirement leverging the requirements.txt file.
For example, setting up your conda environment will look something like this:

```bash   
conda create --name my_env  
conda activate my_env  
conda install -c conda-forge --file /path/to/requirements.txt  
```
This can take some time...

Some users have reported issues with installing ASE with conda, if you also have problems consider using pip:
```bash   
pip install ase
```


It is then recommended to add these lines to your .zshrc or .bashrc:  
```bash  
export PATH=$PATH:/path/to/hopdecorate/  
export PYTHONPATH=$PYTHONPATH:/path/to/hopdecorate/  
```

## Bug Reports and Discussions

If you encounter any bugs, have feature requests, or want to discuss anything related to the project, feel free to join the Slack channel [#Hop-Decorate](https://join.slack.com/t/hop-decorate/shared_invite/zt-2e4clgm8w-Hl82df6GMjmLkKm8_hcvcA).


## Code limitations

* All structures that you pass to the code need to have [0,0,0] as their origin.
* All structures must also be cubic

## License & Copyright

This program underwent formal release process with Los Alamos National Lab 
with reference number O4739

© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


