# Xsuite Installation
For more details, refer here: [xsuite installation](https://xsuite.readthedocs.io/en/latest/installation.html).
To use xsuite, you need to have python, a linux-like terminal.
This means that a Mac/Linux computer is fine, while for a windows computer you'll have to first download [wsl](https://learn.microsoft.com/en-us/windows/wsl/install)
If you are using wsl, remember to download in wsl all the necessities you need (for example: git)

## Miniforge
First, install python in your terminal.
Now, you need to create a python environment. Xsuite recommends downloading miniforge. 
For linux/wsl terminal, run:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```
mac terminal:
```
curl -OL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh
bash Miniforge3-MacOSX-$(uname -m).sh
```
If you are using a wsl machine, you need to download miniforge _in the wsl command line_, and _not_ in the cmd.

**Small explanation:**
A python environment is a clean python installation which is seperate from the global python in your computer. 
The seperation is important in case one project needs a specific version of a library, while another one needs a different version. 
This seperation is possible using different python environments for each project.

## Environment
After downloading miniforge, restart the terminal, and then create an environment. you can do that with:
```
conda create -n xsuite-env
conda activate xsuite-env
```
Now, you've created an environment! And also the env is "activated", which means that when you install libraries through the terminal, this libraries will only be downloaded to this environment. 
You can now install all the python libraries you need using pip. 

## Using environment in vscode
Now, I"ll explain how to use that environment using vscode.
For all operating systems, start by creating a new directory. 

_for windows_
In your normal (not wsl) vscode installation, install the wsl extension. 
Next, press ctrl+shift+p, write `WSL: Connect to WSL`
Now you are using vscode in WSL! To use the environment you picked earlier, press ctrl+shift+p, write `Python: Select interpreter`, and choose the created environment. 
And that's it folks. 

_for linux/mac_
You don't need to download the wsl extension, so your life is easier. In vscode, press ctrl+shift+p, write `Python: Select interpreter`, and choose the created environment. 
And that's it folks. 
