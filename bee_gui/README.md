# How to Build on your own
1. download [miniconda](https://docs.anaconda.com/miniconda/)
2. open terminal on linux/mac or open Anaconda Prompt on the windows
3. create conda env
```
conda create -n bee_gui -y
conda activate bee_gui
conda install pip -y
conda install --yes --file requirements-conda.txt
pip install -r requirements-pip.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
4. run&test BEEx GUI
```
cd to the project folder
python app.py
```
5. build the BEEx GUI
```
pyinstaller app.py
```
the runnable file will be in the `dist` folder.
