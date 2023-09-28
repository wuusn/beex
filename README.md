conda env export > environment.yaml
conda env create -f environment.yaml

conda create --name bee
conda activate bee
conda install -c conda-forge r r-base r-lmertest r-emmeans rpy2
pip install pymer4
pip install numpy pillow ipykernel jupyterlab openpyxl  SimpleITK scikit-learn scikit-image plotly openslide-python matplotlib pandas