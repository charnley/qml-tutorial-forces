
PYTHON=./env/bin/python
CONDA=conda

all: env

setup: env pip data download_dft download_ccsd

env:
	${CONDA} env create -f environment.yml -p env

pip: env
	${PYTHON} -m pip install numpy
	${PYTHON} -m pip install -r requirements.txt --no-cache-dir

jupyter-pip:
	${PYTHON} -m pip install nglview ipywidgets
	./env/bin/jupyter-nbextension enable --py --sys-prefix widgetsnbextension
	./env/bin/jupyter-nbextension enable --py --sys-prefix nglview

data:
	mkdir -p data

download_dft: data
	cd data; wget https://ndownloader.figshare.com/files/12842591 -O training_data.tar.bz2
	cd data; bunzip2 training_data.tar.bz2

download_ccsd: data
	cd data

#

test:
	${PYTHON} tutorial/training.py --test

deploy:
	${PYTHON} tutorial/training.py

molecular_dynamics:
	${PYTHON} tutorial/molecular_dynamics.py --model data/_deploy_


#

notebook:
	env/bin/jupyter notebook

#

clean:
	rm *.pyc __pycache__

super-clean:
	rm -fr data env __pycache__

