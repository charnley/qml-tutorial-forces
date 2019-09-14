
PYTHON=./env/bin/python
CONDA=conda

setup: env pip data

env:
	${CONDA} env create -f environment.yml -p env

pip: env
	${PYTHON} -m pip install numpy
	${PYTHON} -m pip install -r requirements.txt --no-cache-dir

data:
	mkdir -p data
	cd data; wget https://ndownloader.figshare.com/files/12842591 -O training_data.tar.bz2
	cd data; bunzip2 training_data.tar.bz2

#




#

clean:
	rm *.pyc __pycache__

super-clean:
	rm -fr data env __pycache__

