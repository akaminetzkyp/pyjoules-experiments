FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN pip install \
    jupyterlab \
    ipywidgets \
    pandas \
    seaborn \
    pyJoules[nvidia]

CMD jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 8888

