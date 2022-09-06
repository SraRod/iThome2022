FROM projectmonai/monai:0.9.1

# install opencv dependency
RUN apt update &&\
    apt install libgl1-mesa-glx -y &&\
    conda install -c conda-forge nodejs=16.12.0 -y
    # for nodejs > 12...

# specify version if you could...
COPY requirements.txt .

# install pyton dependency
RUN pip install --upgrade pip &&\
    pip install -r requirements.txt

# set workspace to python path
ENV PYTHONPATH $PYTHONPATH