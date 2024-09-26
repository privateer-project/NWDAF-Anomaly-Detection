FROM nvidia/cuda:11.4.3-base-ubuntu20.04

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y \
        python3-pip 
        # libglib2.0-0

RUN pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt

RUN pip install notebook

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]