FROM python:3.11-slim

WORKDIR /app

COPY ./load_dataset/requirements.txt .

RUN pip install -r requirements.txt

ADD ./common_libraries /app/common_libraries/
ADD ./load_dataset /app/load_dataset/

COPY ./runLoadDataset.py /app/runLoadDataset.py

CMD ["python", "runLoadDataset.py"]