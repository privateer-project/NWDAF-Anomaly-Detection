FROM python:3.11-slim

WORKDIR /app

COPY ./load_models/requirements.txt .

RUN pip install -r requirements.txt

ADD ./common_libraries /app/common_libraries/
ADD ./load_models /app/load_models/

COPY ./runModels.py /app/runModels.py

CMD ["python", "runModels.py"]