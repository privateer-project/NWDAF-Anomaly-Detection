FROM python:3.11-slim

WORKDIR /app

COPY ./xAI_lime/requirements.txt .

RUN pip install -r requirements.txt

ADD ./xAI_lime /app/xAI_lime/
ADD ./common_libraries /app/common_libraries/

COPY ./runLime.py /app/runLime.py

CMD ["python", "runLime.py"]