FROM python:3.11-slim

WORKDIR /app

COPY ./xAI_shap/requirements.txt .

RUN pip install -r requirements.txt

ADD ./common_libraries /app/common_libraries/
ADD ./xAI_shap /app/xAI_shap/

COPY ./runShap.py /app/runShap.py

CMD ["python", "runShap.py"]