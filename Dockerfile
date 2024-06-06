FROM python:3.9.17

COPY ./requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./index.py /index.py

CMD ["python", "index.py"]