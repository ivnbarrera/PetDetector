FROM python:3.8-slim-buster
WORKDIR /code
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV SECRET_KEY=15219f95-e41a-11eb-90a9-00e045b52e33
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt --upgrade
EXPOSE 5000
COPY . .
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000","--access-logfile","-", "main:app"]
#CMD ["flask", "run"]