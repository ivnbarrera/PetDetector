FROM tensorflow/tensorflow:2.4.1
WORKDIR /code
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt --upgrade
EXPOSE 5000
COPY . .
CMD ["flask", "run"]