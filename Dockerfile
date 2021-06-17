FROM python:3.8.5
COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"
RUN apt-get update 
RUN pip3 install --upgrade pip
RUN apt-get -y install default-jre
RUN pip3 install -Ur /tmp/requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "dashboard.py" ]