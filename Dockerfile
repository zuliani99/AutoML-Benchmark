FROM python
COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"
RUN apt-get update 
RUN apt-get -y install default-jre
RUN pip3 install -Ur /tmp/requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "dashboard.py" ]