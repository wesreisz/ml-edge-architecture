FROM adoptopenjdk/openjdk13

RUN apt-get update
RUN apt-get -y install libopenblas-dev libblas-dev liblapack-dev gfortran

ADD run.sh /usr/local/classifier/run.sh
RUN chmod a+rx  /usr/local/classifier/run.sh

ADD image-classifier-web*.jar /usr/local/classifier/


ADD model/imagenet1000_clsidx_to_labels.txt /usr/local/classifier/model/
ADD model/xception.h5 /usr/local/classifier/model/

CMD cd /usr/local/classifier/ && ./run.sh

EXPOSE 8080