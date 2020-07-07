#!/usr/bin/env bash

echo "Starting spring boot server..."
export CLASSIFIER_HOME=`/usr/local/classifier/`
APP_JAR=`find $CLASSIFIER_HOME/ -name image-classifier-web*.jar -exec echo -n {} \;`
java -jar -Djava.library.path="/usr/lib/x86_64-linux-gnudocker:/usr/lib/x86_64-linux-gnu/openblas:usr/java/packages/lib:/usr/lib64:/lib64:/usr/lib" $APP_JAR
