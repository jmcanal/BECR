#!/bin/sh

cd ../../lib/openie

tweets='../../code/data/preprocessed/final/filtered_tweets_train.txt'
output='../../code/outputs/openie/test_openie.op'

java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ${tweets} ${output}