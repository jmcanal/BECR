#!/bin/sh

cd ../../lib/openie

tweets='../../code/data/preprocessed/hashtag-emotweets.txt'
output='../../code/outputs/openie/hashtag_emo-openie.op'

java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ${tweets} ${output}