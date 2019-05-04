#!/bin/sh

cd ../../lib/openie
#java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ../../code/data/openie-test.txt ../../code/outputs/eng5dev-openie.op

#java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ../../code/data/preprocessed/emotweets.txt ../../code/outputs/eng5dev-openie.op
java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ../../code/data/preprocessed/emotweets2.txt ../../code/outputs/18EnEc-openie.op