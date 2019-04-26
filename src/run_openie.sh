#!/bin/sh

cd ../lib/openie
echo ls
java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar ../../code/data/openie-test.txt ../../code/outputs/eng5dev-openie.op

#java -Xmx10g -XX:+UseConcMarkSweepGC -jar ../lib/openie/openie-assembly.jar ../data/5_pt/eng/twitter-2016dev-CE-sents.txt ../outputs/eng5dev-openie.op