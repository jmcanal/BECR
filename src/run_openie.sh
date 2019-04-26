#!/bin/sh


java -Xmx10g -XX:+UseConcMarkSweepGC -jar ../lib/openie/openie-assembly.jar ../data/open_ie-test.txt ../outputs/eng5dev-openie.op

#java -Xmx10g -XX:+UseConcMarkSweepGC -jar ../lib/openie/openie-assembly.jar ../data/5_pt/eng/twitter-2016dev-CE-sents.txt ../outputs/eng5dev-openie.op