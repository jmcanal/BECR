#!/bin/sh

java -Xmx10g -XX:+UseConcMarkSweepGC -jar ../lib/openie-assembly.jar  ../data/open_ie_test.txt
../outputs/test-openie.op