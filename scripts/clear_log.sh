#!/bin/bash
cd ../data/log
truncate -s 0 hc.log
echo 'hc.log cleared!'