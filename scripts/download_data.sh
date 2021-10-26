#!/bin/bash


URL_DATA="https://docs.google.com/uc?export=download&id=1O-YcCkPRDZTKX3srdk4dcB0JNA3JDEyF"

wget --no-check-certificate $URL_DATA -O "../datasets/data_sample.zip"
unzip ../datasets/data_sample.zip