#!/bin/bash

cd ./model_save
find . -name "checkpoint.%" -type f -delete
find . -name "trained.%" -type f -delete