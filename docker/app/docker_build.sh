#!/bin/bash

cp ../../pyproject.toml .
docker build -f Dockerfile -t aimilefth/privateer-ad:app . --push
rm pyproject.toml