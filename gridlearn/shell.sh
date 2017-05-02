#!/bin/bash

docker build -t bmeg/gridlearn .

docker run -ti --rm -u `id -u` -e HOME=$HOME -v $HOME:$HOME -w `pwd` bmeg/gridlearn /bin/bash