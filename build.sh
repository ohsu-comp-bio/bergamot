#!/bin/bash

protoc --python_out ml_schema -I proto proto/ml_schema.proto 
