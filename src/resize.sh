#!/bin/bash

file_name=$(basename "$1"); file_name="${file_name%.*}"
dir_name=$(dirname "$1")

convert $1 -resize 1280x720! ${dir_name}/${file_name}.bmp