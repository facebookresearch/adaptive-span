#!/bin/bash

if [[ ! -d 'data/enwik8' ]]; then
  mkdir -p data/enwik8
  cd data/enwik8
  echo "Downloading enwik8 data ..."
  wget --continue http://mattmahoney.net/dc/enwik8.zip
  wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
  python3 prep_enwik8.py
  cd ../..
fi


if [[ ! -d 'data/text8' ]]; then
  mkdir -p data/text8
  cd data/text8
  echo "Downloading text8 data ..."
  wget --continue http://mattmahoney.net/dc/text8.zip
  wget https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/prep_text8.py
  python prep_text8.py
  cd ../..
fi
