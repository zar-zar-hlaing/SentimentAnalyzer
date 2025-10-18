#!/bin/bash
# downloading prerequisite models for polyglot:sentiment analysis
languages=(
	   	en
	   )

for lang in "${languages[@]}"
do
    polyglot download sentiment2.$lang
done

