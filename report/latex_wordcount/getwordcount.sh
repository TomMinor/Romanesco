#!/bin/bash

echo "Report word count :"
texcount report.tex | grep "Words in text:" | awk '{print $4}'

echo "Annotated bibliography word count :"
./count_bibliography.py bib_icereport.bib
