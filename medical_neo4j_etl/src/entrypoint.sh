#!/bin/bash

echo "Running ETL to move medical data from csvs to Neo4j..."

python dedupe_csvs.py

python bulk_csv_writer.py
