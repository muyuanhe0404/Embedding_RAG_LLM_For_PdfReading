#!/usr/bin/env bash

# run (uvicorn main:app --reload), then (bash test_api.sh)

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who is Gatsby?"}'