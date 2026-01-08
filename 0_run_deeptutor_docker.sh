docker run -d --name deeptutor \
  -p 8001:8001 -p 3782:3782 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config:ro \
  deeptutor  # or ghcr.io/hkuds/deeptutor:latest