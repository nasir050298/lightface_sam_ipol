# LightFace-SAM v2 — IPOL online demo (pair verification)

## Local (Docker) test

```bash
docker build -f .ipol/Dockerfile -t lightface-ipol .
docker run --rm -v "$PWD:/work" -w /work lightface-ipol \
  python3 src/ipol_online_demo.py data/sample_a.png data/sample_b.png --th 0.30
```
