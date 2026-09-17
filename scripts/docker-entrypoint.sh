#!/bin/sh
# CTranslate2 (faster-whisper) needs cuBLAS/cuDNN: reuse the NVIDIA libraries installed with the CUDA build of PyTorch.
set -e
NV_LIBS="$(python -c "import glob, os, site; print(':'.join(sorted({os.path.dirname(p) for s in site.getsitepackages() for p in glob.glob(os.path.join(s, 'nvidia', '*', 'lib', '*.so*'))})))" 2>/dev/null || true)"
if [ -n "$NV_LIBS" ]; then
  export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$@"
