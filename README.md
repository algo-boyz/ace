

# Deps

```zsh
brew install opencv cmake pkg-config

# ONNX Runtime (official arm64 package includes CoreML EP)
# Download: https://github.com/microsoft/onnxruntime/releases

# hnswlib (header-only, for 1:N search)
git clone https://github.com/nmslib/hnswlib.git
# copy hnswlib/ folder into .
```

## Models
```zsh
# YOLOv11n-face
# https://github.com/akanametov/yolo-face/releases

# ArcFace
# https://huggingface.co/onnx-community/arcface-onnx  or InsightFace w600k_r50
hf download onnx-community/arcface-onnx arcface_r50.onnx --local-dir models

# 2. Build
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$HOME/work/libs/onnxruntime-osx-arm64-1.22.0
make -j$(sysctl -n hw.ncpu)

# 3. Run
./face_rec                 # webcam
./face_rec /path/to/video.mp4
```
