# RUNETag Python Port

This is a Python port of the original RUNETag project: https://github.com/artursg/RUNEtag

The port exists because the original project has been largely unattended for a long time and a fresher Python codebase was needed for ongoing work.

Some performance improvements are still needed, especially in the detector path when a marker is present in the frame.

This folder contains a pure Python / OpenCV port of the RUNETag generator and detector.

It currently includes:

- `generator.py` to generate `codes.txt`, render a tag image, and export a `RUNE_direct` descriptor
- `detector.py` to load descriptors and detect tags from images
- `webcam_poc.py` to run the detector on a webcam feed

This is a working port in progress. The geometry and file formats follow the original project closely, but the detector is still being optimized and the original NTL BCH decode path is not ported yet.

## Setup

Create a virtual environment if you want one, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Generate Codes

Generate a `codes.txt` file:

```bash
python3 generator.py --generate-codes codes.txt
```

## Generate a Tag

Create a marker image and its descriptor:

```bash
python3 generator.py codes.txt --tag-index 0 --png tag0.png --descriptor tag0.txt --name TAG0 --marker-size-mm 200
```

The important outputs are:

- `tag0.png` with the rendered marker
- `tag0.txt` with the marker descriptor used by the detector

## Detect From Webcam

Run the webcam proof of concept:

```bash
python3 webcam_poc.py ./ --camera 0 --width 1280 --height 720 --fx 1200 --fy 1200 --cx 640 --cy 360 --process-width 640
```

If you have a single descriptor, you can point directly to it:

```bash
python3 webcam_poc.py tag0.txt --camera 0 --width 1280 --height 720 --fx 1200 --fy 1200 --cx 640 --cy 360 --process-width 640
```

Press `q` or `Esc` to quit.

## Detect From an Image

Run the detector on a saved image:

```bash
python3 detector.py frame.png tag0.txt --fx 1200 --fy 1200 --cx 640 --cy 360
```

## Notes

- `--tag-index` is the row index inside `codes.txt`
- the detector displays the marker `idx` stored in the descriptor, which is the RUNETag codebook id
- the webcam script accepts either one descriptor file or a directory containing descriptor files
- performance is still under active work, especially on the marker-present path
