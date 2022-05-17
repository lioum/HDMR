# HDMR
Harris Detector Master Race

# Install project
poetry install

# Compile Code
meson setup --buildtype=release release
meson compile -C release

# Run program

## Detect Points

### Don't forget to install tkinter on Linux

poetry run detect_points <path to image> <number of points>

## Process Video

poetry run process_video <path to video>

Result in out.mp4
