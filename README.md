# vs-frameblender
Modified version of VapourSynth's AverageFrames which allows for more weights (128 instead of 31)

> [!WARNING]
> This fork is completely and utterly Vibe Coded, use at your own risk, I recommend you use: https://github.com/couleurm/vs-frameblender

## Compiling

### Windows
````
g++ main.cpp -shared -std=c++17 -ID:\GitHub\vs-frameblender\vs-frameblender\vapoursynth -o foo.dll
````
### Linux
g++
````
clang++ main.cpp -shared -std=c++17 -I~\GitHub\vs-frameblender\vs-frameblender\vapoursynth -o foo.so
sudo mv foo.so /usr/lib/vapoursynth
````
