#pragma once

#include "EasyBMP/EasyBMP.h"

void create_BMP_grayscale(BMP& inputBMP, BMP& outputBMP,
		uint8_t* outputPixelsRed, uint8_t* outputPixelsGreen, uint8_t* outputPixelsBlue) {
    const int w = inputBMP.TellWidth();
    const int h = inputBMP.TellHeight();

    outputBMP.SetSize(w, h);
    outputBMP.SetBitDepth(inputBMP.TellBitDepth());
/*
    //записываем палитру и отфильтрованные пиксели
    if (inputBMP.TellBitDepth() <= 24) {
        int numColors = inputBMP.TellNumberOfColors();

        for (int i = 0; i < numColors; i++) {
            RGBApixel color = inputBMP.GetColor(i);
            outputBMP.SetColor(i, color);
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint8_t color = outputPixels[y * w + x];
                RGBApixel pixel = outputBMP.GetColor(color);
                outputBMP.SetPixel(x, y, pixel);
            }
        }
    }
*/
    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            outputBMP(x, y)->Red = outputPixelsRed[y*w + x];
            outputBMP(x, y)->Green = outputPixelsGreen[y*w + x];
            outputBMP(x, y)->Blue = outputPixelsBlue[y*w + x];
	}
    }
}
