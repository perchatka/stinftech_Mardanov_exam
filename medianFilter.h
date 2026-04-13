#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include "utils.h"




class MedianFilter {
private:
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_3x3(size_t num_ch, const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
};




uint8_t MedianFilter::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}


//width - ширина изображения в пикселях
//stride - фактическая ширина строки пикселей в байтах с учетом всех отсутпов и выравниваний (может быть больше)
void MedianFilter::median_filter_3x3(size_t num_ch, const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    stride *= num_ch;
    //основной цикл
    for (size_t y = 0; y < height; ++y) {
        const uint8_t* y0 = input + (y > 0 ? y - 1 : 0) * stride;
        const uint8_t* y1 = input + y * stride;                   
    	const uint8_t* y2 = input + (y < height - 1 ? y + 1 : y) * stride;

        uint8_t* out_row = output + y * stride;

        for (size_t x = 0; x < width; ++x) {
            const size_t x0 = (x > 0 ? x - 1 : 0);
            const size_t x1 = x;
            const size_t x2 = (x + 1 < width ? x + 1 : x);

            const size_t p0 = x0 * num_ch;
            const size_t p1 = x1 * num_ch;
            const size_t p2 = x2 * num_ch;

            for (size_t ch = 0; ch < num_ch; ++ch) {
                uint8_t window[9];
                window[0] = y0[x0 * num_ch + ch]; window[1] = y0[x1 * num_ch + ch]; window[2] = y0[x2 * num_ch + ch];
                window[3] = y1[x0 * num_ch + ch]; window[4] = y1[x1 * num_ch + ch]; window[5] = y1[x2 * num_ch + ch];
                window[6] = y2[x0 * num_ch + ch]; window[7] = y2[x1 * num_ch + ch]; window[8] = y2[x2 * num_ch + ch];

                out_row[x * num_ch + ch] = median_9(window);
            }
        }
    }
}
