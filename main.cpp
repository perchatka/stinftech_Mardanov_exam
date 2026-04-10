#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

#include <sycl/sycl.hpp>
#include "processImageData.h"
#include "medianFilter.h"
#include "medianFilterGPU.h"
//#include "medianFilterSIMD.h"
//using namespace sycl;



//сравнение результатов 2-ух фильтров
bool compare_data(const uint8_t* A, const uint8_t* B, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (A[i] != B[i]) return false;
    return true;
}

//функция для "разогрева" GPU (инициализация очереди, выделение ресурсов, компиляция)
void warmupGPU(sycl::queue q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {});
        });
    q.wait();
}



int main() {
    //-------------------------- ЗАШУМЛЕННОЕ ИЗОБРАЖЕНИЕ --------------------------

    BMP inputBMP;
    const int ITERATIONS = 1000;//количество повторений для нагрузки

    //std::string inputfilepath = "img/noise/";
    std::string filename = "test.bmp";
    //std::string filename = "gaussian_50.bmp";
    //inputBMP.ReadFromFile((inputfilepath + filename).c_str());
    inputBMP.ReadFromFile((filename).c_str());
    const int w = inputBMP.TellWidth();
    const int h = inputBMP.TellHeight();
    uint8_t *inputPixelsRed = new uint8_t[w * h];
    uint8_t *inputPixelsGreen = new uint8_t[w * h];
    uint8_t *inputPixelsBlue = new uint8_t[w * h];
    std::cout<<"Bit depth: "<<inputBMP.TellBitDepth()<<"\nnumber of colors: "<<inputBMP.TellNumberOfColors()<<"\n";
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ebmpBYTE pixelRed = inputBMP.GetPixel(x, y).Red;
            ebmpBYTE pixelGreen = inputBMP.GetPixel(x, y).Green;
            ebmpBYTE pixelBlue = inputBMP.GetPixel(x, y).Blue;
            inputPixelsRed[y * w + x] = static_cast<uint8_t>(pixelRed);
            inputPixelsGreen[y * w + x] = static_cast<uint8_t>(pixelGreen);
            inputPixelsBlue[y * w + x] = static_cast<uint8_t>(pixelBlue);
        }
    }

    //-------------------------- ФИЛЬТРАЦИЯ + ЗАМЕРЫ --------------------------



    //ОДНОПОТОЧНАЯ ВЕРСИЯ
    uint8_t* outputPixelsRed = new uint8_t[w * h];
    uint8_t* outputPixelsGreen = new uint8_t[w * h];
    uint8_t* outputPixelsBlue = new uint8_t[w * h];

    auto start1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < ITERATIONS; ++i) {
        MedianFilter::median_filter_3x3(inputPixelsRed, outputPixelsRed, w, h, w);
        MedianFilter::median_filter_3x3(inputPixelsGreen, outputPixelsGreen, w, h, w);
        MedianFilter::median_filter_3x3(inputPixelsBlue, outputPixelsBlue, w, h, w);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms" << std::endl;


    //GPU ВЕРСИЯ

    //warmup
    sycl::queue q;
    warmupGPU(q);

    //base version (GPU)
    uint8_t* outputPixelsRed_gpu = new uint8_t[w * h]; 
    uint8_t* outputPixelsGreen_gpu = new uint8_t[w * h];
    uint8_t* outputPixelsBlue_gpu = new uint8_t[w * h];

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_v1(inputPixelsRed, outputPixelsRed_gpu, w, h, w, q);
        MedianFilterGPU::median_filter_3x3_v1(inputPixelsGreen, outputPixelsGreen_gpu, w, h, w, q);
        MedianFilterGPU::median_filter_3x3_v1(inputPixelsBlue, outputPixelsBlue_gpu, w, h, w, q);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "GPU version v1: " << duration2.count() << " ms" << std::endl;

    //tiled version (GPU)
    uint8_t* outputPixelsRed_gpu_v2 = new uint8_t[w * h];
    uint8_t* outputPixelsGreen_gpu_v2 = new uint8_t[w * h];
    uint8_t* outputPixelsBlue_gpu_v2 = new uint8_t[w * h];

    auto start3 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_v2(inputPixelsRed, outputPixelsRed_gpu_v2, w, h, w, q);
        MedianFilterGPU::median_filter_3x3_v2(inputPixelsGreen, outputPixelsGreen_gpu_v2, w, h, w, q);
        MedianFilterGPU::median_filter_3x3_v2(inputPixelsBlue, outputPixelsBlue_gpu_v2, w, h, w, q);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "GPU version v2: " << duration3.count() << " ms" << std::endl;

    //-------------------------- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ --------------------------

    //создаем отфильтрованное изображение (GPU)
    BMP outputBMP;
    create_BMP_grayscale(inputBMP, outputBMP, outputPixelsRed_gpu, outputPixelsGreen_gpu, outputPixelsBlue_gpu);
    //std::string outputfilepath = "img/filtered/";
    //std::string prefix = "filtered_";
    //outputBMP.WriteToFile((outputfilepath + prefix + filename).c_str());
    outputBMP.WriteToFile("out.bmp");

    //-------------------------- ПРОВЕРКА --------------------------

    assert(compare_data(outputPixelsRed, outputPixelsRed_gpu, w * h));
    assert(compare_data(outputPixelsGreen, outputPixelsGreen_gpu, w * h));
    assert(compare_data(outputPixelsBlue, outputPixelsBlue_gpu, w * h));

    assert(compare_data(outputPixelsRed, outputPixelsRed_gpu_v2, w * h));
    assert(compare_data(outputPixelsGreen, outputPixelsGreen_gpu_v2, w * h));
    assert(compare_data(outputPixelsBlue, outputPixelsBlue_gpu_v2, w * h));

    std::cout << "Processing complete!" << std::endl;
    delete[] inputPixelsRed; delete[] inputPixelsGreen; delete[] inputPixelsBlue;
    delete[] outputPixelsRed; delete[] outputPixelsGreen; delete[] outputPixelsBlue;
    delete[] outputPixelsRed_gpu; delete[] outputPixelsGreen_gpu; delete[] outputPixelsBlue_gpu;
    delete[] outputPixelsRed_gpu_v2; delete[] outputPixelsGreen_gpu_v2; delete[] outputPixelsBlue_gpu_v2;
    return 0;
}
