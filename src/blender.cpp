#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FB_USE_AVX2 1
#endif
#endif

#include "../vapoursynth/VSHelper4.h"
#include "../vapoursynth/VapourSynth4.h"

struct FrameBlendData {
    VSNode *node = nullptr;
    const VSVideoInfo *videoInfo = nullptr;
    std::vector<float> weightPercents;
    bool processPlane[3] = { false, false, false };
};

struct NodeScopeGuard {
    VSNode *node = nullptr;
    const VSAPI *vsapi = nullptr;

    ~NodeScopeGuard() {
        if (node && vsapi) {
            vsapi->freeNode(node);
        }
    }

    VSNode *release() {
        VSNode *released = node;
        node = nullptr;
        return released;
    }
};

struct FrameListGuard {
    std::vector<const VSFrame *> &frames;
    const VSAPI *vsapi = nullptr;

    ~FrameListGuard() {
        if (vsapi) {
            for (const VSFrame *frame : frames) {
                if (frame) {
                    vsapi->freeFrame(frame);
                }
            }
        }
    }
};

struct FrameWindow {
    int firstFrame;
    int lastFrame;
    int centerIndex;
};

#if defined(FB_USE_AVX2)
static inline void loadU8x16ToF32(const uint8_t *source, __m256 &lowVector, __m256 &highVector) {
    __m128i u8Low = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(source));
    __m128i u8High = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(source + 8));
    lowVector = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8Low));
    highVector = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8High));
}

static inline void storeClampedU8x16(uint8_t *destination, __m256 lowVector, __m256 highVector, __m256 halfBias) {
    __m256 biasedLow = _mm256_add_ps(lowVector, halfBias);
    __m256 biasedHigh = _mm256_add_ps(highVector, halfBias);

    __m256i intLow = _mm256_cvttps_epi32(biasedLow);
    __m256i intHigh = _mm256_cvttps_epi32(biasedHigh);

    __m256i packed16 = _mm256_packus_epi32(intLow, intHigh);
    packed16 = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));

    __m128i laneLow16 = _mm256_castsi256_si128(packed16);
    __m128i laneHigh16 = _mm256_extracti128_si256(packed16, 1);
    __m128i packed8 = _mm_packus_epi16(laneLow16, laneHigh16);

    _mm_storeu_si128(reinterpret_cast<__m128i *>(destination), packed8);
}

static inline void loadU16x16ToF32(const uint16_t *source, __m256 &lowVector, __m256 &highVector) {
    __m128i u16Low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
    __m128i u16High = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 8));
    lowVector = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(u16Low));
    highVector = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(u16High));
}

static inline void storeClampedU16x16(
    uint16_t *destination,
    __m256 lowVector,
    __m256 highVector,
    __m256 halfBias,
    __m256i maxValueVector,
    bool clampMax
) {
    __m256 biasedLow = _mm256_add_ps(lowVector, halfBias);
    __m256 biasedHigh = _mm256_add_ps(highVector, halfBias);

    __m256i intLow = _mm256_cvttps_epi32(biasedLow);
    __m256i intHigh = _mm256_cvttps_epi32(biasedHigh);

    __m256i packed16 = _mm256_packus_epi32(intLow, intHigh);
    packed16 = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));

    if (clampMax) {
        packed16 = _mm256_min_epu16(packed16, maxValueVector);
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination), packed16);
}

static void blendRowU8Avx2(
    uint8_t *VS_RESTRICT destination,
    const uint8_t * const *sources,
    const float *weights,
    size_t sourceCount,
    int width,
    float *rowAccumulator
) {
    const int simdWidth = width & ~15;

    const uint8_t *firstSrc = sources[0];
    const __m256 firstWeight = _mm256_set1_ps(weights[0]);
    for (int x = 0; x < simdWidth; x += 16) {
        __m256 low, high;
        loadU8x16ToF32(&firstSrc[x], low, high);
        _mm256_storeu_ps(&rowAccumulator[x], _mm256_mul_ps(low, firstWeight));
        _mm256_storeu_ps(&rowAccumulator[x + 8], _mm256_mul_ps(high, firstWeight));
    }
    for (int x = simdWidth; x < width; ++x) {
        rowAccumulator[x] = firstSrc[x] * weights[0];
    }

    for (size_t i = 1; i < sourceCount; ++i) {
        const uint8_t *currentSrc = sources[i];
        const __m256 currentWeight = _mm256_set1_ps(weights[i]);
        for (int x = 0; x < simdWidth; x += 16) {
            __m256 low, high;
            loadU8x16ToF32(&currentSrc[x], low, high);
            __m256 accLow = _mm256_loadu_ps(&rowAccumulator[x]);
            __m256 accHigh = _mm256_loadu_ps(&rowAccumulator[x + 8]);
            _mm256_storeu_ps(&rowAccumulator[x], _mm256_fmadd_ps(low, currentWeight, accLow));
            _mm256_storeu_ps(&rowAccumulator[x + 8], _mm256_fmadd_ps(high, currentWeight, accHigh));
        }
        for (int x = simdWidth; x < width; ++x) {
            rowAccumulator[x] += currentSrc[x] * weights[i];
        }
    }

    const __m256 halfBias = _mm256_set1_ps(0.5f);
    for (int x = 0; x < simdWidth; x += 16) {
        __m256 accLow = _mm256_loadu_ps(&rowAccumulator[x]);
        __m256 accHigh = _mm256_loadu_ps(&rowAccumulator[x + 8]);
        storeClampedU8x16(&destination[x], accLow, accHigh, halfBias);
    }
    for (int x = simdWidth; x < width; ++x) {
        destination[x] = static_cast<uint8_t>(std::clamp(int(rowAccumulator[x] + 0.5f), 0, 255));
    }
}

static void blendRowU16Avx2(
    uint16_t *VS_RESTRICT destination,
    const uint16_t * const *sources,
    const float *weights,
    size_t sourceCount,
    int width,
    unsigned maxValue,
    int bitsPerSample,
    float *rowAccumulator
) {
    const int simdWidth = width & ~15;

    const uint16_t *firstSrc = sources[0];
    const __m256 firstWeight = _mm256_set1_ps(weights[0]);
    for (int x = 0; x < simdWidth; x += 16) {
        __m256 low, high;
        loadU16x16ToF32(&firstSrc[x], low, high);
        _mm256_storeu_ps(&rowAccumulator[x], _mm256_mul_ps(low, firstWeight));
        _mm256_storeu_ps(&rowAccumulator[x + 8], _mm256_mul_ps(high, firstWeight));
    }
    for (int x = simdWidth; x < width; ++x) {
        rowAccumulator[x] = firstSrc[x] * weights[0];
    }

    for (size_t i = 1; i < sourceCount; ++i) {
        const uint16_t *currentSrc = sources[i];
        const __m256 currentWeight = _mm256_set1_ps(weights[i]);
        for (int x = 0; x < simdWidth; x += 16) {
            __m256 low, high;
            loadU16x16ToF32(&currentSrc[x], low, high);
            __m256 accLow = _mm256_loadu_ps(&rowAccumulator[x]);
            __m256 accHigh = _mm256_loadu_ps(&rowAccumulator[x + 8]);
            _mm256_storeu_ps(&rowAccumulator[x], _mm256_fmadd_ps(low, currentWeight, accLow));
            _mm256_storeu_ps(&rowAccumulator[x + 8], _mm256_fmadd_ps(high, currentWeight, accHigh));
        }
        for (int x = simdWidth; x < width; ++x) {
            rowAccumulator[x] += currentSrc[x] * weights[i];
        }
    }

    const __m256 halfBias = _mm256_set1_ps(0.5f);
    const __m256i maxValueVector = _mm256_set1_epi16(static_cast<short>(maxValue));
    const bool clampMax = (bitsPerSample < 16);

    for (int x = 0; x < simdWidth; x += 16) {
        __m256 accLow = _mm256_loadu_ps(&rowAccumulator[x]);
        __m256 accHigh = _mm256_loadu_ps(&rowAccumulator[x + 8]);
        storeClampedU16x16(&destination[x], accLow, accHigh, halfBias, maxValueVector, clampMax);
    }
    for (int x = simdWidth; x < width; ++x) {
        destination[x] = static_cast<uint16_t>(std::clamp(int(rowAccumulator[x] + 0.5f), 0, static_cast<int>(maxValue)));
    }
}

static void blendRowFloatAvx2(
    float *VS_RESTRICT destination,
    const float * const *sources,
    const float *weights,
    size_t sourceCount,
    int width
) {
    const int simdWidth = width & ~15;

    const float *firstSrc = sources[0];
    const __m256 firstWeight = _mm256_set1_ps(weights[0]);
    for (int x = 0; x < simdWidth; x += 16) {
        __m256 f0 = _mm256_loadu_ps(&firstSrc[x]);
        __m256 f1 = _mm256_loadu_ps(&firstSrc[x + 8]);
        _mm256_storeu_ps(&destination[x], _mm256_mul_ps(f0, firstWeight));
        _mm256_storeu_ps(&destination[x + 8], _mm256_mul_ps(f1, firstWeight));
    }
    for (int x = simdWidth; x < width; ++x) {
        destination[x] = firstSrc[x] * weights[0];
    }

    for (size_t i = 1; i < sourceCount; ++i) {
        const float *currentSrc = sources[i];
        const __m256 currentWeight = _mm256_set1_ps(weights[i]);
        for (int x = 0; x < simdWidth; x += 16) {
            __m256 d0 = _mm256_loadu_ps(&destination[x]);
            __m256 d1 = _mm256_loadu_ps(&destination[x + 8]);
            __m256 s0 = _mm256_loadu_ps(&currentSrc[x]);
            __m256 s1 = _mm256_loadu_ps(&currentSrc[x + 8]);
            _mm256_storeu_ps(&destination[x], _mm256_fmadd_ps(s0, currentWeight, d0));
            _mm256_storeu_ps(&destination[x + 8], _mm256_fmadd_ps(s1, currentWeight, d1));
        }
        for (int x = simdWidth; x < width; ++x) {
            destination[x] += currentSrc[x] * weights[i];
        }
    }
}
#endif

template<typename T>
static void blendRowScalar(
    T *VS_RESTRICT destination,
    const T * const *sources,
    const float *weights,
    size_t sourceCount,
    int width,
    unsigned maxValue,
    float *rowAccumulator
) {
    const T *firstSrc = sources[0];
    const float firstWeight = weights[0];
    for (int x = 0; x < width; ++x) {
        rowAccumulator[x] = firstSrc[x] * firstWeight;
    }

    for (size_t i = 1; i < sourceCount; ++i) {
        const T *currentSrc = sources[i];
        const float currentWeight = weights[i];
        for (int x = 0; x < width; ++x) {
            rowAccumulator[x] += currentSrc[x] * currentWeight;
        }
    }

    for (int x = 0; x < width; ++x) {
        destination[x] = static_cast<T>(std::clamp(int(rowAccumulator[x] + 0.5f), 0, static_cast<int>(maxValue)));
    }
}

static void blendRowFloatScalar(
    float *VS_RESTRICT destination,
    const float * const *sources,
    const float *weights,
    size_t sourceCount,
    int width
) {
    const float *firstSrc = sources[0];
    const float firstWeight = weights[0];
    for (int x = 0; x < width; ++x) {
        destination[x] = firstSrc[x] * firstWeight;
    }

    for (size_t i = 1; i < sourceCount; ++i) {
        const float *currentSrc = sources[i];
        const float currentWeight = weights[i];
        for (int x = 0; x < width; ++x) {
            destination[x] += currentSrc[x] * currentWeight;
        }
    }
}

template<typename T>
static inline void blendRow(
    T *VS_RESTRICT destination,
    const T * const *sources,
    const float *weights,
    size_t sourceCount,
    int width,
    unsigned maxValue,
    int bitsPerSample,
    float *rowAccumulator
) {
#if defined(FB_USE_AVX2)
    if constexpr (std::is_same_v<T, uint8_t>) {
        blendRowU8Avx2(destination, reinterpret_cast<const uint8_t * const *>(sources), weights, sourceCount, width, rowAccumulator);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        blendRowU16Avx2(destination, reinterpret_cast<const uint16_t * const *>(sources), weights, sourceCount, width, maxValue, bitsPerSample, rowAccumulator);
    } else {
        blendRowScalar(destination, sources, weights, sourceCount, width, maxValue, rowAccumulator);
    }
#else
    blendRowScalar(destination, sources, weights, sourceCount, width, maxValue, rowAccumulator);
#endif
}

template<typename T>
static void blendPlaneInteger(
    VSFrame *destinationFrame,
    const VSFrame * const *sourceFrames,
    const float *weights,
    size_t sourceCount,
    int plane,
    int bitsPerSample,
    const VSAPI *vsapi
) {
    const int width = vsapi->getFrameWidth(destinationFrame, plane);
    const int height = vsapi->getFrameHeight(destinationFrame, plane);
    const ptrdiff_t dstStride = vsapi->getStride(destinationFrame, plane) / sizeof(T);
    T *VS_RESTRICT dst = reinterpret_cast<T *>(vsapi->getWritePtr(destinationFrame, plane));

    if (sourceCount == 0) {
        for (int y = 0; y < height; ++y) {
            std::memset(dst, 0, width * sizeof(T));
            dst += dstStride;
        }
        return;
    }

    const unsigned maxValue = (1U << bitsPerSample) - 1;
    std::vector<const T *> sourcePointers(sourceCount);
    std::vector<ptrdiff_t> sourceStrides(sourceCount);

    for (size_t i = 0; i < sourceCount; ++i) {
        sourcePointers[i] = reinterpret_cast<const T *>(vsapi->getReadPtr(sourceFrames[i], plane));
        sourceStrides[i] = vsapi->getStride(sourceFrames[i], plane) / sizeof(T);
    }

    std::vector<float> rowAccumulator(width + 32, 0.0f);

    for (int y = 0; y < height; ++y) {
        blendRow<T>(dst, sourcePointers.data(), weights, sourceCount, width, maxValue, bitsPerSample, rowAccumulator.data());

        for (size_t i = 0; i < sourceCount; ++i) {
            sourcePointers[i] += sourceStrides[i];
        }
        dst += dstStride;
    }
}

static void blendPlaneFloat(
    VSFrame *destinationFrame,
    const VSFrame * const *sourceFrames,
    const float *weights,
    size_t sourceCount,
    int plane,
    const VSAPI *vsapi
) {
    const int width = vsapi->getFrameWidth(destinationFrame, plane);
    const int height = vsapi->getFrameHeight(destinationFrame, plane);
    const ptrdiff_t dstStride = vsapi->getStride(destinationFrame, plane) / sizeof(float);
    float *VS_RESTRICT dst = reinterpret_cast<float *>(vsapi->getWritePtr(destinationFrame, plane));

    if (sourceCount == 0) {
        for (int y = 0; y < height; ++y) {
            std::memset(dst, 0, width * sizeof(float));
            dst += dstStride;
        }
        return;
    }

    std::vector<const float *> sourcePointers(sourceCount);
    std::vector<ptrdiff_t> sourceStrides(sourceCount);

    for (size_t i = 0; i < sourceCount; ++i) {
        sourcePointers[i] = reinterpret_cast<const float *>(vsapi->getReadPtr(sourceFrames[i], plane));
        sourceStrides[i] = vsapi->getStride(sourceFrames[i], plane) / sizeof(float);
    }

    for (int y = 0; y < height; ++y) {
#if defined(FB_USE_AVX2)
        blendRowFloatAvx2(dst, sourcePointers.data(), weights, sourceCount, width);
#else
        blendRowFloatScalar(dst, sourcePointers.data(), weights, sourceCount, width);
#endif

        for (size_t i = 0; i < sourceCount; ++i) {
            sourcePointers[i] += sourceStrides[i];
        }
        dst += dstStride;
    }
}

static FrameWindow calculateFrameWindow(int currentFrame, int windowRadius, int maxFrameIndex) {
    const int first = std::clamp(currentFrame - windowRadius, 0, maxFrameIndex);
    const int last = std::clamp(currentFrame + windowRadius, 0, maxFrameIndex);
    const int center = std::clamp(currentFrame, 0, maxFrameIndex) - first;
    return { first, last, center };
}

static void requestFrameRange(int firstFrame, int lastFrame, VSNode *node, VSFrameContext *frameCtx, const VSAPI *vsapi) {
    for (int frame = firstFrame; frame <= lastFrame; ++frame) {
        vsapi->requestFrameFilter(frame, node, frameCtx);
    }
}

static void collapseWindowWeights(
    const std::vector<float> &weights,
    int currentFrame,
    int windowRadius,
    int firstFrame,
    int maxFrameIndex,
    std::vector<float> &collapsedWeights
) {
    const int weightCount = static_cast<int>(weights.size());
    for (int k = 0; k < weightCount; ++k) {
        int targetFrame = std::clamp(currentFrame - windowRadius + k, 0, maxFrameIndex);
        collapsedWeights[targetFrame - firstFrame] += weights[k];
    }
}

static VSFrame *createDestinationFrame(
    const VSFrame *referenceFrame,
    const bool processPlane[3],
    VSCore *core,
    const VSAPI *vsapi
) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(referenceFrame);
    const int planeIndices[] = { 0, 1, 2 };
    const VSFrame *planeSources[] = {
        processPlane[0] ? nullptr : referenceFrame,
        processPlane[1] ? nullptr : referenceFrame,
        processPlane[2] ? nullptr : referenceFrame
    };

    return vsapi->newVideoFrame2(
        format,
        vsapi->getFrameWidth(referenceFrame, 0),
        vsapi->getFrameHeight(referenceFrame, 0),
        planeSources,
        planeIndices,
        referenceFrame,
        core
    );
}

static void dispatchFrameBlending(
    VSFrame *destinationFrame,
    const VSFrame * const *sourceFrames,
    const float *weights,
    size_t sourceCount,
    const bool processPlane[3],
    const VSAPI *vsapi
) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(destinationFrame);

    for (int plane = 0; plane < format->numPlanes; ++plane) {
        if (!processPlane[plane]) {
            continue;
        }

        if (format->sampleType == stInteger) {
            if (format->bytesPerSample == 1) {
                blendPlaneInteger<uint8_t>(destinationFrame, sourceFrames, weights, sourceCount, plane, 8, vsapi);
            } else if (format->bytesPerSample == 2) {
                blendPlaneInteger<uint16_t>(destinationFrame, sourceFrames, weights, sourceCount, plane, format->bitsPerSample, vsapi);
            }
        } else if (format->sampleType == stFloat && format->bytesPerSample == 4) {
            blendPlaneFloat(destinationFrame, sourceFrames, weights, sourceCount, plane, vsapi);
        }
    }
}

static const VSFrame *VS_CC frameBlendGetFrame(
    int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core,
    const VSAPI *vsapi
) {
    auto *filterData = static_cast<FrameBlendData *>(instanceData);
    const int totalWeights = static_cast<int>(filterData->weightPercents.size());
    const int windowRadius = totalWeights / 2;
    const int maxFrameIndex = (filterData->videoInfo->numFrames > 0) ? (filterData->videoInfo->numFrames - 1) : (INT_MAX - 1);

    const FrameWindow window = calculateFrameWindow(n, windowRadius, maxFrameIndex);

    if (activationReason == arInitial) {
        requestFrameRange(window.firstFrame, window.lastFrame, filterData->node, frameCtx, vsapi);
        return nullptr;
    }

    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const int uniqueCount = window.lastFrame - window.firstFrame + 1;
    std::vector<const VSFrame *> uniqueFrames(uniqueCount);
    for (int i = 0; i < uniqueCount; ++i) {
        uniqueFrames[i] = vsapi->getFrameFilter(window.firstFrame + i, filterData->node, frameCtx);
    }
    FrameListGuard framesGuard{ uniqueFrames, vsapi };

    std::vector<float> uniqueWeights(uniqueCount, 0.0f);
    collapseWindowWeights(filterData->weightPercents, n, windowRadius, window.firstFrame, maxFrameIndex, uniqueWeights);

    const VSFrame *centerFrame = uniqueFrames[window.centerIndex];
    VSFrame *destinationFrame = createDestinationFrame(centerFrame, filterData->processPlane, core, vsapi);
    if (!destinationFrame) {
        vsapi->setFilterError("FrameBlend: failed to allocate new destination frame", frameCtx);
        return nullptr;
    }

    std::vector<const VSFrame *> activeFrames;
    std::vector<float> activeWeights;
    activeFrames.reserve(uniqueCount);
    activeWeights.reserve(uniqueCount);

    for (int i = 0; i < uniqueCount; ++i) {
        if (std::abs(uniqueWeights[i]) > 1e-7f) {
            activeFrames.push_back(uniqueFrames[i]);
            activeWeights.push_back(uniqueWeights[i]);
        }
    }

    dispatchFrameBlending(
        destinationFrame,
        activeFrames.data(),
        activeWeights.data(),
        activeFrames.size(),
        filterData->processPlane,
        vsapi
    );

    return destinationFrame;
}

static void VS_CC frameBlendFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    auto *filterData = static_cast<FrameBlendData *>(instanceData);
    if (filterData) {
        if (filterData->node) {
            vsapi->freeNode(filterData->node);
        }
        delete filterData;
    }
}

static bool validateVideoFormat(const VSVideoInfo *videoInfo, VSMap *out, const VSAPI *vsapi) {
    if (!vsh::isConstantVideoFormat(videoInfo)) {
        vsapi->mapSetError(out, "FrameBlend: only clips with constant video format and dimensions are supported");
        return false;
    }

    const auto &format = videoInfo->format;
    if (format.sampleType != stInteger && format.sampleType != stFloat) {
        vsapi->mapSetError(out, "FrameBlend: unsupported sample type");
        return false;
    }

    if (format.sampleType == stInteger && (format.bytesPerSample != 1 && format.bytesPerSample != 2)) {
        vsapi->mapSetError(out, "FrameBlend: integer formats must be 8-bit or 9-16 bit");
        return false;
    }

    if (format.sampleType == stFloat && format.bytesPerSample != 4) {
        vsapi->mapSetError(out, "FrameBlend: float formats must be 32-bit");
        return false;
    }

    return true;
}

static std::optional<std::vector<float>> parseNormalizedWeights(const VSMap *in, VSMap *out, const VSAPI *vsapi) {
    const int count = vsapi->mapNumElements(in, "weights");
    if (count <= 0 || (count % 2) != 1) {
        vsapi->mapSetError(out, "FrameBlend: number of weights must be odd and greater than 0");
        return std::nullopt;
    }

    std::vector<float> weights(count);
    double total = 0.0;

    for (int i = 0; i < count; ++i) {
        int error = 0;
        weights[i] = static_cast<float>(vsapi->mapGetFloat(in, "weights", i, &error));
        if (error) {
            vsapi->mapSetError(out, "FrameBlend: failed to read weight");
            return std::nullopt;
        }
        total += weights[i];
    }

    if (std::abs(total) < 1e-6 || !std::isfinite(total)) {
        vsapi->mapSetError(out, "FrameBlend: total sum of weights cannot be zero or non-finite");
        return std::nullopt;
    }

    for (int i = 0; i < count; ++i) {
        weights[i] = static_cast<float>(weights[i] / total);
    }

    return weights;
}

static bool parsePlaneSelection(const VSMap *in, int availablePlanes, bool processPlane[3], VSMap *out, const VSAPI *vsapi) {
    const int count = vsapi->mapNumElements(in, "planes");
    if (count <= 0) {
        for (int i = 0; i < 3; ++i) {
            processPlane[i] = (i < availablePlanes);
        }
        return true;
    }

    for (int i = 0; i < 3; ++i) {
        processPlane[i] = false;
    }

    for (int i = 0; i < count; ++i) {
        int error = 0;
        int plane = vsapi->mapGetIntSaturated(in, "planes", i, &error);
        if (error || plane < 0 || plane >= availablePlanes) {
            vsapi->mapSetError(out, "FrameBlend: plane index out of range");
            return false;
        }
        processPlane[plane] = true;
    }

    return true;
}

static void VS_CC frameBlendCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    int error = 0;
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, &error);
    if (error || !node) {
        vsapi->mapSetError(out, "FrameBlend: clip is not a valid VideoNode");
        return;
    }
    NodeScopeGuard nodeGuard{ node, vsapi };

    const VSVideoInfo *videoInfo = vsapi->getVideoInfo(node);
    if (!validateVideoFormat(videoInfo, out, vsapi)) {
        return;
    }

    auto weights = parseNormalizedWeights(in, out, vsapi);
    if (!weights) {
        return;
    }

    bool processPlane[3] = { false, false, false };
    if (!parsePlaneSelection(in, videoInfo->format.numPlanes, processPlane, out, vsapi)) {
        return;
    }

    auto filterData = std::make_unique<FrameBlendData>();
    filterData->node = nodeGuard.release();
    filterData->videoInfo = videoInfo;
    filterData->weightPercents = std::move(*weights);
    std::memcpy(filterData->processPlane, processPlane, sizeof(filterData->processPlane));

    VSFilterDependency dependencies[] = {
        { filterData->node, rpGeneral }
    };

    vsapi->createVideoFilter(
        out,
        "FrameBlend",
        filterData->videoInfo,
        frameBlendGetFrame,
        frameBlendFree,
        fmParallelRequests,
        dependencies,
        1,
        filterData.release(),
        core
    );
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "com.vapoursynth.frameblender",
        "frameblender",
        "Frame blender",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );
    vspapi->registerFunction(
        "FrameBlend",
        "clip:vnode;weights:float[];log:int:opt;planes:int[]:opt;",
        "clip:vnode;",
        frameBlendCreate,
        nullptr,
        plugin
    );
}
