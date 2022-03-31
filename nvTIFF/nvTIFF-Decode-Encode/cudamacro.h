#ifndef __CUDA_MACRO_H__
#define __CUDA_MACRO_H__

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_ERROR(errorMessage) {                                          \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }}

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors4[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors4 = sizeof(colors4)/sizeof(colors4[0]);

#define START_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors4;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors4[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define END_RANGE { \
        nvtxRangePop(); \
}
#else
#define START_RANGE(name,cid)
#define END_RANGE
#endif

#endif
