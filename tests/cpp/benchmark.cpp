
#include "../../hnswlib/hnswlib.h"
#include <thread>
#include "opt_dist_128v.h"

class StopW {
    std::chrono::steady_clock::time_point time_begin;

public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};


#define EPSILON (0.0005f)
#define is_not_equal(x,y) (    ( fabs((x)*(x) - (y)*(y))  > EPSILON  ) ? 1 : 0   )
#define is_not_equal_error(x,y)  ( fabs((x)*(x) - (y)*(y))  - EPSILON  )



inline void prefetch_128x32(const void * p)
{
    _mm_prefetch(((char *)(p)) + (64 * 0), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 1), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 2), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 3), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 4), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 5), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 6), _MM_HINT_T0);
    _mm_prefetch(((char *)(p)) + (64 * 8), _MM_HINT_T0);
}


inline void prefetch_128x32_l3(const void * p)
{
    _mm_prefetch(((char *)(p)) + (64 * 0), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 1), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 2), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 3), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 4), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 5), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 6), _MM_HINT_NTA);
    _mm_prefetch(((char *)(p)) + (64 * 8), _MM_HINT_NTA);
}


const int dim = 128;               // Dimension of the elements
const int max_elements = 1000000;   // Maximum number of elements, should be known beforehand

__attribute__((aligned(512))) float aligned_v1[max_elements * dim];
__attribute__((aligned(512))) float aligned_v2[max_elements * dim];


__attribute__((aligned(512))) float unaligned_v1[max_elements * dim + 64];
__attribute__((aligned(512))) float unaligned_v2[max_elements * dim + 64];



int aligned_opt_test(const float* v1, const float* v2, float* res,
                     const int* ur_index, int max_elements, int dim);

int opt_test(const float* v1, const float* v2, float* res,
             const int* ur_index, int max_elements, int dim);

int baseline(const float* v1, const float* v2, float* res,
             const int* ur_index, int max_elements, int dim);



int main() {



#if defined(USE_SSE)
    printf("use SSE\n");
#endif
#if defined(USE_AVX)
    printf("use AVX\n");
#endif
#if defined(USE_AVX512)
    printf("use AVX512\n");
#endif

    // Initing index
    hnswlib::L2Space space(dim);
    void *dist_func_param_{nullptr};
    dist_func_param_ = space.get_dist_func_param();
    hnswlib::DISTFUNC<float> fstdistfunc = space.get_dist_func();

    StopW e2e;
    float time_us = 0;

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* v1 = nullptr;
    float* v2 = nullptr;

    float* res = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        float tmp = distrib_real(rng);
        aligned_v1[i] = tmp;
        unaligned_v1[i + 1] = tmp;
    }

    for (int i = 0; i < dim * max_elements; i++) {
        float tmp = distrib_real(rng);
        aligned_v2[i] = tmp;
        unaligned_v2[i + 1] = tmp;
    }

    std::uniform_int_distribution<> distrib_ui(0, max_elements);

    int* ur_index = new int[max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        ur_index[i] = distrib_ui(rng);
    }


    printf("========== unaligned ===========\n");
    v1 = &unaligned_v1[1];
    v2 = &unaligned_v2[1];

    baseline(v1, v2, res, ur_index, max_elements, dim);
    printf("========== OPT ==============================================\n");
    opt_test(v1, v2, res, ur_index, max_elements, dim);




    printf("\n\n\n========== aligned ===========\n");
    v1 = &aligned_v1[0];
    v2 = &aligned_v2[0];

    baseline(v1, v2, res, ur_index, max_elements, dim);
    printf("========== OPT ==============================================\n");
    opt_test(v1, v2, res, ur_index, max_elements, dim);
    printf("========== Aligned AVX instruction ==========================\n");
    aligned_opt_test(v1, v2, res, ur_index, max_elements, dim);




#if 0
// only work for seq accces

    for (int i = 0;  i < max_elements; i ++)
    {
        res[i] = 0;
    }

    e2e.reset();
    //prefetch v1:
    prefetch_128x32((void *) (v1));
    for (int k = 0; k < dim / 16; k ++ )
    {
        __m512 m_v1 = _mm512_loadu_ps(v1 + k * 16);
        for (int i = 0; i < max_elements; i++) {
            int index_current = ur_index[i];
            _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
            int index_n1 = ur_index[i + 1];
            prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
            int index_n2 = ur_index[i + 3]; // jump 2
            prefetch_128x32((void *) (v2 +  (index_n2 * dim)));


            __m512 diff, tv2;
            __m512 sum = _mm512_set1_ps(0);

            tv2 = _mm512_loadu_ps(v2 + (index_current * dim + k * 16));
            diff = _mm512_sub_ps(m_v1, tv2);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));

            float temp_res = _mm512_reduce_add_ps(sum);


            res[i] += temp_res;
        }
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("full opt: exe time %f us\n", time_us);





#endif


    printf("vaild data\n");
    int error_count  = 0;
    for (int i = 0; i < max_elements; i++) {
        float golden = 0.0f;
        int index_current = ur_index[i];
        for (int j = 0; j < dim; j ++)
        {
            float *p_v1 = v1;
            float *p_v2 = v2 + index_current * dim;
            float inter = (p_v1[j] - p_v2[j]);
            golden += inter * inter;
        }
        if (is_not_equal (res[i], golden))
        {
            error_count ++;
            //printf("[%d] ne calculated: %f golden: %f diff:%f\n", i, res[i], golden, is_not_equal_error(res[i], golden));
        }

    }
    printf("error count %d\n", error_count);




    return 0;
}


int baseline(const float* v1, const float* v2, float* res, const int* ur_index, int max_elements, int dim)
{

    hnswlib::L2Space space(dim);
    void *dist_func_param_{nullptr};
    dist_func_param_ = space.get_dist_func_param();
    hnswlib::DISTFUNC<float> fstdistfunc = space.get_dist_func();


    StopW e2e;
    float time_us = 0;


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        res[i] = fstdistfunc(v1 + i * dim, v2 + i * dim, dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("v1 sequenial, v2 sequenial: exe time %f us\n", time_us);

    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        res[i] = fstdistfunc(v1 + i * dim, v2 + index_current * dim, dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("v1 sequenial, v2 random: exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        res[i] = fstdistfunc(v1 , v2 + i * dim, dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("v1 same, v2 sequenial: exe time %f us\n", time_us);

    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        res[i] = fstdistfunc(v1 , v2 + index_current * dim, dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("v1 same, v2 random: exe time %f us\n", time_us);
    return 0;




}




int opt_test(const float* v1, const float* v2, float* res, const int* ur_index, int max_elements, int dim)
{

    hnswlib::L2Space space(dim);
    void *dist_func_param_{nullptr};
    dist_func_param_ = space.get_dist_func_param();
    hnswlib::DISTFUNC<float> fstdistfunc = space.get_dist_func();


    StopW e2e;
    float time_us = 0;

    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        int index = ur_index[i + 1];
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 0), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 1), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 2), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 3), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 4), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 5), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 6), _MM_HINT_T0);
        _mm_prefetch((char *) (v2 +  (index * dim)) + (64 * 8), _MM_HINT_T0);


        res[i] = fstdistfunc(v1 , v2 + index_current * dim, dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("prefetch 128 float: exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));


        res[i] = ori_L2SqrSIMD16ExtAVX512(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch: exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));


        res[i] = reduce_L2SqrSIMD16ExtAVX512(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with add reduce : exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_L2SqrSIMD16ExtAVX512(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with add reduce & NTA index: exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_ma_L2SqrSIMD16ExtAVX512(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with madd reduce & NTA index: exe time %f us\n", time_us);



    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_ma_L2SqrSIMD16ExtAVX512_128(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with madd reduce & NTA index & manually unroll: exe time %f us\n", time_us);



    return 0;

}



int aligned_opt_test(const float* v1, const float* v2, float* res, const int* ur_index, int max_elements, int dim)
{

    hnswlib::L2Space space(dim);
    void *dist_func_param_{nullptr};
    dist_func_param_ = space.get_dist_func_param();
    hnswlib::DISTFUNC<float> fstdistfunc = space.get_dist_func();


    StopW e2e;
    float time_us = 0;


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));


        res[i] = reduce_L2SqrSIMD16ExtAVX512_aligned(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with add reduce : exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_L2SqrSIMD16ExtAVX512_aligned(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with add reduce & NTA index: exe time %f us\n", time_us);


    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_ma_L2SqrSIMD16ExtAVX512_aligned(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with madd reduce & NTA index: exe time %f us\n", time_us);



    e2e.reset();
    for (int i = 0; i < max_elements; i++) {
        int index_current = ur_index[i];
        _mm_prefetch((char *) (ur_index + 1), _MM_HINT_NTA);
        int index_n1 = ur_index[i + 1];
        prefetch_128x32((void *) (v2 +  (index_n1 * dim)));
        int index_n2 = ur_index[i + 3]; // jump 2
        prefetch_128x32((void *) (v2 +  (index_n2 * dim)));

        res[i] = reduce_ma_L2SqrSIMD16ExtAVX512_128_aligned(v1 , v2 + (index_current * dim), dist_func_param_);
    }
    time_us = e2e.getElapsedTimeMicro();
    printf("jump 2 prefetch with madd reduce & NTA index & manually unroll: exe time %f us\n", time_us);


    return 0;

}
