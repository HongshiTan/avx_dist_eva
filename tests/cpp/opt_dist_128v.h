

float ori_L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

float reduce_L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    float res = _mm512_reduce_add_ps(sum);

    return (res);
}

float reduce_ma_L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        //sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    float res = _mm512_reduce_add_ps(sum);

    return (res);
}


float reduce_ma_L2SqrSIMD16ExtAVX512_128(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;

    __m512 diff, v1, v2;

    __m512 sum1 = _mm512_set1_ps(0);
    __m512 sum2;

    v1 = _mm512_loadu_ps(pVect1 + 16 * 0);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 0);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 1);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 1);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 2);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 2);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 3);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 3);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 4);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 4);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 5);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 5);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 6);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 6);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_loadu_ps(pVect1 + 16 * 7);
    v2 = _mm512_loadu_ps(pVect2 + 16 * 7);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);


    return _mm512_reduce_add_ps(sum1);


}





float reduce_L2SqrSIMD16ExtAVX512_aligned(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_load_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_load_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    float res = _mm512_reduce_add_ps(sum);

    return (res);
}

float reduce_ma_L2SqrSIMD16ExtAVX512_aligned(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_load_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_load_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        //sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    float res = _mm512_reduce_add_ps(sum);

    return (res);
}


float reduce_ma_L2SqrSIMD16ExtAVX512_128_aligned(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;

    __m512 diff, v1, v2;

    __m512 sum1 = _mm512_set1_ps(0);
    __m512 sum2;

    v1 = _mm512_load_ps(pVect1 + 16 * 0);
    v2 = _mm512_load_ps(pVect2 + 16 * 0);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_load_ps(pVect1 + 16 * 1);
    v2 = _mm512_load_ps(pVect2 + 16 * 1);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_load_ps(pVect1 + 16 * 2);
    v2 = _mm512_load_ps(pVect2 + 16 * 2);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_load_ps(pVect1 + 16 * 3);
    v2 = _mm512_load_ps(pVect2 + 16 * 3);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_load_ps(pVect1 + 16 * 4);
    v2 = _mm512_load_ps(pVect2 + 16 * 4);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_load_ps(pVect1 + 16 * 5);
    v2 = _mm512_load_ps(pVect2 + 16 * 5);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);

    v1 = _mm512_load_ps(pVect1 + 16 * 6);
    v2 = _mm512_load_ps(pVect2 + 16 * 6);
    diff = _mm512_sub_ps(v1, v2);

    sum2= _mm512_fmadd_ps(diff, diff, sum1);

    v1 = _mm512_load_ps(pVect1 + 16 * 7);
    v2 = _mm512_load_ps(pVect2 + 16 * 7);
    diff = _mm512_sub_ps(v1, v2);

    sum1= _mm512_fmadd_ps(diff, diff, sum2);


    return _mm512_reduce_add_ps(sum1);


}

