#ifndef CUBLASDX_FLOPS_H
#define CUBLASDX_FLOPS_H

/*flop count is copied from magma - flop.h*/

#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

#define FLOPS_ZGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_CGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_SGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )

#define FMULS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))


#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FADDS_TRMM
#define FMULS_TRMM(side_, m_, n_) ( ( (side_) == 'L' ) ? FMULS_TRMM_2((m_), (n_)) : FMULS_TRMM_2((n_), (m_)) )
#define FADDS_TRMM(side_, m_, n_) ( ( (side_) == 'L' ) ? FADDS_TRMM_2((m_), (n_)) : FADDS_TRMM_2((n_), (m_)) )

#define FLOPS_ZTRSM(side_, m_, n_) (6. * FMULS_TRSM(side_, (double)(m_), (double)(n_)) + 2.0 * FADDS_TRSM(side_, (double)(m_), (double)(n_)) )
#define FLOPS_CTRSM(side_, m_, n_) (6. * FMULS_TRSM(side_, (double)(m_), (double)(n_)) + 2.0 * FADDS_TRSM(side_, (double)(m_), (double)(n_)) )
#define FLOPS_DTRSM(side_, m_, n_) (     FMULS_TRSM(side_, (double)(m_), (double)(n_)) +       FADDS_TRSM(side_, (double)(m_), (double)(n_)) )
#define FLOPS_STRSM(side_, m_, n_) (     FMULS_TRSM(side_, (double)(m_), (double)(n_)) +       FADDS_TRSM(side_, (double)(m_), (double)(n_)) )


namespace example {
    template<class T>
    double cublasDxXgemm_flop([[maybe_unused]] int M, [[maybe_unused]] int N, [[maybe_unused]] int K) {
        return 0.0;
    }

    template<>
    double cublasDxXgemm_flop<float>(int M, int N, int K) {
        return (1.0 * FLOPS_SGEMM(M, N, K));
    }
 
    template<>
    double cublasDxXgemm_flop<double>(int M, int N, int K) {
        return (1.0 * FLOPS_DGEMM(M, N, K));
    }

    template<>
    double cublasDxXgemm_flop<cuComplex>(int M, int N, int K) {
        return (1.0 * FLOPS_CGEMM(M, N, K));
    }

    template<>
    double cublasDxXgemm_flop<cuDoubleComplex>(int M, int N, int K) {
        return (1.0 * FLOPS_ZGEMM(M, N, K));
    }

    template<class T>
    double cublasDxXtrsm_flop([[maybe_unused]] char side, [[maybe_unused]] int M, [[maybe_unused]] int N) {
        return 0.0;
    }

    template<>
    double cublasDxXtrsm_flop<float>(char side, int M, int N) {
        return (1.0 * FLOPS_STRSM(side, M, N));
    }
 
    template<>
    double cublasDxXtrsm_flop<double>(char side, int M, int N) {
        return (1.0 * FLOPS_DTRSM(side, M, N));
    }

    template<>
    double cublasDxXtrsm_flop<cuComplex>(char side, int M, int N) {
        return (1.0 * FLOPS_CTRSM(side, M, N));
    }

    template<>
    double cublasDxXtrsm_flop<cuDoubleComplex>(char side, int M, int N) {
        return (1.0 * FLOPS_ZTRSM(side, M, N));
    }

}


#endif /* CUBLASDX_FLOPS_H */

