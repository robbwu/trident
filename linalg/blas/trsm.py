# profile:
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/extras/CUPTI/lib64/ python3  trsm.py
# proton-viewer -m time/ms trsm.hatchet
# Sample output:
# 0.401 ROOT
# └─ 0.401 /home/worker/trident/linalg/blas/trsm.py:<module>@40
#    ├─ 0.297 _Z19kernel_trsm_l_mul32IfLi8ELb1ELb1ELb0ELb0EEviiPKT_S2_iPS0_iS0_i
#    ├─ 0.022 _ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_
#    ├─ 0.046 ampere_sgemm_128x64_tn
#    └─ 0.036 ampere_sgemm_64x32_sliced1x4_tn



import torch
import triton
import triton.language as tl
import triton.profiler as proton

# lower triangular solve, solve X from equation L * X = Y
# L: Input, lower triangular matrix of shape (m, m)
# Y: Input/Output, matrix of shape (m, n). The result X will overwrite Y
def trsm_driver(L, Y, m, n):
    return None


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False  # Enable TF32 for GEMMs
    # torch.backends.cublas.allow_fp16_reduced_precision_reduction = False
    session_id = proton.start(name="trsm", context="python")
    proton.deactivate(session_id)

    n = 2048
    m = 1024
    A = torch.randn(m, m, device='cuda', dtype=torch.float32)
    LU, _ = torch.linalg.lu_factor(A)
    L = torch.tril(LU)
    L.diagonal().fill_(1.0)  # ensure diagonal is 1, which is what LU factorization assumes
    cond_L = torch.linalg.cond(L)
    print(f"condition number of L after adjustment: {cond_L.item():e}")
    Y_original = torch.randn(m, n, device='cuda', dtype=torch.float32)
    norm_Y = torch.linalg.matrix_norm(Y_original, ord=2)

    def check_backward_error(L, XX, Y_expected, Y_norm):
        # check the backward error: meaning ||L * Y - X||
        Y_computed = torch.matmul(L, XX)
        # relative backward error
        backward_error = torch.linalg.matrix_norm(Y_computed - Y_expected, ord=2) / norm_Y / Y_expected.shape[0]
        print("Backward error:", backward_error.item())
        # assert backward_error.item() < 1e-6, "The backward error is too large!"

    X = Y_original.clone()
    trsm_driver(L, X, m, n)
    check_backward_error(L, X, Y_original, norm_Y)
    proton.activate(session_id)
    with proton.scope(f"torch.linalg.solve_triangular_{m}_{n}",metrics={
                        "flops": m*m*n,
                        "bytes": (m*m/2 + m*n) * L.element_size(),
                    }):
        X2 = torch.linalg.solve_triangular(L, Y_original, upper=False)
    proton.deactivate(session_id)
    check_backward_error(L, X2, Y_original, norm_Y)

    # directly call cublas
    import sys
    import os
    script_dir = os.path.dirname(__file__)
    build_dir = os.path.abspath(os.path.join(script_dir, '../../contrib/strsm_extension'))
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import  strsm_extension
    LL = L.t().contiguous().t()
    X = Y_original.clone().t().contiguous().t()
    proton.activate(session_id)
    with proton.scope(f"cublas_strsm_{m}_{n}",metrics={
                        "flops": m*m*n,
                        "bytes": (m*m/2 + m*n) * L.element_size(),}):
        strsm_extension.strsm(LL, X)
    proton.finalize()

    check_backward_error(L, X, Y_original, norm_Y)
