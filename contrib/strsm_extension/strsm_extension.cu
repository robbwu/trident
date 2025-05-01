#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// Helper function to check cuBLAS status
#define CUBLAS_CHECK(status) \
do { \
    cublasStatus_t cublasStat = (status); \
    if (cublasStat != CUBLAS_STATUS_SUCCESS) { \
        AT_ERROR("cuBLAS error: ", cublasGetStatusString(cublasStat), " ", __FILE__, ":", __LINE__); \
    } \
} while (0)

// Helper function to check CUDA status
#define CUDA_CHECK(status) \
do { \
    cudaError_t cudaStat = (status); \
    if (cudaStat != cudaSuccess) { \
        AT_ERROR("CUDA error: ", cudaGetErrorString(cudaStat), " ", __FILE__, ":", __LINE__); \
    } \
} while (0)


// Our C++ function that will be callable from Python
// Solves AX = alpha * B where A is lower triangular (non-unit)
// B is overwritten with the solution X
torch::Tensor cublas_strsm_wrapper(
    torch::Tensor A,      // Triangular matrix (m x m)
    torch::Tensor B,      // Right-hand side matrix (m x n), overwritten by solution X
    float alpha,          // Scalar alpha
    bool left_side,       // If true, solve op(A)X = alpha*B, else X*op(A) = alpha*B
    bool upper,           // If true, A is upper triangular, else lower
    bool transpose_a,     // If true, use op(A) = A^T, else op(A) = A
    bool unit_diagonal    // If true, A has unit diagonal
)
{
    // --- Input Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Input tensor A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "Input tensor B must be float32");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2D");

    // For simplicity, assume contiguous or handle strides carefully
    // cuBLAS expects column-major by default. PyTorch is row-major.
    // If tensors are not Fortran contiguous, strides need careful handling or copy.
    // Let's enforce Fortran contiguity for simplicity here, although this is restrictive.
    // A more robust implementation handles strides.
    // TORCH_CHECK(A.is_contiguous(at::MemoryFormat::Contiguous), "A must be contiguous");
    // TORCH_CHECK(B.is_contiguous(at::MemoryFormat::Contiguous), "B must be contiguous");
     // Better: check if Fortran contiguous (column-major)
    TORCH_CHECK(A.t().is_contiguous(), "A must be Fortran contiguous (use A.t().contiguous().t() if needed)");
    TORCH_CHECK(B.t().is_contiguous(), "B must be Fortran contiguous (use B.t().contiguous().t() if needed)");


    int m = left_side ? A.size(0) : B.size(0);
    int n = left_side ? B.size(1) : B.size(1); // Should be B.size(1) regardless
    int k = left_side ? A.size(1) : A.size(0); // Dimension of A

    TORCH_CHECK( (left_side ? A.size(1) : A.size(0)) == k, "A dimension mismatch");
    TORCH_CHECK( (left_side ? B.size(0) : B.size(0)) == m, "B dimension mismatch"); // B.size(0) should be m

    if (left_side) {
        TORCH_CHECK(A.size(0) == A.size(1), "For left_side=True, A must be square");
        TORCH_CHECK(A.size(0) == B.size(0), "For left_side=True, A.size(0) must equal B.size(0)");
        m = A.size(0);
        n = B.size(1);
    } else { // right_side
        TORCH_CHECK(A.size(0) == A.size(1), "For left_side=False, A must be square");
        TORCH_CHECK(A.size(0) == B.size(1), "For left_side=False, A.size(0) must equal B.size(1)");
        m = B.size(0);
        n = A.size(0);
    }


    // --- cuBLAS Setup ---
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(); // Get stream object

    // If cublasSetStream requires a raw cudaStream_t:
    CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));


    // --- Prepare cuBLAS Arguments ---
    // Note: PyTorch tensors are row-major, cuBLAS assumes column-major by default.
    // If we pass data_ptr directly, we need to account for this, often by
    // swapping roles of m/n, transposing operations, or ensuring Fortran contiguity.
    // Here, assuming Fortran contiguous input simplifies lda/ldb.
    int lda = A.stride(1); // Leading dimension of A (Fortran)
    int ldb = B.stride(1); // Leading dimension of B (Fortran)

    cublasSideMode_t side = left_side ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    // If input A was row-major C-contiguous, and we want to compute op(A) * X,
    // this is equivalent to X^T * op(A)^T in column-major. We need to adjust arguments.
    // But since we required Fortran contiguity, we can map more directly.
    cublasOperation_t trans = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasDiagType_t diag = unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    const float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>(); // B is input and output

    // --- Call cublasStrsm ---
    CUBLAS_CHECK(cublasStrsm(handle,
                             side, uplo, trans, diag,
                             m, n,              // Dimensions of B (output X)
                             &alpha,            // Scalar alpha
                             A_ptr, lda,        // Matrix A and its leading dimension
                             B_ptr, ldb));      // Matrix B (in/out) and its leading dimension

    // --- Cleanup ---
    CUBLAS_CHECK(cublasDestroy(handle));

    // B has been modified in-place, return it
    return B;
}

// --- Pybind11 Bindings ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("strsm", &cublas_strsm_wrapper, "Wrapper around cublasStrsm",
        pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("alpha") = 1.0f,
        pybind11::arg("left_side") = true, pybind11::arg("upper") = false,
        pybind11::arg("transpose_a") = false, pybind11::arg("unit_diagonal") = false);
}
