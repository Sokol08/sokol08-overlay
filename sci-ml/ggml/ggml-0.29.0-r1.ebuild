# Copyright 2026 Gentoo Authors
# Copyright 2026 Sokol
# Distributed under the terms of the GNU General Public License v2

# Fork of sci-ml/ggml-0.29.0 from ::gentoo.
#
# Root cause of the stock 0.29.0 configure FAIL_ON this box:
# /usr/bin/hipconfig (from dev-util/hipcc, the CUDA flavor) reports
# "HIP_PLATFORM: nvidia" and "HIP_COMPILER: nvcc" because
# /opt/therock-bin/bin is not on the build PATH. CMake's
# CMakeDetermineHIPCompiler honors that output: it takes the NVIDIA
# branch, picks /opt/cuda/bin/nvcc, and then
# cmake_cuda_architectures_validate() checks CMAKE_HIP_ARCHITECTURES
# against a numeric-only CUDA regex and dies on "gfx1100". The AMD
# branch performs no such check and accepts gfx targets as-is.
#
# So in the rocm branch we force the AMD path:
# - CMAKE_HIP_PLATFORM=amd (skips the hipconfig dispatch),
# - CMAKE_HIP_COMPILER pinned to the AMD clang 24 from /opt/therock-bin,
# - CMAKE_HIP_COMPILER_ROCM_ROOT stated explicitly (skip the builtin
#   "Found HIP installation" / hipconfig probes entirely),
# - a clean CMAKE_HIP_ARCHITECTURES: rocm.eclass appends ";" to every
#   AMDGPU_TARGETS element, and the trailing one leaves an empty CMake
#   list element; GPU_TARGETS is kept for toolchains that consume it,
# - the host C/CXX compiler stays the system toolchain (gcc): the
#   standard ROCm practice is gcc for host code and clang for the HIP
#   device code, and forcing amdclang as C/CXX breaks here because the
#   system CFLAGS contain gcc-only switches (e.g. -fvect-cost-model=dynamic),
#   - CMAKE_HIP_FLAGS is set explicitly to a clang-compatible subset of
#   CXXFLAGS, because CMake initializes it from CMAKE_CXX_FLAGS and
#   amdclang++ falls over on the gcc-only ones mentioned above.
#
# The GGML Hips CMake (src/ggml-hip/CMakeLists.txt) forwards
# GPU_TARGETS/CMAKE_HIP_ARCHITECTURES to enable_language(HIP) and
# never overrides a CMAKE_HIP_ARCHITECTURES we pre-set.

EAPI=8

PYTHON_COMPAT=( python3_{12..15} )
ROCM_VERSION=7.2
inherit cuda cmake python-any-r1 rocm toolchain-funcs flag-o-matic

DESCRIPTION="Tensor library for machine learning"
HOMEPAGE="https://ggml.ai/"
SRC_URI="https://github.com/ggml-org/${PN}/archive/refs/tags/v${PV}.tar.gz
	-> ${P}.tar.gz"

LICENSE="MIT"
SLOT="0/${PV}"
KEYWORDS="~amd64 ~arm64"

X86_CPU_FLAGS=(
	amx_bf16
	amx_int8
	amx_tile
	avx
	avx_vnni
	avx2
	avx512_bf16
	avx512bw
	avx512f
	avx512vbmi
	avx512_vnni
	bmi2
	fma3
	f16c
	sse4_2
)
CPU_FLAGS=( "${X86_CPU_FLAGS[@]/#/cpu_flags_x86_}" )
IUSE="${CPU_FLAGS[*]} cuda opencl openmp rocm test vulkan"

REQUIRED_USE="rocm? ( ${ROCM_REQUIRED_USE} )"

RESTRICT="!test? ( test )"

# Should be >=sci-libs/hipBLAS-${ROCM_VERSION}[${ROCM_USEDEP}]
# But pkgcheck can't elaborate that
RDEPEND="
	cuda? (
		dev-util/nvidia-cuda-toolkit:=
	)
	opencl? (
		dev-libs/opencl-icd-loader
	)
	vulkan? ( media-libs/vulkan-loader )
	rocm? (
		>=dev-util/hip-${ROCM_VERSION}:=
		>=sci-libs/hipBLAS-${ROCM_VERSION}:=
	)
"
DEPEND="${RDEPEND}
	opencl? ( dev-util/opencl-headers )
	vulkan? ( dev-util/vulkan-headers )
"
BDEPEND="
	opencl? ( ${PYTHON_DEPS} )
	vulkan? ( media-libs/shaderc )
"

DOCS=( AUTHORS CONTRIBUTING.md docs/gguf.md README.md )

PATCHES=(
	"${FILESDIR}"/${PN}-0.28.0-llvm-23.patch
)

pkg_pretend() {
	[[ ${MERGE_TYPE} != binary ]] && use openmp && tc-check-openmp
}

pkg_setup() {
	if use opencl; then
		python-any-r1_pkg_setup
	fi

	[[ ${MERGE_TYPE} != binary ]] && use openmp && tc-check-openmp
}

src_prepare() {
	cmake_src_prepare

	if use cuda; then
		cuda_src_prepare
		filter-lto #bug 983121
	fi
}

src_configure() {
	local mycmakeargs=(
		-DGGML_BACKEND_DL=OFF
		-DGGML_BUILD_EXAMPLES=OFF
		-DGGML_NATIVE=OFF
		-DGGML_LTO=OFF
		-DGGML_HIP_MMQ_MFMA=OFF
		-DGGML_CCACHE=OFF

		# CPU Flags
		-DGGML_AMX_BF16=$(usex cpu_flags_x86_amx_bf16)
		-DGGML_AMX_INT8=$(usex cpu_flags_x86_amx_int8)
		-DGGML_AMX_TILE=$(usex cpu_flags_x86_amx_tile)
		-DGGML_AVX=$(usex cpu_flags_x86_avx)
		-DGGML_AVX_VNNI=$(usex cpu_flags_x86_avx_vnni)
		-DGGML_AVX2=$(usex cpu_flags_x86_avx2)
		-DGGML_AVX512_BF16=$(usex cpu_flags_x86_avx512_bf16)
		-DGGML_AVX512_VBMI=$(usex cpu_flags_x86_avx512vbmi)
		-DGGML_AVX512_VNNI=$(usex cpu_flags_x86_avx512_vnni)
		-DGGML_BMI2=$(usex cpu_flags_x86_bmi2)
		-DGGML_FMA=$(usex cpu_flags_x86_fma3)
		-DGGML_F16C=$(usex cpu_flags_x86_f16c)
		-DGGML_SSE42=$(usex cpu_flags_x86_sse4_2)

		-DGGML_CUDA=$(usex cuda)
		-DGGML_HIP=$(usex rocm)
		-DGGML_OPENCL=$(usex opencl)
		-DGGML_OPENMP=$(usex openmp)
		-DGGML_VULKAN=$(usex vulkan)

		-DGGML_BUILD_TESTS=$(usex test)
	)

	# Enable AVX512 if ANY of the avx512 flags are present
	if use cpu_flags_x86_avx512f || use cpu_flags_x86_avx512bw; then
		mycmakeargs+=( -DGGML_AVX512=ON )
	else
		mycmakeargs+=( -DGGML_AVX512=OFF )
	fi

	if use cuda; then
		cuda_add_sandbox -w
		addpredict "/dev/char/"
		cuda_sanitize
		mycmakeargs+=(
			-DCMAKE_CUDA_FLAGS="${NVCCFLAGS}"
		)
	fi

	if use rocm; then
		local rocm_root="${ROCM_PATH:-/opt/therock-bin}"
		export ROCM_PATH="${rocm_root}"
		export HIP_PATH="${rocm_root}"

		# rocm.eclass tags every target with ";" (gfx1100;); join the
		# elements back into one list and drop the trailing separator
		# so CMake does not see an empty element.
		local hips_archs="${AMDGPU_TARGETS[*]/; /;}"
		hips_archs="${hips_archs%;}"
		[[ -n ${hips_archs} ]] || die "amdgpu_targets USE flags are empty but the rocm USE flag requires amdgpu targets"

		# CMake initializes CMAKE_HIP_FLAGS from CMAKE_CXX_FLAGS, so the
		# HIP device compiler would inherit the full system CXXFLAGS and
		# abort on gcc-only switches (the profile's COMMON_FLAGS carries
		# -fvect-cost-model=dynamic, which amdclang++ does not know).
		# Build an explicitly clang-compatible subset from CXXFLAGS;
		# extend the skip list if a newer amdclang/DIST combination
		# rejects more of them.
		local hip_flags=''
		local f
		for f in ${CXXFLAGS}; do
			case ${f} in
				-fvect-cost-model=*) ;;	# gcc-only cost-model knob
				*) hip_flags+=" ${f}" ;;
			esac
		done

		mycmakeargs+=(
			-DCMAKE_HIP_PLATFORM=amd
			-DCMAKE_HIP_COMPILER="${rocm_root}/lib/llvm/bin/amdclang++"
			-DCMAKE_HIP_COMPILER_ROCM_ROOT="${rocm_root}"
			-DCMAKE_HIP_ARCHITECTURES="${hips_archs}"
			-DCMAKE_HIP_FLAGS="${hip_flags# }"
			-DGPU_TARGETS=$(get_amdgpu_flags)
		)
	fi

	cmake_src_configure
}
