# Copyright 2026 Gentoo Authors
# Copyright 2026 Sokol
# Distributed under the terms of the GNU General Public License v2

# Strata (https://github.com/Niko1221/Strata): a local LLM inference engine
# (C++20) for GGUF Flash-Next/MoE models - an inference server with an
# OpenAI/Anthropic-compatible API plus CLI generation.  Built straight from
# the release tarball with plain CMake (>=3.24).
#
# SYSTEM LIBRARIES: every ROCm dependency is already wired through
# find_package(... CONFIG) in cmake/hip_backend.cmake - hip, hipBLAS,
# rocBLAS, hipBLASLt come from the system (dev-util/hip, sci-libs/hipBLAS,
# on this box /opt/therock-bin).  The project has no FetchContent /
# ExternalProject / file(DOWNLOAD) anywhere in the cmake paths, so there is
# nothing to patch for "system libs" - it is already system.  (The end-user
# setup.py can pull TheRock wheels, but the ebuild goes through CMake
# directly, exactly the path docs/AMD_HIP.md documents, and downloads
# nothing.)
#
# VENDORED, deliberately (all MIT, all inside the tarball, no network):
# - third_party/strata-ggml: a llama.cpp/ggml snapshot the maintainers have
#   EDITED - third_party/strata-ggml/UPSTREAM.md lists the edits (the MMQ
#   tq-overlay, template instances).  The project's -DSTRATA_GGML_DIR
#   override expects such an edited checkout; pointing it at stock sci-ml/ggml
#   would NOT unvendor anything, it would produce an invalid build because
#   the system copy lacks the Strata MMQ patches.  Left vendored for that
#   reason.
# - src/split (sFNX split-engine, upstream johnfink8/sfnx) and src/dsv41
#   (Project Maya, mw00/project-maya): MIT, no packaged equivalents exist;
#   sFNX only compiles into the STRATA_BUILD_SPLIT target which stays OFF.
#
# NOT built / NOT installed by this ebuild (see src_install for the full
# rationale): the Python CLI launcher layer (bin/strata -> strata_cli/),
# the Tauri/Rust desktop app (apps/desktop), tools/vision (strata-vision -
# a separate CMake project that downloads a llama.cpp checkout) and the
# strata-dsv41 / strata-q36(-ref) model-family engines.
#
# The HIP build reuses the four ROCm fixes proven in sci-ml/ggml on this
# same machine:
# (a) /usr/bin/hipconfig reports the NVIDIA flavour ("HIP_PLATFORM: nvidia"),
#     which sends CMake's HIP compiler check to nvcc - so CMAKE_HIP_PLATFORM
#     is forced to amd and the AMD clang from the ROCm install is pinned
#     explicitly (plus CMAKE_HIP_COMPILER_ROCM_ROOT to skip the probes).
# (b) the rocm eclass tags every AMDGPU_TARGETS element with ";", the
#     trailing one leaves an empty CMake list element - cleaned here; and
#     Strata's hip_backend.cmake answers FATAL_ERROR to any architecture
#     outside its wave32 list (all gfx9xx CDNA parts included), so the
#     targets are intersected with that list before configure.
# (c) CMake seeds CMAKE_HIP_FLAGS from CMAKE_CXX_FLAGS and amdclang++ dies
#     on gcc-only profile switches (-fvect-cost-model=dynamic here), so the
#     device flags get an explicitly clang-compatible subset.
# (d) the host C/CXX compiler stays the system gcc (standard ROCm practice;
#     the same gcc-only flags would kill amdclang++ in the host role).
# (e) the profile here carries -flto -ffat-lto-objects: ld.bfd loads the
#     LLVMgold plugin for the fat-LTO records amdclang++ embeds in HIP
#     objects and dies on them when linking strata/strata-q36 -
#     "bfd plugin: LLVM gold plugin has failed to create LTO module:
#     Invalid record" (observed 2026-11-16, Clang 24 / ROCm 7.2).  Same
#     disease as sci-ml/ggml's bug 983121, same cure: filter-lto.

EAPI=8

ROCM_VERSION=7.2
inherit cmake flag-o-matic rocm

DESCRIPTION="Local LLM inference engine for GGUF MoE models with an OpenAI-compatible API server"
HOMEPAGE="https://github.com/Niko1221/Strata"
# Release tarball; no git submodules (.gitmodules is absent) and the vendored
# deps above are in-tree, so no git-r3 / live ebuild is needed.  The project's
# internal project(strata VERSION 0.1.59.1) string is stale upstream metadata,
# the release tag is authoritative; without .git the engine reports its
# build-id as "unknown", which is expected for a tarball build.
SRC_URI="https://github.com/Niko1221/Strata/archive/refs/tags/v${PV}.tar.gz -> ${P}.tar.gz"
S="${WORKDIR}/Strata-${PV}"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64"

# cuda/sycl exist as CMake options (STRATA_ENABLE_CUDA/STRATA_ENABLE_SYCL)
# but are deliberately NOT USE flags here: each would drag a whole vendor
# toolkit (NVIDIA CUDA, Intel oneAPI) into the graph for paths this box does
# not run, and the two are mutually exclusive with HIP anyway.  There is no
# Vulkan backend in the project at all.  Without USE=rocm the build is the
# project's portable CPU set (no engine binary - see src_install).
IUSE="rocm test"

REQUIRED_USE="rocm? ( ${ROCM_REQUIRED_USE} )"

RESTRICT="!test? ( test )"

# hip + hipBLAS are what cmake/hip_backend.cmake requires by name
# (find_package(hip CONFIG REQUIRED), find_package(hipblas CONFIG REQUIRED)).
# rocBLAS rides along with hipBLAS (its own dependency) and is only probed
# with QUIET for the gfx103x tuning table; hipBLASLt is probed with QUIET too
# and its absence is handled gracefully - neither is spelled out here.
RDEPEND="
	rocm? (
		>=dev-util/hip-${ROCM_VERSION}:=
		>=sci-libs/hipBLAS-${ROCM_VERSION}:=
	)
"
DEPEND="${RDEPEND}"

DOCS=( README.md docs/AMD_HIP.md )

# Upstream link bug (files/${P}-fused-stubs.patch): the #ifndef
# STRATA_PREFILL_FUSED stub block of namespace strata::prefill::fused
# in src/prefill/prefill.cpp - the block every HIP build with
# STRATA_PREFILL_MMQ off (i.e. our configuration) compiles instead of
# the CUDA-only moe_fused*.cu - forgot the two gfx11 F16 down-row
# accessors want_dm_f16()/dm_f16_used() that Prefill::run_impl always
# references, so strata/strata-q36 died at link with
# "undefined symbol: strata::prefill::fused::want_dm_f16(bool)"
# (verified on this box 2026-11-16, also visible in upstream v0.2.13.1
# cmake + sources: nothing else ever defines them).  The patch adds the
# two trivial stubs: without the fused kernels nothing writes F16 down
# rows, so want is a no-op and used is always false.
PATCHES=( "${FILESDIR}/${P}-fused-stubs.patch" )

src_configure() {
	local mycmakeargs=(
		# GPU backend selection: HIP under USE=rocm, otherwise the plain
		# portable CPU build (see the IUSE comment).
		-DSTRATA_ENABLE_HIP=$(usex rocm)
		-DSTRATA_ENABLE_CUDA=OFF
		-DSTRATA_ENABLE_SYCL=OFF
		-DSTRATA_BUILD_TESTS=$(usex test)

		# strata-split multi-GPU mode (needs 2-4 cards) and the gfx906
		# wave64 compat mode (a mutually exclusive alternative to
		# STRATA_ENABLE_HIP) are off.
		-DSTRATA_BUILD_SPLIT=OFF
		-DSTRATA_HIP_GFX906=OFF
	)

	if use rocm; then
		local rocm_root="${ROCM_PATH:-/opt/therock-bin}"
		export ROCM_PATH="${rocm_root}"
		export HIP_PATH="${rocm_root}"

		# (e) drop LTO for the HIP build (see the header note): with
		# -flto the linker runs the gold plugin over amdclang++'s
		# fat-LTO HIP objects and the strata/strata-q36 links fail.
		# filter-lto only strips -flto*; -ffat-lto-objects without
		# -flto is an inert no-op for both gcc and clang, so it needs
		# no separate filtering (but the hip_flags loop below skips it
		# anyway to keep the device flags minimal).
		filter-lto

		# (b) the rocm eclass ";"-tags AMDGPU_TARGETS; strip and intersect
		# with the architectures hip_backend.cmake accepts (validated:
		# gfx1100 gfx1201; community: gfx1101 gfx1200; unvalidated but
		# listed: the rest).  Anything else - every gfx9xx CDNA part for
		# example - is a CMake FATAL_ERROR, so filter here and fail early
		# with a readable message when nothing supported is left.
		local strata_hip_archs=(
			gfx1010 gfx1011 gfx1012
			gfx1030 gfx1031 gfx1032 gfx1034
			gfx1100 gfx1101 gfx1102 gfx1103
			gfx1150 gfx1151 gfx1152
			gfx1200 gfx1201
		)
		local hips_archs='' t
		for t in "${AMDGPU_TARGETS[@]%;}"; do
			[[ " ${strata_hip_archs[*]} " == *" ${t} "* ]] || continue
			hips_archs+=" ${t}"
		done
		hips_archs="${hips_archs# }"
		[[ -n ${hips_archs} ]] || die "USE=rocm but none of the selected targets (${AMDGPU_TARGETS[*]}) is supported; Strata builds wave32 RDNA only: ${strata_hip_archs[*]}"

		# (c) build a clang-compatible subset of CXXFLAGS for the HIP
		# device compiler; extend the skip list if a newer amdclang
		# rejects more gcc-only switches.
		local hip_flags='' f
		for f in ${CXXFLAGS}; do
			case ${f} in
				-fvect-cost-model=*) ;;	# gcc-only cost-model knob
				-flto*|-ffat-lto-objects) ;;	# LTO: see (e) / filter-lto above
				*) hip_flags+=" ${f}" ;;
			esac
		done
		hip_flags="${hip_flags# }"

		mycmakeargs+=(
			# (a) force the AMD HIP path; hipconfig on this box lies
			# (reports the NVIDIA flavour) and CMake would pick nvcc.
			-DCMAKE_HIP_PLATFORM=amd
			-DCMAKE_HIP_COMPILER="${rocm_root}/lib/llvm/bin/amdclang++"
			-DCMAKE_HIP_COMPILER_ROCM_ROOT="${rocm_root}"
			# hip/hipblas/rocblas/hipblaslt CONFIG packages: the
			# ROCm install first, the ::gentoo ones in /usr/lib64/cmake
			# remain visible behind it.
			-DCMAKE_PREFIX_PATH="${rocm_root}"
			-DCMAKE_HIP_FLAGS="--rocm-path=${rocm_root} --rocm-device-lib-path=${rocm_root}/lib/llvm/amdgcn/bitcode${hip_flags:+ ${hip_flags}}"
			# space-separated is fine: hip_backend.cmake normalizes it
			# into its own list and validates each element.
			-DCMAKE_HIP_ARCHITECTURES="${hips_archs}"
		)
		# (d) no tc-export here: the host compiler stays the system gcc.
	fi

	cmake_src_configure
}

src_install() {
	# The project has zero install() rules (verified in 0.2.13.1), and with
	# the Ninja generator CMake does not create an 'install' target at all
	# in that case - cmake_src_install would die with "ninja: error:
	# unknown target 'install'" (observed 2026-11-16).  Install the docs
	# directly; the binaries land flat in ${BUILD_DIR} and go in by hand.
	einstalldocs

	# Always built (portable targets, no toolkit needed): the GGUF reader,
	# the scalar dequantizer and the planner self-check tools.
	dobin "${BUILD_DIR}"/strata-plan
	dobin "${BUILD_DIR}"/strata-gguf
	dobin "${BUILD_DIR}"/strata-dequant

	# The engine and its GPU diagnostics are only defined inside
	# if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) (cmake/strata_gpu_engine.cmake).
	# The engine serves its OpenAI/Anthropic-compatible API from a plain
	# launch (see docs/AMD_HIP.md); no systemd unit is installed upstream,
	# none is created here (kept minimal on purpose).
	if use rocm; then
		dobin "${BUILD_DIR}"/strata
		dobin "${BUILD_DIR}"/strata-device
		dobin "${BUILD_DIR}"/strata-load
		dobin "${BUILD_DIR}"/strata-concurrent
		dobin "${BUILD_DIR}"/strata-overlap
	fi

	# NOT installed, on purpose:
	# - bin/strata and the Python CLI layer (strata_cli/, strata_control/,
	#   serve/): the "strata launch" terminal app needs CPython 3.10+ and
	#   ~14 pip packages (numpy, jinja2, pyyaml, pillow, requests, ...) that
	#   the CMake build never touches; wiring it to system dev-python/* is a
	#   separate revision's work, and installing the launcher without its
	#   python files would ship a broken command.  The engine above is fully
	#   self-contained.
	# - strata-dsv41, strata-q36, strata-q36ref: extra model-family engines
	#   upstream does not ship in its HIP Linux packaging either.
}

src_test() {
	# Many of the 43 ctest cases need a GPU or a model pack and skip
	# themselves with a message (SKIP_RETURN_CODE 77) - harmless noise.
	# check_amdgpu (rocm eclass) dies with a friendly message when the
	# portage user cannot read/write /dev/kfd and the render nodes.
	use rocm && check_amdgpu
	cmake_src_test
}
