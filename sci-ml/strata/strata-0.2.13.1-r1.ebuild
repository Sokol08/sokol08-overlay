# Copyright 2026 Gentoo Authors
# Copyright 2026 Sokol
# Distributed under the terms of the GNU General Public License v2

# Strata (https://github.com/Niko1221/Strata): a local LLM inference engine
# (C++20) for GGUF Flash-Next/MoE models - an inference server with an
# OpenAI/Anthropic-compatible API plus CLI generation.  Built straight from
# the release tarball with plain CMake (>=3.24).
#
# WHAT -r0 BUILT, AND THE TWO THINGS -r1 ADDS.  The CMake engine itself is
# unchanged: the same v0.2.13.1 sources, the same configuration, every ROCm
# fix listed at the very bottom of this header kept verbatim, and the
# vendored-deps notes below still apply.  -r1 adds:
#
#  (1) rocBLAS and hipBLASLt as spelled-out build deps.  -r0 only asked for
#      hip and hipBLAS, so a HIP build silently came out with
#      STRATA_HIPBLASLT_AVAILABLE=OFF (hipBLASLt is not pulled in by
#      anything).  See "ROCM SOLUTION TABLES" below.
#  (2) The Python CLI layer, which -r0 deliberately left out: the terminal,
#      the front end of the API server and the model installer.  They go in
#      as plain data under ${STRATA_PREFIX} and run on the system Python
#      through a <root>/python/bin/python3 shim - the layout upstream's own
#      terminal archive has, so nothing in the launcher has to be patched.
#      See "PYTHON LAYER" and "FILE NAMES" below.
#
# SYSTEM LIBRARIES: every ROCm dependency the engine links against is already
# wired through find_package(... CONFIG) in cmake/hip_backend.cmake - hip,
# hipBLAS, rocBLAS and hipBLASLt come from the system (dev-util/hip and
# sci-libs/hipBLAS*, on this box /opt/therock-bin is the tree CMake resolves
# first).  The project has no FetchContent / ExternalProject / file(DOWNLOAD)
# anywhere in the cmake paths the engine build takes, so there is nothing to
# patch for "system libs" - it already is system.  (The end-user setup.py can
# pull TheRock wheels, but the ebuild goes through CMake directly, exactly the
# path docs/AMD_HIP.md documents, and downloads nothing.)
#
# ROCM SOLUTION TABLES (why rocBLAS and hipBLASLt are spelled out here).
# cmake/hip_backend.cmake probes both with find_package(... CONFIG QUIET):
# when one is missing it sets STRATA_HIPBLASLT_AVAILABLE /
# STRATA_ROCBLAS_AVAILABLE to OFF and prints
#     Strata: hipBLASLt not found; solution-table dispatch is unavailable
#     Strata: rocBLAS CMake package not found; the rocBLAS solution table (gfx103x) is unavailable
# When a package IS found it prints nothing at all.  So the proof a configure
# was good is the ABSENCE of those two lines in the build log, plus the two
# cache variables reading ON, plus the tune_hipblaslt / tune_rocblas tools
# being built (cmake/strata_gpu_engine.cmake:434-447).
#   - hipBLASLt is the solution-table dispatch of the prompt path plus the
#     tuning tables tools/hip/<arch>-hipblaslt-<version>.txt.  This box
#     (gfx1100) has gfx1100-hipblaslt-100401.txt, i.e. hipBLASLt 1.4.1, and
#     packaging/stage_tuning.py stages exactly that table into the engine
#     component.  Upstream only switches a table on by itself for gfx1151
#     (strata_control/hipblaslt_tuning.py: AUTO_ARCHS = ("gfx1151",)), so on
#     gfx1100 it stays opt-in: STRATA_HIPBLASLT_TABLE=on (or "hipblaslt_table"
#     in the run config).
#   - rocBLAS carries the solution table for gfx103x (RDNA2).  This box has
#     no gfx103x, so it changes nothing here; it is still spelled out because
#     the probe is unconditional and the package only reaches the build
#     transitively through sci-libs/hipBLAS (its own RDEPEND) - the kind of
#     silent breakage that shows up after a bump, not before.
# The ROCm CMake PACKAGE_VERSION and the Portage version are different
# numbers (sci-libs/hipBLASLt-10.1.0 reports 1.4.1, sci-libs/rocBLAS-10.1.0
# 5.7.0), so the dep floors stay at ${ROCM_VERSION} = 7.2 like the hip ones;
# the installed 10.1.0 in slot 0/10.1 satisfies them through :=.
#
# PYTHON LAYER: python-any-r1, NOT python-single-r1.  Every dev-python/*
# package the terminal needs is a distutils-r1 (=> python-r1) package, i.e.
# built for several targets; python-single-r1 would force the user's
# PYTHON_SINGLE_TARGET onto a set of packages that only have to run on ONE
# interpreter at run time - and that interpreter is picked at run time by the
# launcher anyway (bin/strata:32-64 looks for <root>/python/bin/python3
# before anything on PATH, which is where the shim below lands).  So:
# python-any-r1, the runtime set through $(python_gen_any_dep ...) with
# [${PYTHON_USEDEP}] and NO version floors (the pip pins do not match the
# Portage versions: numpy 2.5.3 vs 2.5.5, regex 2026.9.10 vs the installed
# 2026.11.15, certifi 2026.10.19 vs 3024.7.22), python_check_deps() through
# python_has_version, and pkg_setup -> python_setup.  ::pf4public's
# sci-misc/strata does the same for the same upstream.
#   REQUIRED_USE has NO `python? ( ${PYTHON_REQUIRED_USE} )` clause on
#   purpose: python-any-r1 deliberately sets
#     PYTHON_REQUIRED_USE='I-DO-NOT-EXIST-IN-PYTHON-ANY-R1'
#     (python-any-r1.eclass, "fake var to catch mistaken usage")
#   - expanding it would turn USE=python into a dependency on an atom that
#   does not exist.  An any-of dependency needs no REQUIRED_USE.
#   jsonschema and referencing are NOT in RDEPEND either: serve/structured.py
#   imports them lazily, only for response_format: json_schema, and prints an
#   actionable hint instead of failing when they are absent.  They are offered
#   as optfeature; the `[format]` USE of the pip extra cannot be spelled here
#   anyway - jsonschema-4.26.0 has no IUSE at all.  colorama is not depended
#   on: requirements.txt marks it sys_platform == "win32".
#
# FILE NAMES - THERE IS DELIBERATELY NO /usr/bin/strata HERE.  /usr/bin/strata
# and /usr/bin/strata-desktop belong to the INSTALLED sci-misc/strata-0.2.15
# (::pf4public, the very same upstream, and its own launchers), and that
# package owns 462 more files under /usr/lib64/strata/** (its strata.py,
# setup.py, VERSION, bin/strata, serve/, strata_cli/, components/engine/...).
# A second package putting its engine or its python layer there would abort
# the merge with file collisions, and not only on /usr/bin/strata: the two
# trees differ in content (VERSION, setup.py, every serve/*.py of a different
# release), so this is not the identical-file exception.  Therefore:
#   - everything of this package lives under ${STRATA_PREFIX} =
#     /usr/libexec/strata instead of /usr/$(get_libdir)/strata.  The layout
#     below it is upstream's own (<root>/bin/strata, <root>/python/bin/
#     python3, <root>/strata.py, <root>/components/engine), which is what the
#     launcher resolves relatively, so only the prefix moves.
#   - the terminal's command is **strata-cli**, not strata.
#   - the engine is **strata-engine** and is NOT a command: it is an engine
#     component package under ${STRATA_PREFIX}/components/engine, which is
#     exactly how upstream starts it - strata_control/engine_component.py
#     resolves the reference "component:strata-engine" and never searches
#     PATH (its shutil.which calls are for tailscale / nvidia-smi / rocminfo /
#     ... only), so a /usr/bin symlink would buy nothing.  --engine <path>
#     stays the manual override for a bare binary.
#   - /usr/bin keeps only the CMake-built tools that do not collide:
#     strata-plan, strata-gguf, strata-dequant and the HIP diagnostics
#     strata-device, strata-load, strata-concurrent, strata-overlap.
#   - strata-cli links the packaged engine into the user's own component
#     store before starting (files/${P}-strata-cli.in).  STRATA_COMPONENTS_DIR
#     is NOT exported system-wide: doing so once sent every write of Strata
#     into the read-only /usr tree (PF4Public/gentoo-overlay#533,
#     Niko1221/Strata#5591).
#   - Coexistence caveat, upstream behaviour: engine_component.resolve()
#     picks the NEWEST usable package in the store, so a user who has run the
#     launchers of BOTH packages gets 0.1.61 (0.2.15's engine) over 0.1.59.1
#     (this one).  `strata-cli components` / `strata-cli doctor` show what is
#     actually seen; --engine <component folder> forces a specific one.
#
# NOT built / NOT installed: the Tauri/Rust desktop app (apps/desktop), the
# strata-dsv41 / strata-q36(-ref) model-family engines and tools/vision
# (strata-vision, the picture reader - the engine component is packed with
# --no-picture-reader, as ::pf4public does).  See the VISION note at the
# bottom for why the flag exists but refuses to build in this revision.
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
PYTHON_COMPAT=( python3_{11..15} )
inherit cmake flag-o-matic python-any-r1 rocm

DESCRIPTION="Local LLM inference engine for GGUF MoE models with an OpenAI-compatible API server"
HOMEPAGE="https://github.com/Niko1221/Strata"
# Release tarball; no git submodules (.gitmodules is absent) and the vendored
# deps above are in-tree, so no git-r3 / live ebuild is needed.  -r1 is a
# packaging-only revision: the very same v0.2.13.1 sources, only the packaging
# changed, so MY_PV keeps the tag and the patch name free of the revision
# suffix.  The project's internal project(strata VERSION 0.1.59.1) string is
# stale upstream metadata, the release tag is authoritative; without .git the
# engine reports its build-id as "unknown", which is expected for a tarball
# build - and is why --build-id is passed explicitly below.
#
# DISTFILE NAME: renamed explicitly to ${PN}-${MY_PV}.tar.gz instead of ${P}.
#   -r0 pulls the identical file under that name already (there PV == MY_PV),
#   so ONE distfile - strata-0.2.13.1.tar.gz, one DIST line in the Manifest -
#   covers both ebuilds.  Renaming to ${P} would have leaked the revision
#   suffix into the file name (strata-0.2.13.1-r1.tar.gz): a byte-identical
#   second copy of a 35 MB tarball in DISTDIR, a redundant DIST entry, and a
#   deviation from the convention that a distfile is named after its sources,
#   not after the ebuild revision.
MY_PV="${PV%%-r*}"
SRC_URI="${HOMEPAGE}/archive/refs/tags/v${MY_PV}.tar.gz -> ${PN}-${MY_PV}.tar.gz"
S="${WORKDIR}/Strata-${MY_PV}"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64"

# cuda/sycl exist as CMake options (STRATA_ENABLE_CUDA/STRATA_ENABLE_SYCL)
# but are deliberately NOT USE flags here: each would drag a whole vendor
# toolkit (NVIDIA CUDA, Intel oneAPI) into the graph for paths this box does
# not run, and the two are mutually exclusive with HIP anyway.  There is no
# Vulkan backend in the project at all.  Without USE=rocm the build is the
# project's portable CPU set (no engine binary and no engine component -
# see src_install).
#
# VISION (an OFF-by-default flag that refuses to build here): tools/vision
# builds strata-vision, the engine's picture reader.  Its CMakeLists does
# add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL) and links PRIVATE mtmd
# llama - it needs a full llama.cpp checkout on the pin
# third_party/ggml/VERSION.txt names
# (3cf03257f219afbe7334045ff7c6a06ac68c627d), with BUILD_SHARED_LIBS OFF and
# a dozen FORCE'd options, i.e. it statically links llama.cpp's sources; it
# never downloads anything itself.  The requested "system ggml" route does not
# work for 0.2.13.1: sci-ml/ggml exposes the ggml API only (ggml*.h /
# libggml* / ggml CMake package), not llama_* or mtmd_* - of the 34 functions
# tools/vision calls it is 4, with 0 of the 22 mtmd_* and 0 of the 11 llama_*
# (nm -D --defined-only over /usr/lib64/libggml*.so*, checked 2026-11-17).
# The installed sci-misc/llama-cpp is the other half of the problem, and the
# distinction matters: its libraries DO carry the symbols (22 mtmd_* in
# /usr/lib64/llama.cpp/libmtmd.so.1.0.0, 8 llama_* in libllama.so.1.0.0), what
# it does not give is the public headers (no mtmd.h, mtmd-helper.h or llama.h
# anywhere in /usr/include), a usable `mtmd` CMake target (it ships packages
# ggml and llama only, and llama-config.cmake declares
# `add_library(llama UNKNOWN IMPORTED)`), and above all a source tree - which
# is what add_subdirectory + BUILD_SHARED_LIBS OFF + ~20 FORCE'd options ask
# for, i.e. static linking of llama.cpp's own sources, not of these .so's.
# Add to that a live git HEAD (lib*.so.1.0.0) against a 2026-09-20 pin.  A
# working vision flag would therefore need a separate llama.cpp distfile on the
# above pin, a separate `cmake -S tools/vision` build (tools/vision is NOT in
# the top-level project), the bf16-wmma mma.cuh patch, dropping
# --no-picture-reader with BUILD.json vision_archs recorded, and a real
# `strata-vision --help` validation (the HIP variant is named
# strata-vision-hip).  Until that is done, USE=vision is present (so the
# feature is visible in package USE) and defaults OFF, but pkg_setup() dies
# with this evidence instead of silently ignoring the flag or pretending system
# ggml can do it.
IUSE="python rocm test vision"

REQUIRED_USE="vision? ( rocm )
	rocm? ( ${ROCM_REQUIRED_USE} )
"

RESTRICT="!test? ( test )"

# The terminal's own runtime set, taken straight from requirements.txt: only
# the direct imports, everything else is those packages' own dependencies.
# No version floors (see the PYTHON LAYER note) and no colorama
# (sys_platform == "win32").
STRATA_PY_DEPS=$(python_gen_any_dep '
	dev-python/jinja2[${PYTHON_USEDEP}]
	dev-python/numpy[${PYTHON_USEDEP}]
	dev-python/pillow[${PYTHON_USEDEP}]
	dev-python/psutil[${PYTHON_USEDEP}]
	dev-python/pyyaml[${PYTHON_USEDEP}]
	dev-python/regex[${PYTHON_USEDEP}]
	dev-python/requests[${PYTHON_USEDEP}]
	dev-python/tqdm[${PYTHON_USEDEP}]
	')

# hip + hipBLAS are what cmake/hip_backend.cmake requires by name
# (find_package(hip CONFIG REQUIRED), find_package(hipblas CONFIG REQUIRED));
# rocBLAS and hipBLASLt are the QUIET probes described in the header.
RDEPEND="
	python? ( ${STRATA_PY_DEPS} )
	rocm? (
		>=dev-util/hip-${ROCM_VERSION}:=
		>=sci-libs/hipBLAS-${ROCM_VERSION}:=
		>=sci-libs/hipBLASLt-${ROCM_VERSION}:=
		>=sci-libs/rocBLAS-${ROCM_VERSION}:=
	)
"
DEPEND="${RDEPEND}"

# packaging/{stage_tuning,package_engine}.py run at install time for every HIP
# build, USE=python or not - any PYTHON_COMPAT interpreter is enough for them
# (they import nothing outside the standard library and the tree's own
# strata_control/).  With USE=python this is also the interpreter the terminal
# runs on (the shim below starts exactly what python_setup picked).
BDEPEND="${PYTHON_DEPS}"

# Where this package installs its data.  NOT /usr/$(get_libdir)/strata - that
# prefix belongs to the installed sci-misc/strata-0.2.15 (header: "FILE
# NAMES").  Everything under it keeps upstream's relative layout.
STRATA_PREFIX="usr/libexec/strata"

DOCS=( README.md docs/AMD_HIP.md )

# Upstream link bug (files/strata-0.2.13.1-fused-stubs.patch - the same patch
# -r0 uses, named after the source version, not the revision): the #ifndef
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
PATCHES=( "${FILESDIR}/strata-${MY_PV}-fused-stubs.patch" )

python_check_deps() {
	# the terminal runs on the implementation python_setup picks: all of its
	# direct requirements must be built for it
	python_has_version \
		"dev-python/jinja2[${PYTHON_USEDEP}]" \
		"dev-python/numpy[${PYTHON_USEDEP}]" \
		"dev-python/pillow[${PYTHON_USEDEP}]" \
		"dev-python/psutil[${PYTHON_USEDEP}]" \
		"dev-python/pyyaml[${PYTHON_USEDEP}]" \
		"dev-python/regex[${PYTHON_USEDEP}]" \
		"dev-python/requests[${PYTHON_USEDEP}]" \
		"dev-python/tqdm[${PYTHON_USEDEP}]"
}

pkg_setup() {
	if use vision; then
		# Every number below is command output taken on this box on
		# 2026-11-17 (nm -D --defined-only, qlist/CONTENTS, ls of
		# /usr/include and /usr/lib64/cmake), not a guess - see the VISION
		# note in the header.
		die "USE=vision is not supported by sci-ml/strata-${PV} yet.
tools/vision is its own CMake project (it is NOT in the top-level project):
add_subdirectory(\${LLAMA_DIR} llama EXCLUDE_FROM_ALL) plus
target_link_libraries(strata-vision PRIVATE mtmd llama) with
BUILD_SHARED_LIBS OFF and ~20 FORCE'd cache options, i.e. it statically links
a full llama.cpp checkout on the pin third_party/ggml/VERSION.txt names -
3cf03257f219afbe7334045ff7c6a06ac68c627d (2026-09-20) - and downloads nothing
itself; third_party/llama.cpp is not in the release tarball.
What this box has does not cover it:
  sci-ml/ggml exports 4 of the 34 functions tools/vision calls, 0 of the 22
      mtmd_* and 0 of the 11 llama_* (nm -D --defined-only over
      /usr/lib64/libggml*.so*); it installs ggml*.h, /usr/lib64/cmake/ggml and
      ggml.pc only.
  sci-misc/llama-cpp DOES export them (22 mtmd_* in
      /usr/lib64/llama.cpp/libmtmd.so.1.0.0, 8 llama_* in libllama.so.1.0.0),
      but installs no public headers (no mtmd.h, mtmd-helper.h or llama.h), no
      'mtmd' CMake target (packages ggml and llama only, llama-config.cmake
      has add_library(llama UNKNOWN IMPORTED)), and hands over shared objects
      of a live git HEAD where the build wants the pinned sources compiled in.
What would make it work: a llama.cpp distfile on that pin
(https://github.com/ggml-org/llama.cpp/archive/<pin>.tar.gz) unpacked in
src_unpack, -DLLAMA_DIR=<that tree>, a separate 'cmake -S tools/vision' build
(the HIP variant is named strata-vision-hip), the amdgcn bf16-wmma mma.cuh
fix, packaging without --no-picture-reader plus a BUILD.json vision_archs
record, and a real 'strata-vision --help' smoke test."
	fi
	python_setup
}

src_prepare() {
	cmake_src_prepare

	# The engine version for the component folder name and its manifest - the
	# CMake project version, not ${PV} (the two differ upstream; see the
	# SRC_URI note).  strata_control compares it when listing components.
	ENGINE_PV=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
		"${S}/CMakeLists.txt" | head -n 1)
	if use rocm; then
		[[ -n ${ENGINE_PV} ]] || die "cannot find the engine version in CMakeLists.txt"
	fi
}

# strata_rocm_root: same ROCm root for configure and install phases.  The
# src_configure() export does not survive into src_install(), and the tuning
# scripts need a root with include/hipblaslt/hipblaslt-version.h.
strata_rocm_root() {
	echo "${ROCM_PATH:-${HIP_PATH:-/opt/therock-bin}}"
}

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
		local rocm_root="$(strata_rocm_root)"
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
	local libdir="${STRATA_PREFIX}"

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
	if use rocm; then
		# The GPU diagnostics go to /usr/bin (no name clash there); the
		# engine itself does NOT - it becomes an engine component (see the
		# header's FILE NAMES note and strata_engine_component below).
		dobin "${BUILD_DIR}"/strata-device
		dobin "${BUILD_DIR}"/strata-load
		dobin "${BUILD_DIR}"/strata-concurrent
		dobin "${BUILD_DIR}"/strata-overlap

		strata_engine_component
	fi

	if use python; then
		# ---- the terminal: the strata-cli command, the API server, the model
		# installer.  Data only: no pip, no python-install, no venv.  This is
		# upstream's archive layout, which is also what makes `strata-cli
		# install` run setup.py in its download-only Control mode instead of
		# the source-checkout mode that would try to build an engine into the
		# install prefix.
		insinto "/${libdir}"
		doins strata.py setup.py requirements.txt requirements-monitor.txt
		doins -r strata_cli strata_control serve
		doins -r data

		insinto "/${libdir}/bench"
		doins -r bench/prompts

		insinto "/${libdir}/bin"
		doins bin/strata
		# the tarball ships bin/strata 0775, the git checkout 0755: keep the
		# mode explicit rather than inherit it from the umask
		fperms 0755 "/${libdir}/bin/strata"

		insinto "/${libdir}/ref"
		doins ref/load.py

		insinto "/${libdir}/docs"
		doins docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md

		# The tools modules the terminal's own paths import - the same closure
		# packaging/package_terminal.py stages into its archive, computed from
		# the sources instead of guessed at (setup_tool_imports +
		# local_imports over INSTALLER_TOOLS / INSTALLER_SCRIPTS /
		# TOOLS_SUBSET).  chat.py and highlight.py are added by name: the
		# closure misses them (no terminal path imports them by name), but
		# they are reachable from the tools package.
		local stage t
		stage="${T}/tools-stage"
		mkdir -p "${stage}" || die
		while IFS= read -r t; do
			cp --parents "tools/${t}.py" "${stage}/" || die "tools/${t}.py: cannot stage"
		done < <(
			"${PYTHON:-python3}" - "${S}" <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pt", sys.argv[1] + "/packaging/package_terminal.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
names = set(pt.INSTALLER_TOOLS) | set(pt.INSTALLER_SCRIPTS) | set(pt.setup_tool_imports(pt.ROOT))
tools = pt.local_imports(pt.ROOT, tuple(sorted(names)))
[print(name[:-3] if name.endswith(".py") else name) for name in sorted(set(tools) | set(pt.TOOLS_SUBSET))]
EOF
		)
		cp --parents tools/chat.py tools/highlight.py "${stage}/" || die
		doins -r "${stage}/tools"

		# The terminal's own version label.  strata_cli.version() reads exactly
		# this file (and strata_cli.cli only treats the install as a Terminal
		# bundle when VERSION and python/ are both there), so it must name the
		# Portage revision, not the upstream tag.
		echo "${PV}" > "${D}/${libdir}/VERSION" || die

		# The interpreter the terminal runs on: bin/strata looks for
		# <root>/python/bin/python3 first (bin/strata:44-47, upstream's
		# private runtime), before anything on PATH.  These shims start the
		# implementation python_setup picked in pkg_setup, with the user's
		# site-packages kept out - the launcher does that for this path too.
		strata_python_shims "${D}/${libdir}/python/bin" "/${libdir}/python/bin"

		# /usr/bin/strata-cli is not a symlink to bin/strata: it links the
		# packaged engine into the user's own component store first (see the
		# header and files/${P}-strata-cli.in), then runs the terminal.
		strata_cli_launcher
	fi

	# No env.d on purpose: the terminal finds its interpreter through the
	# python/ shim above, and STRATA_COMPONENTS_DIR must stay unset (header).
}

# strata_python_shims <staging dir> <image path>: the python, python3 and
# ${EPYTHON} launchers that start the implementation python_setup picked,
# without the user's site-packages - the layout upstream's bundles have under
# python/bin/, which is where bin/strata looks first.
strata_python_shims() {
	local stage="$1" image="$2" pname
	mkdir -p "${stage}" || die
	for pname in python python3 "${EPYTHON}"; do
		cat > "${stage}/${pname}" <<-EOF || die
		#!/bin/sh
		# Strata (sci-ml/strata): starts the system python (${PYTHON})
		export PYTHONNOUSERSITE=1
		export PYTHONDONTWRITEBYTECODE=1
		exec ${PYTHON} "\$@"
		EOF
		fperms 0755 "${image}/${pname}"
	done
}

# strata_cli_launcher: /usr/bin/strata-cli from files/${P}-strata-cli.in -
# links the packaged engine store into the user's component store, then runs
# <prefix>/bin/strata.
strata_cli_launcher() {
	local prefix="/${STRATA_PREFIX}"
	sed -e "s|@NAME@|strata-cli|g" \
		-e "s|@WHAT@|the Strata terminal|g" \
		-e "s|@PV@|${PV}|g" \
		-e "s|@SYSCOMPS@|${prefix}/components|g" \
		-e "s|@TARGET@|${prefix}/bin/strata|g" \
		"${FILESDIR}/${P}-strata-cli.in" > "${T}/strata-cli" || die
	newbin "${T}/strata-cli" strata-cli
}

# strata_engine_component: turn the built engine into an engine component
# package (strata_control/engine_component.py, packaging/ENGINE-COMPONENT.md).
# The terminal refuses to start an engine that is not such a package: its
# manifest (engine-component.json, schema 2) hashes every file of the
# component and resolve() re-checks them on every start, so a bare symlink
# into /usr/bin is not a supported installation at all - the only bypass is
# --engine <absolute path>.
#
# The tuning tables go in with it (packaging/stage_tuning.py records the
# hipBLASLt version the package carries in BUILD.json and copies the matching
# tools/hip/<arch>-hipblaslt-<version>.txt into <stage>/tuning/; without that
# BUILD.json field package_engine.py refuses the package outright).
strata_engine_component() {
	local libdir="${STRATA_PREFIX}"
	local rocm_root stage archs arch arch_args allow_libs

	archs=$(sed -n 's/^CMAKE_HIP_ARCHITECTURES:STRING=\(.*\)/\1/p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)
	archs="${archs//;/ }"
	[[ -n ${archs} ]] || die "cannot read CMAKE_HIP_ARCHITECTURES from ${BUILD_DIR}/CMakeCache.txt"

	stage="${T}/engine-stage"
	mkdir -p "${stage}" || die
	cp "${BUILD_DIR}/strata" "${stage}/" || die

	# the hipBLASLt version of the ROCm tree the engine links against, and the
	# tuning tables for this build's architectures (see the ROCM SOLUTION
	# TABLES note in the header)
	rocm_root="$(strata_rocm_root)"
	"${PYTHON:-python3}" "${S}/packaging/stage_tuning.py" \
		--stage "${stage}" --archs "${archs}" --rocm "${rocm_root}" || die

	# package_engine.py refuses a component whose binaries import libraries
	# that are neither bundled nor listed as belonging to the OS or the GPU
	# driver (packaging/binary_deps.py).  The ROCm runtime and BLAS libraries
	# are RDEPEND-provided and their libdir is in the ldconfig cache, so they
	# are named here rather than copied into the package.
	allow_libs=(
		--allow-lib libamdhip64.so.7
		--allow-lib libhipblas.so.3
		--allow-lib libhipblaslt.so.1
		--allow-lib librocblas.so.5
	)

	arch_args=()
	for arch in ${archs}; do
		# --arch is repeatable; passed separately so each lands as its own
		# value in the manifest's archs list.
		arch_args+=(--arch "${arch}")
	done

	# --no-picture-reader: strata-vision is not built (see the VISION note),
	# so the engine ships text-only on purpose and the packager does not have
	# to be told twice.
	"${PYTHON:-python3}" "${S}/packaging/package_engine.py" \
		--src "${stage}" \
		--backend hip \
		--version "${ENGINE_PV}" \
		--platform linux-x86_64 \
		--out "${D}/${libdir}/components/engine" \
		"${arch_args[@]}" \
		"${allow_libs[@]}" \
		--build-id "gentoo-${PV}" \
		--no-picture-reader || die

	# estrip (which runs after src_install) would break the component's
	# integrity: engine-component.json records the binaries' SHA-256, so the
	# engine must reach the image unstripped.
	dostrip -x "/${libdir}/components/engine"
}

pkg_postinst() {
	elog "sci-ml/strata-${PV} installs under /${STRATA_PREFIX}, not under the sci-misc/strata prefix."
	elog "The terminal command is strata-cli; /usr/bin/strata and /usr/bin/strata-desktop remain owned by sci-misc/strata."
	elog "The engine is a component package under /${STRATA_PREFIX}/components/engine; strata-cli links it into the per-user component store."
	if use python; then
		optfeature "structured output for serve/ API calls (response_format: json_schema)" \
			dev-python/jsonschema dev-python/referencing
	fi
}

src_test() {
	# Many of the 43 ctest cases need a GPU or a model pack and skip
	# themselves with a message (SKIP_RETURN_CODE 77) - harmless noise.
	# check_amdgpu (rocm eclass) dies with a friendly message when the
	# portage user cannot read/write /dev/kfd and the render nodes.
	use rocm && check_amdgpu
	cmake_src_test
}
