# Copyright 2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

# Strata (https://github.com/Niko1221/Strata): a local LLM inference engine
# (C++20) for GGUF Flash-Next/MoE models - an inference server with an
# OpenAI/Anthropic-compatible API plus CLI generation.  Built straight from the
# release tarball with plain CMake (>=3.24), and this revision needs no network
# at configure time (see "NO llama.cpp DISTFILE" below).
#
# WHICH PV, AND WHY.  v0.2.15 is the newest tag of the upstream repository:
# tags on 2026-11-18 (git ls-remote / the GitHub tags API, 78 tags) run
#   v0.2.15 c77572806996349b49995e4f98d41b5294d36a21  (release 2026-11-16)
#   v0.2.14.1 3af562d541ec76c8da63d568248a919c8a54906b (tag, no release)
#   v0.2.14   09b603886925994d0d4461857b01321ef3aa431c  (release 2026-11-15)
#   v0.2.13.1 628f74955a56fcba092495b800dd02af75373a0f  (what our -r0/-r1 build)
# and the only prerelease-shaped tag in the whole list is v0.2.0-beta; the
# release listing marks v0.2.15 "prerelease: false".  So 0.2.15 is the last
# stable version, and it is the same source tree ::pf4public's
# sci-misc/strata-0.2.15 builds.  Its engine version string is
# project(strata VERSION 0.1.61) (CMakeLists.txt:13), not ${PV}: the release tag
# is authoritative for Portage, ENGINE_PV below is what the engine component is
# named after.
#
# Our sci-ml/strata-0.2.13.1 and -0.2.13.1-r1 stay in the tree untouched (they
# install under /usr/libexec/strata with the command strata-cli and collide with
# nothing).  This ebuild is a NEW PV, not a revision of those, so the two
# layouts coexist.
#
# LAYOUT - HERE THIS PACKAGE FOLLOWS ::pf4public INSTEAD OF OUR -r1, ON PURPOSE.
# -r1 had to dodge the INSTALLED sci-misc/strata-0.2.15: that package owns
# /usr/bin/strata and 462 files under /usr/lib64/strata/**, so -r1 moved
# everything to /usr/libexec/strata and renamed the command strata-cli.  This
# revision is meant to REPLACE that package (the user unmerges sci-misc/strata
# first - pkg_pretend says so), so it takes the upstream-standard prefix
# /usr/$(get_libdir)/strata and the standard command /usr/bin/strata: same
# layout as ::pf4public's, which means the user's $XDG_DATA_HOME/strata
# component store, run configs and model folders keep working across the switch
# unchanged.  Consequences:
#   - do NOT install this while sci-misc/strata is merged: the file collision
#     aborts the merge (/usr/bin/strata, VERSION, every serve/*.py).
#   - the engine is again a component package under
#     /usr/$(get_libdir)/strata/components/engine (upstream's own
#     engine-component schema), NOT a /usr/bin binary; /usr/bin/strata is the
#     launcher that links it into the user's own component store.
#   - STRATA_COMPONENTS_DIR is still never exported system-wide: pointing it at
#     the read-only system store sent every write of Strata into /usr
#     (PF4Public/gentoo-overlay#533, Niko1221/Strata#5591).
#   - the CMake-built diagnostics stay in /usr/bin (strata-plan, strata-gguf,
#     strata-dequant, and under USE=rocm strata-device, strata-load,
#     strata-concurrent, strata-overlap); ::pf4public does not install them, and
#     they are useful: they are the read-the-GGUF / plan / device probes.
#
# PERFORMANCE: WHAT THIS REVISION BUILDS IN (the reason to move off 0.2.13.1).
# Our 0.2.13.1 built the HIP engine with the portable FP16 prompt path only:
# STRATA_MMQ was never passed, so it kept its upstream default OFF, and
# src/prefill/prefill.cpp ran the FP16 experts.  0.2.15 builds llama.cpp's MMQ
# prompt kernels on HIP, which is what an Unsloth UD K-quant pack (this box runs
# qwen3.8-flash-next-unsloth-ud) wants.  Every option below was read in the
# v0.2.15 sources, with its upstream default in brackets:
#   STRATA_MMQ=ON                       [OFF] CMakeLists.txt:112 "Build llama.cpp's
#       MMQ prompt kernels (CUDA; HIP together with STRATA_PREFILL_MMQ)".  USE=mmq.
#   STRATA_MMQ_KQUANTS=ON               [ON]  CMakeLists.txt:154, "the experts of
#       Unsloth's K-quant files, and on HIP the dense GGUF projections
#       (STRATA_DENSE_MMQ)".  cmake/backends/hip.cmake:57-60 adds the instances
#       q4_k q5_k q5_1 q6_k to the HIP strata_mmq library.  USE=mmq.
#   STRATA_PREFILL_MMQ=ON               [OFF] CMakeLists.txt:105, "the optional GGML
#       MMQ prefill path for HIP (only together with STRATA_MMQ=ON)";
#       CMakeLists.txt:172 FATAL_ERRORs without STRATA_ENABLE_HIP, hence
#       REQUIRED_USE "mmq? ( rocm )".  USE=mmq.
#   STRATA_NATIVE_EXPERTS=ON            [ON]  CMakeLists.txt:207 (native IQ experts
#       through ggml-cpu).  Passed explicitly: it is also what makes the ggml tree
#       and the fused prompt kernels be built at all - CMakeLists.txt:600-619 skips
#       strata_hip_mmq/strata_hip_pf_own entirely without it.
#   STRATA_Q6K_EXPERTS=ON               [ON]  CMakeLists.txt:158, "the Q6_K gate/up
#       grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)".  Passed explicitly
#       against default drift.
#   STRATA_MMQ_TQ_OVERLAY left at its default ON (CMakeLists.txt:153): with the
#       vendored ggml the trellis MMQ comes from the pre-patched folder
#       third_party/strata-ggml/ggml/src/ggml-cuda-tq (see "NO llama.cpp DISTFILE").
#   STRATA_PORTABLE=OFF (the upstream default): ggml-cpu and Strata's CPU kernels
#       are built for THIS host (CMake sets GGML_NATIVE ON, and make.conf here is
#       -march=znver4).  ::pf4public builds the AVX2-baseline portable engine
#       instead, because its package ships to many CPUs; an overlay for one box
#       takes the native one.  Their `old-cpu`/STRATA_ISA_FLOOR knob is not offered.
#   STRATA_BUILD_SPLIT=OFF (multi-GPU strata-split, 2-4 cards), STRATA_HIP_GFX906=OFF
#       (the wave64 CUDA-compat build, mutually exclusive with USE=rocm),
#       STRATA_ENABLE_CUDA=OFF, STRATA_ENABLE_SYCL=OFF.
# libstrata_kq.so (the K-quant MMQ library the engine dlopens) is a CUDA-only
# target - cmake/backends/cuda.cmake:132-259 is the only place it is created.  On
# HIP the same K-quant MMQ instances are compiled into strata_mmq
# (cmake/backends/hip.cmake:57-64), so a HIP build has them in the engine binary
# and src_compile installs no second library.
#
# NO llama.cpp DISTFILE - THE ONE PLACE THIS EBUILD IS SIMPLER THAN ::pf4public.
# pf4public downloads https://github.com/ggml-org/llama.cpp/archive/<pin>.tar.gz
# and passes -DSTRATA_GGML_DIR=<that tree> plus -DLLAMA_DIR for the picture
# reader.  0.2.15 vendors both trees instead, at exactly the pin pf4public
# downloads - third_party/ggml/VERSION.txt:
#   3cf03257f219afbe7334045ff7c6a06ac68c627d Sun Sep 20 16:08:11 2026 +0800
#   - third_party/strata-ggml: ggml-base + ggml-cpu (x86/Arm) + exactly the
#     ggml-cuda MMQ sources and instances (q2_0, iq2_*, iq3_*, iq4_*, q8_0, q5_0,
#     q4_k, q5_k, q5_1, q6_k), plus the trellis types as the already-patched
#     folder ggml/src/ggml-cuda-tq.  Its UPSTREAM.md: "This folder holds exactly
#     those sources, copied from llama.cpp, so an engine build needs no network
#     and no llama.cpp download", and "-DSTRATA_GGML_DIR=<llama.cpp checkout>
#     still builds from such a checkout instead of this folder (an override kept
#     for two releases, then removed)".
#   - third_party/strata-llama: the whole llama.cpp (llama, mtmd, ggml with its
#     backends), byte-identical to the pin, for strata-vision.
# So STRATA_GGML_DIR/LLAMA_DIR stay unset (the vendored tree is the default), no
# 37 MB distfile is added to SRC_URI, nothing is fetched at configure time, and
# the trellis MMQ needs no configure-time header overlay.
#
# PATCHES.  All three were re-derived from the v0.2.15 sources on this box
# (patch --dry-run + a real apply), and the two wmma fixes are ports of
# PF4Public/gentoo-overlay's sci-misc/strata files/ (their strata-wmma-bf16.patch
# and strata-llama-wmma-bf16.patch; GPL-2.0, Gentoo Authors - compatible with
# this package's MIT license, they are separate files and the engine's own code
# stays MIT).  Both are applied only when the buggy text is actually there, so a
# future upstream fix makes them no-ops instead of a broken patch:
#   files/${PN}-${MY_PV}-wmma-bf16.patch - src/prefill/wmma_gemm.cu
#     (WmmaTraits<__bf16>::vec_t line 80, hb16 line 489) and src/prefill/kernels.cu
#     (um_b16 line 2603): a bf16 WMMA fragment declared as
#     __bf16 __attribute__((ext_vector_type(16))) is rejected by the amdgcn
#     builtins, which want a short-vector fragment; the typedefs become short
#     (bit-identical - the bytes reach the builtin only through a reinterpret_cast
#     or memcpy).  Grep condition: "typedef __bf16 hb16 __attribute__((ext_vector_type(16)));".
#   files/${PN}-${MY_PV}-ggml-wmma-bf16.patch - the same bug in the vendored
#     pinned ggml: third_party/strata-ggml/ggml/src/ggml-cuda/mma.cuh lines 1262
#     (RDNA4, bf16x8_t) and 1269 (RDNA3, bf16x16_t), reached from every MMQ
#     instance through mmq-vec-dot.cuh:4.  Grep condition:
#     "bf16x16_t = __attribute__((ext_vector_type(16))) __bf16".  The CDNA MFMA
#     fragments (bf16x4_t/bf16x2_t) are left alone: they are not the WMMA path.
#     (pf4public patch the downloaded llama.cpp; here the file is in-tree, so the
#     patch carries the third_party/ path and no distfile is unpacked.)
#   files/${PN}-${MY_PV}-disable-updater.patch - port of their
#     strata-disable-updater.patch, reduced to its Python half:
#     strata_cli/update_cmd.py DEFAULT_FEED = "" (no unsigned release feed is
#     fetched by an install Portage manages).  The tauri updater half is dropped,
#     this ebuild does not build apps/desktop.
# THE 0.2.13.1 LINK FIX (files/strata-0.2.13.1-fused-stubs.patch) IS NOT CARRIED
# HERE, and that is a checked result, not an omission.  In 0.2.13.1 the
# #ifndef STRATA_PREFILL_FUSED stub block of namespace strata::prefill::fused in
# src/prefill/prefill.cpp lacked want_dm_f16()/dm_f16_used(), which
# Prefill::run_impl references, so a HIP build without the fused kernels died at
# link.  In 0.2.15 prefill.cpp still calls them (lines 5541/5553/6709/8307) and
# the stub block (lines 175-203) still lacks them - but every configuration that
# compiles prefill.cpp now also defines STRATA_PREFILL_FUSED: the file is only
# built as part of strata_prefill (cmake/strata_gpu_engine.cmake:444, inside the
# CUDA/HIP engine block), and each backend macro sets the definition -
# cmake/backends/hip.cmake:98 (strata_hip_mmq, i.e. USE=mmq) and :119
# (strata_hip_pf_own, i.e. USE=-mmq), cmake/backends/cuda.cmake:277 and :306; the
# dispatch is CMakeLists.txt:610-619.  The one build where the stub block is live
# is the one that skips that whole dispatch, STRATA_NATIVE_EXPERTS=OFF - which
# this ebuild always sets ON.  The 0.2.13.1 patch is kept where it belongs (for
# the 0.2.13.1 ebuilds), it does not apply to 0.2.15 (Hunk #1 fails: the stub
# block moved and gained p1a_*/set_down_rotation/hadamard128_rows stubs).
#
# USE FLAGS.  `rocm`, not ::pf4public's `hip`: this overlay names the AMD stack
# rocm (sci-ml/ggml here does the same), our own sci-ml/strata-0.2.13.1-r1 uses
# rocm, so an existing package.use line keeps working across the bump, and
# make.conf on this box already carries USE=rocm globally - which builds the
# engine here without any package.use entry.  REQUIRED_USE pins the pairs that
# CMake would otherwise FATAL_ERROR on.  `+mmq` is the performance switch of the
# whole MMQ/expert block above, default ON (USE=-mmq is the escape hatch if a
# future release breaks the MMQ path on a card; then the engine keeps Strata's
# own FP16 prompt kernels).  `+python` default ON: the terminal IS the only
# supported way to start an engine component (engine_component.py resolves
# "component:strata-engine" and never searches PATH), so a build without it
# installs tools and an engine nobody can start - the flag exists for a
# tools-only install, not as the normal case.  `test`, `tui` (the launch/monitor
# screens: strata_cli/launch/cmd.py:592-596 names the two packages they need and
# falls back to plain lines without them, so they are an option, not a
# dependency) and `vision` (present so the feature is visible, but see the die in
# pkg_setup).  There is no cuda/sycl/desktop/cargo/tauri/npm/9999 here on
# purpose: each drags a foreign toolkit or a toolchain into the graph for paths
# this box does not run, and CUDA/SYCL are mutually exclusive with HIP anyway
# (the CMake options exist; the ROCm-only focus of this overlay is the point).
#
# THE ROCm FIXES -r1 PROVED ON THIS BOX, kept verbatim (see their letters in
# src_configure): (a) /usr/bin/hipconfig here reports the NVIDIA flavour
# ("HIP_PLATFORM: nvidia") and CMake would send the HIP compiler check to nvcc -
# CMAKE_HIP_PLATFORM is forced to amd, the AMD clang of the ROCm install is
# pinned, CMAKE_HIP_COMPILER_ROCM_ROOT skips the probes; (b) the rocm eclass
# ";"-tags AMDGPU_TARGETS, the trailing tag leaves an empty CMake list element,
# and cmake/hip_backend.cmake answers FATAL_ERROR to anything outside its wave32
# list - so the targets are intersected with that list (its exact 0.2.15 sets are
# spelled out below) and the build dies early with a readable message when
# nothing supported is left; (c) CMake seeds CMAKE_HIP_FLAGS from CMAKE_CXX_FLAGS
# and amdclang++ dies on gcc-only profile switches (-fvect-cost-model=dynamic
# here), so the device flags are an explicitly clang-compatible subset plus
# --rocm-path/--rocm-device-lib-path; (d) the host C/CXX compiler stays the
# system gcc (no rocm_use_clang: the same gcc-only flags would kill amdclang++ in
# the host role); (e) filter-lto: the profile here has -flto -ffat-lto-objects
# and ld.bfd loads the LLVMgold plugin over the fat-LTO records amdclang++
# embeds in HIP objects, "bfd plugin: LLVM gold plugin has failed to create LTO
# module: Invalid record" (observed 2026-11-16 on this box, Clang 24 / ROCm 7.2;
# the same disease as sci-ml/ggml's bug 983121, same cure).
# (f) is new here and also performance: cmake.eclass wipes
# CMAKE_{C,CXX,ASM,linker}_FLAGS_<BUILD_TYPE> in its common-config cache but NOT
# the HIP one - and a -DCMAKE_HIP_FLAGS_<BUILD_TYPE>= on the command line does not
# help either: CMake's HIP language module refills it, and build.ninja then puts
# its "-O2 -g -DNDEBUG" AFTER our CMAKE_HIP_FLAGS, where clang lets the last -O
# win (measured on this box: the HIP objects were being compiled at -O2).  The fix
# is a file handed to CMake as -DCMAKE_PROJECT_INCLUDE (>=3.15): it is included at
# the end of every project() call, i.e. after the language defaults exist and
# before any target is created, and it sets the per-config flags of HIP/C/CXX to
# "-DNDEBUG" - what upstream builds with (its CMakeLists hard-sets
# CMAKE_BUILD_TYPE Release = -O3 -DNDEBUG, and it is this ebuild that wipes it),
# with -O3 coming from our flags and no -g added.
# rocBLAS and hipBLASLt are spelled out as dependencies because
# cmake/hip_backend.cmake probes both with find_package(... CONFIG QUIET) and
# only prints
#     Strata: hipBLASLt not found; solution-table dispatch is unavailable   (:82)
#     Strata: rocBLAS CMake package not found; the rocBLAS solution table (gfx103x) is unavailable   (:88)
# when one is missing - the proof a configure was good is the ABSENCE of those
# two lines plus the two cache variables reading ON.
#
# hipBLASLt SOLUTION TABLES: packaging/stage_tuning.py puts
# tools/hip/gfx1100-hipblaslt-100401.txt (hipBLASLt 1.4.1 -> 100401,
# strata_control/hipblaslt_tuning.py:version_number) of THIS ROCm tree into the
# component's tuning/ and records hipblaslt_version in BUILD.json - without that
# field package_engine.py refuses the package.  Carrying a table and USING it are
# two different things: strata_control/hipblaslt_tuning.py:36
# AUTO_ARCHS = ("gfx1151",) with AUTO_USES_TABLE = False, so on gfx1100 the table
# stays opt-in at run time (STRATA_HIPBLASLT_TABLE=on, or "hipblaslt_table": true
# in the run config) - pkg_postinst says so.

EAPI=8

ROCM_VERSION=7.2
PYTHON_COMPAT=( python3_{11..15} )
inherit cmake flag-o-matic linux-info python-any-r1 rocm

DESCRIPTION="Local LLM inference engine for GGUF MoE models with an OpenAI-compatible API server"
HOMEPAGE="https://github.com/Niko1221/Strata"

# Release tarball; no submodules, and both pinned third-party trees are in it
# (header: "NO llama.cpp DISTFILE"), so no git-r3 and no live ebuild.  The
# distfile is named ${PN}-${MY_PV}.tar.gz - byte-identical to ::pf4public's
# ${P}.tar.gz for the same release, so the one file already in DISTDIR covers
# both packages, and MY_PV keeps a future revision (-r1) from renaming it.
# Without .git the engine reports its build-id as "unknown", which is why
# --build-id is passed to the packager below.
MY_PV="${PV%%-r*}"
SRC_URI="${HOMEPAGE}/archive/refs/tags/v${MY_PV}.tar.gz -> ${PN}-${MY_PV}.tar.gz"
S="${WORKDIR}/Strata-${MY_PV}"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64"

IUSE="+mmq +python rocm test tui vision"

REQUIRED_USE="mmq? ( rocm )
	tui? ( python )
	vision? ( rocm )
	rocm? ( ${ROCM_REQUIRED_USE} )
"

RESTRICT="!test? ( test )"

# The terminal's runtime set, taken from 0.2.15's requirements.txt: the direct
# imports only.  No version floors - the pip pins do not match the Portage
# versions (numpy 2.5.3 vs 2.5.5, regex 2026.9.10 vs 2026.11.15, certifi
# 2026.10.19 vs 3024.7.22), and dev-python/* here are distutils-r1 packages that
# only have to RUN on one interpreter, the one the python/ shim below starts.
# colorama is not depended on (requirements.txt: sys_platform == "win32");
# jsonschema/referencing are lazily imported by serve/structured.py with an
# actionable hint when missing, so they are optfeature, not RDEPEND.
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

# The two packages `strata launch`/`strata monitor` draw their screens with
# (strata_cli/launch/cmd.py:592-596 names exactly these two and prints plain
# lines when either is missing) - so they are an option, not a dependency.  Same
# any-of shape as the terminal's own set: the screens run on one interpreter.
STRATA_TUI_DEPS=$(python_gen_any_dep '
	dev-python/rich[${PYTHON_USEDEP}]
	dev-python/textual[${PYTHON_USEDEP}]
	')

# hip and hipBLAS are what cmake/hip_backend.cmake requires by name
# (find_package(hip CONFIG REQUIRED), find_package(hipblas CONFIG REQUIRED));
# hipBLASLt and rocBLAS are the QUIET probes of the header.  The version floors
# stay at ${ROCM_VERSION}: the CMake PACKAGE_VERSION and the Portage version are
# different numbers (sci-libs/hipBLASLt-10.1.0 reports 1.4.1, rocBLAS-10.1.0
# 5.7.0), and the installed slot satisfies the floor through :=.
# REQUIRED_USE has no `python? ( ${PYTHON_REQUIRED_USE} )` on purpose:
# python-any-r1 sets PYTHON_REQUIRED_USE to a sentinel ("I-DO-NOT-EXIST-IN-
# PYTHON-ANY-R1") to catch mistaken usage, and an any-of dependency needs none.
RDEPEND="
	python? (
		${STRATA_PY_DEPS}
		tui? ( ${STRATA_TUI_DEPS} )
	)
	rocm? (
		>=dev-util/hip-${ROCM_VERSION}:=
		>=sci-libs/hipBLAS-${ROCM_VERSION}:=
		>=sci-libs/hipBLASLt-${ROCM_VERSION}:=
		>=sci-libs/rocBLAS-${ROCM_VERSION}:=
	)
"
DEPEND="${RDEPEND}"

# packaging/{stage_tuning,package_engine}.py run at install time for every HIP
# build, USE=python or not (they import nothing outside the standard library and
# the tree's own strata_control/); the src_install closure of the tools/ modules
# runs package_terminal.py with the same interpreter.  With USE=python this is
# also the interpreter the terminal runs on.
BDEPEND="${PYTHON_DEPS}"

DOCS=( README.md docs/AMD_HIP.md )

PATCHES=( "${FILESDIR}/${PN}-${MY_PV}-disable-updater.patch" )

# strata_prefix: the package's own tree, computed where get_libdir is safe to
# call (src_install, the engine-component helper, pkg_postinst).
strata_prefix() {
	echo "usr/$(get_libdir)/strata"
}

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
		"dev-python/tqdm[${PYTHON_USEDEP}]" || return 1
	if use tui; then
		python_has_version \
			"dev-python/rich[${PYTHON_USEDEP}]" \
			"dev-python/textual[${PYTHON_USEDEP}]" || return 1
	fi
}

pkg_pretend() {
	local libdir="$(strata_prefix)"
	if [[ -e ${EPREFIX}/${libdir}/VERSION || -e ${EPREFIX}/usr/bin/strata ]]; then
		ewarn "This package owns /${libdir}/** and /usr/bin/strata - the same files"
		ewarn "sci-misc/strata (::pf4public) owns.  If it is installed, unmerge it first:"
		ewarn "  emerge -C sci-misc/strata"
		ewarn "(our sci-ml/strata:0.2.13.1 installs under /usr/libexec/strata as strata-cli and"
		ewarn "needs nothing: it can stay, but the newest engine in the component store wins)."
	fi
}

pkg_setup() {
	python_setup

	if use rocm; then
		linux-info_pkg_setup
		if linux-info_get_any_version && linux_config_exists; then
			if ! linux_chkconfig_present HSA_AMD_SVM; then
				ewarn "USE=rocm: the kernel config has no HSA_AMD_SVM.  ROCm needs it for"
				ewarn "unified memory (the KV host/VRAM exchange of the engine); without it the"
				ewarn "GPU paths can fail at run time."
			fi
		fi
	fi

	if use vision; then
		die "USE=vision is not implemented by sci-ml/strata-${PV} yet.
tools/vision is a CMake project of its own (it is NOT in the top-level project):
cmake -S tools/vision, add_subdirectory(\${LLAMA_DIR} llama EXCLUDE_FROM_ALL) plus
target_link_libraries(strata-vision PRIVATE mtmd llama strata_vit_own).  What 0.2.15
does give is the sources: third_party/strata-llama is the whole llama.cpp (llama,
mtmd, ggml with its backends) byte-identical to the pin third_party/ggml/VERSION.txt
names (3cf03257f219afbe7334045ff7c6a06ac68c627d) and tools/vision/CMakeLists.txt:18
takes it as LLAMA_DIR by default - nothing is downloaded.  What this ebuild does not
do is the second configure+compile pass (the HIP variant is named strata-vision-hip),
the amdgcn bf16-wmma fix in that tree's ggml-cuda, packing the component without
--no-picture-reader with BUILD.json vision_archs recorded (package_engine.py
--stamp-vision), and a real 'strata-vision --help' validation.  Until that is done
the flag dies instead of being silently ignored."
	fi
}

src_prepare() {
	cmake_src_prepare

	# the engine version for the component folder name and its manifest - the
	# CMake project version (0.1.61 for 0.2.15), not ${PV}.  strata_control
	# compares it when listing components.
	ENGINE_PV=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' \
		"${S}/CMakeLists.txt" | head -n 1)
	if use rocm; then
		[[ -n ${ENGINE_PV} ]] || die "cannot find the engine version in CMakeLists.txt"
	fi

	# Strata's own RDNA3 wmma gemms (HIP-only files) declare their bf16 fragments
	# as __bf16 vectors, which the amdgcn builtins reject.  Applied only while the
	# buggy typedef is in the file (header: "PATCHES").
	if use rocm && grep -q "typedef __bf16 hb16 __attribute__((ext_vector_type(16)));" \
		"${S}/src/prefill/wmma_gemm.cu"; then
		eapply "${FILESDIR}/${PN}-${MY_PV}-wmma-bf16.patch"
	fi

	# the same bug in the pinned vendored ggml, reached by every MMQ instance
	# (mmq-vec-dot.cuh -> mma.cuh).  The CDNA MFMA fragments are not touched.
	if use rocm && grep -q "bf16x16_t = __attribute__((ext_vector_type(16))) __bf16" \
		"${S}/third_party/strata-ggml/ggml/src/ggml-cuda/mma.cuh"; then
		eapply "${FILESDIR}/${PN}-${MY_PV}-ggml-wmma-bf16.patch"
	fi
}

# strata_rocm_root: the same ROCm root for the configure and the install phases.
# The src_configure() export does not survive into src_install(), and
# packaging/stage_tuning.py needs a root with include/hipblaslt/hipblaslt-version.h.
strata_rocm_root() {
	echo "${ROCM_PATH:-${HIP_PATH:-/opt/therock-bin}}"
}

src_configure() {
	local mycmakeargs=(
		# GPU backend selection: HIP under USE=rocm, otherwise the portable CPU
		# build (no engine binary, no engine component - see src_install).
		-DSTRATA_ENABLE_HIP=$(usex rocm)
		-DSTRATA_ENABLE_CUDA=OFF
		-DSTRATA_ENABLE_SYCL=OFF
		-DSTRATA_BUILD_TESTS=$(usex test)

		# strata-split needs 2-4 cards; the gfx906 wave64 compat mode is a
		# mutually exclusive alternative to USE=rocm.
		-DSTRATA_BUILD_SPLIT=OFF
		-DSTRATA_HIP_GFX906=OFF

		# the expert paths (header: "PERFORMANCE").  NATIVE_EXPERTS also gates the
		# vendored ggml and the fused prompt kernels (CMakeLists.txt:600-619), and
		# PREFILL_MMQ FATAL_ERRORs without HIP - REQUIRED_USE mmq? ( rocm ).
		-DSTRATA_NATIVE_EXPERTS=ON
		-DSTRATA_Q6K_EXPERTS=ON
		-DSTRATA_MMQ=$(usex mmq)
		-DSTRATA_MMQ_KQUANTS=$(usex mmq)
		-DSTRATA_PREFILL_MMQ=$(usex mmq)

		# native CPU code for this host (GGML_NATIVE ON in cmake/strata_ggml.cmake);
		# the AVX2-baseline portable build is ::pf4public's shipping default.
		-DSTRATA_PORTABLE=OFF
	)

	if use rocm; then
		local rocm_root="$(strata_rocm_root)"
		export ROCM_PATH="${rocm_root}"
		export HIP_PATH="${rocm_root}"

		# (e) no LTO for the HIP build: with -flto the linker runs the gold plugin
		# over amdclang++'s fat-LTO HIP objects and the engine links fail.
		# filter-lto only strips -flto*; -ffat-lto-objects alone is inert for both
		# compilers, but the hip_flags loop below skips it anyway to keep the
		# device flags minimal.
		filter-lto

		# (b) the exact 0.2.15 wave32 sets of cmake/hip_backend.cmake:13-15 -
		# validated (gfx1100 RX 7900 XT/XTX, gfx1201 RX 9070), community-validated
		# (gfx1101 RX 7800 XT, gfx1200 RX 9060 XT), unvalidated-but-built (the rest;
		# CMake prints a warning for those).  Anything else - every gfx9xx CDNA part
		# for example - is a CMake FATAL_ERROR, so filter here and die early with a
		# readable message when make.conf selects nothing supported.
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

		# (c) a clang-compatible subset of CXXFLAGS for the device compiler; extend
		# the skip list if a newer amdclang rejects more gcc-only switches.
		local hip_flags='' f
		for f in ${CXXFLAGS}; do
			case ${f} in
				-fvect-cost-model=*) ;;	# gcc-only cost-model knob
				-flto*|-ffat-lto-objects) ;;	# LTO: see (e) / filter-lto above
				*) hip_flags+=" ${f}" ;;
			esac
		done
		hip_flags="${hip_flags# }"

		# the file (f) points at: ${T} is outside the sources and cmake only reads it
		# (its path is handed over as -DCMAKE_PROJECT_INCLUDE=).
		cat > "${T}/strata-gentoo-flags.cmake" <<-EOF || die
			set(CMAKE_HIP_FLAGS_${CMAKE_BUILD_TYPE^^} "-DNDEBUG" CACHE STRING "" FORCE)
			set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE^^} "-DNDEBUG" CACHE STRING "" FORCE)
			set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE^^} "-DNDEBUG" CACHE STRING "" FORCE)
		EOF

		mycmakeargs+=(
			# (a) force the AMD HIP path: hipconfig on this box reports the NVIDIA
			# flavour and CMake would pick nvcc.
			-DCMAKE_HIP_PLATFORM=amd
			-DCMAKE_HIP_COMPILER="${rocm_root}/lib/llvm/bin/amdclang++"
			-DCMAKE_HIP_COMPILER_ROCM_ROOT="${rocm_root}"
			# the hip/hipblas/rocblas/hipblaslt CONFIG packages: the ROCm install
			# first, the ::gentoo ones in /usr/lib64/cmake stay visible behind it.
			-DCMAKE_PREFIX_PATH="${rocm_root}"
			-DCMAKE_HIP_FLAGS="--rocm-path=${rocm_root} --rocm-device-lib-path=${rocm_root}/lib/llvm/amdgcn/bitcode${hip_flags:+ ${hip_flags}}"
			# (f) CMake refills its own per-config HIP flags after the eclass wiped
			# them for C/CXX - ninja then puts "-O2 -g -DNDEBUG" AFTER our
			# CMAKE_HIP_FLAGS, and clang lets the last -O win: the device code would
			# build at -O2 (measured in build.ninja with a plain
			# -DCMAKE_HIP_FLAGS_<BUILD_TYPE>=, which the HIP module simply refills).
			# CMAKE_PROJECT_INCLUDE (CMake >=3.15) is the one supported hook that runs
			# at the end of every project() call, i.e. after the HIP module set its
			# defaults and before any target exists - so the per-config flags are set
			# there, to what upstream builds with: its own CMakeLists hard-sets
			# CMAKE_BUILD_TYPE Release (-O3 -DNDEBUG for every language, and it is
			# this ebuild that wipes it).  -O3 comes from our flags, NDEBUG from here;
			# the 49 asserts of the HIP sources stay compiled out as they are upstream
			# instead of aborting a prompt on a card, and no -g is added.
			-DCMAKE_PROJECT_INCLUDE="${T}/strata-gentoo-flags.cmake"
			# space-separated is fine: hip_backend.cmake normalizes it into its own
			# list and validates every element.
			-DCMAKE_HIP_ARCHITECTURES="${hips_archs}"
		)
		# (d) no tc-export here: the host C/CXX compiler stays the system gcc.
	fi

	cmake_src_configure
}

src_compile() {
	# With USE=test the whole target set (including the ctest binaries) is built;
	# otherwise build exactly the binaries src_install installs - which skips
	# strata-q36, strata-q36ref and the CUDA-shaped extras (a HIP device compile
	# of them would cost minutes for nothing).
	if use test; then
		cmake_src_compile
		return
	fi
	local targets=( strata-plan strata-gguf strata-dequant )
	if use rocm; then
		targets+=( strata strata-device strata-load strata-concurrent strata-overlap )
	fi
	cmake_src_compile "${targets[@]}"
}

src_install() {
	local libdir="$(strata_prefix)"

	# The project has zero install() rules (verified in 0.2.15: no `install(` in
	# CMakeLists.txt or cmake/), and with Ninja CMake then creates no 'install'
	# target at all - cmake_src_install would die with "ninja: error: unknown
	# target 'install'".  Everything goes in by hand.
	einstalldocs

	# Always built (portable targets, no toolkit needed): the planner self-check,
	# the GGUF reader and the scalar dequantizer.
	dobin "${BUILD_DIR}"/strata-plan
	dobin "${BUILD_DIR}"/strata-gguf
	dobin "${BUILD_DIR}"/strata-dequant

	# The engine and its GPU diagnostics exist only inside the CUDA/HIP engine
	# block (cmake/strata_gpu_engine.cmake).
	if use rocm; then
		# The diagnostics go to /usr/bin (no name clash); the engine itself does
		# not - it becomes the engine component below.
		dobin "${BUILD_DIR}"/strata-device
		dobin "${BUILD_DIR}"/strata-load
		dobin "${BUILD_DIR}"/strata-concurrent
		dobin "${BUILD_DIR}"/strata-overlap

		strata_engine_component
	fi

	if use python; then
		# ---- the terminal: the strata command, the API server, the model
		# installer.  Data only: no pip, no python-install, no venv.  Keeping
		# upstream's archive layout is also what makes `strata install` run
		# setup.py in its download-only Control mode instead of the
		# source-checkout mode that would try to build an engine into /usr.
		insinto "/${libdir}"
		doins strata.py setup.py requirements.txt requirements-monitor.txt
		doins -r strata_cli strata_control serve
		doins -r data

		insinto "/${libdir}/bench"
		doins -r bench/prompts

		insinto "/${libdir}/bin"
		doins bin/strata
		# the tarball ships bin/strata 0775, the git checkout 0755: keep the mode
		# explicit rather than inherit it from the umask
		fperms 0755 "/${libdir}/bin/strata"

		insinto "/${libdir}/ref"
		doins ref/load.py

		insinto "/${libdir}/docs"
		doins docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md

		# The tools modules the terminal's own paths import - the same closure
		# packaging/package_terminal.py stages into its archive (setup_tool_imports
		# + local_imports over INSTALLER_TOOLS / INSTALLER_SCRIPTS / TOOLS_SUBSET),
		# computed from the sources instead of guessed at.  chat.py and highlight.py
		# are added by name: the closure misses them (no terminal path imports them
		# by name), but they are reachable from the tools package.
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

		# The terminal's own version label: strata_cli.version() reads exactly this
		# file, and the launcher treats the install as a Terminal bundle only when
		# VERSION and python/ are both there - so it names the Portage revision.
		echo "${PV}" > "${D}/${libdir}/VERSION" || die

		# The interpreter the terminal runs on: bin/strata looks for
		# <root>/python/bin/python3 (upstream's private runtime of its terminal
		# archive) before anything on PATH.  These shims start the implementation
		# python_setup picked, with the user's site-packages kept out.
		strata_python_shims "${D}/${libdir}/python/bin" "/${libdir}/python/bin"

		# /usr/bin/strata is not a symlink to bin/strata: it links the packaged
		# engine into the user's own component store first (see the header and
		# files/${PN}-${MY_PV}-strata.in), then runs the terminal.
		strata_launcher
	fi

	# No env.d on purpose: the terminal finds its interpreter through the python/
	# shim above, and a system-wide STRATA_COMPONENTS_DIR is what broke every
	# install and the desktop app (#533, Niko1221/Strata#5591).
}

# strata_python_shims <staging dir> <image path>: the python, python3 and
# ${EPYTHON} launchers that start the implementation python_setup picked, without
# the user's site-packages - upstream's bundle layout under python/bin/, which is
# where bin/strata looks first.
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

# strata_launcher: /usr/bin/strata from files/${PN}-${MY_PV}-strata.in - links the
# packaged engine store into the user's component store, then runs <prefix>/bin/strata.
strata_launcher() {
	local prefix="/$(strata_prefix)"
	sed -e "s|@NAME@|strata|g" \
		-e "s|@WHAT@|the Strata terminal|g" \
		-e "s|@PV@|${PV}|g" \
		-e "s|@SYSCOMPS@|${prefix}/components|g" \
		-e "s|@TARGET@|${prefix}/bin/strata|g" \
		"${FILESDIR}/${PN}-${MY_PV}-strata.in" > "${T}/strata" || die
	newbin "${T}/strata" strata
}

# strata_engine_component: the built engine as an engine component package
# (strata_control/engine_component.py, packaging/ENGINE-COMPONENT.md).  The
# terminal refuses to start an engine that is not such a package: its manifest
# (engine-component.json, schema 2) hashes every file of the component and
# resolve() re-checks them on every start, so a bare /usr/bin symlink is not a
# supported installation at all - --engine <path> is the only bypass.
#
# The tuning tables go in with it: packaging/stage_tuning.py records the hipBLASLt
# version the package carries in BUILD.json (package_engine.py refuses the package
# without that field) and copies tools/hip/<arch>-hipblaslt-<version>.txt for this
# build's architectures into <stage>/tuning/.
strata_engine_component() {
	local libdir="$(strata_prefix)"
	local rocm_root stage archs arch arch_args allow_libs

	# The type of this cache entry is UNINITIALIZED (hip_backend.cmake only writes
	# the CACHE STRING default when the variable is undefined, and this ebuild always
	# defines it with -D) - so the type is matched as "anything but =", not as STRING.
	archs=$(sed -n 's/^CMAKE_HIP_ARCHITECTURES:[^=]*=\(.*\)/\1/p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)
	archs="${archs//;/ }"
	[[ -n ${archs} ]] || die "cannot read CMAKE_HIP_ARCHITECTURES from ${BUILD_DIR}/CMakeCache.txt"

	stage="${T}/engine-stage"
	mkdir -p "${stage}" || die
	cp "${BUILD_DIR}/strata" "${stage}/" || die

	# the hipBLASLt version of the ROCm tree the engine links against (1.4.1 ->
	# 100401 here, so tools/hip/gfx1100-hipblaslt-100401.txt is the table that
	# lands in the package) and the tables for this build's architectures
	rocm_root="$(strata_rocm_root)"
	"${PYTHON:-python3}" "${S}/packaging/stage_tuning.py" \
		--stage "${stage}" --archs "${archs}" --rocm "${rocm_root}" || die

	# package_engine.py refuses a component whose binaries import libraries that
	# are neither bundled nor listed as OS or GPU-driver libraries
	# (packaging/binary_deps.py).  The ROCm runtime and BLAS libraries are
	# RDEPEND-provided and their libdir is in the ldconfig cache, so they are named
	# here instead of copied into the package.
	allow_libs=(
		--allow-lib libamdhip64.so.7
		--allow-lib libhipblas.so.3
		--allow-lib libhipblaslt.so.1
		--allow-lib librocblas.so.5
	)

	arch_args=()
	for arch in ${archs}; do
		# --arch is repeatable; passed separately so each lands as its own value in
		# the manifest's archs list.
		arch_args+=(--arch "${arch}")
	done

	# strata-vision is not built (pkg_setup dies on USE=vision), so the engine
	# ships text-only on purpose and the packager is told so.
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

	# estrip (which runs after src_install) would break the component's integrity:
	# engine-component.json records the binaries' SHA-256, so the engine must reach
	# the image unstripped.
	dostrip -x "/${libdir}/components/engine"
}

pkg_preinst() {
	# remembered for pkg_postinst: an older sci-misc/strata exported
	# STRATA_COMPONENTS_DIR through /etc/env.d/90strata, and the file is still
	# here while the old version is being replaced.
	[[ -f ${EROOT}/etc/env.d/90strata ]] && STRATA_OLD_ENVD=1
	return 0
}

pkg_postinst() {
	local libdir="$(strata_prefix)"

	elog "sci-ml/strata-${PV} installs under /${libdir} and its command is /usr/bin/strata -"
	elog "the layout ::pf4public's sci-misc/strata has, so a component store and run configs"
	elog "made by that package keep working.  sci-misc/strata must be unmerged before this"
	elog "one is installed (both own /${libdir}/** and /usr/bin/strata)."

	if use rocm; then
		elog "The engine is a component package under /${libdir}/components/engine (schema 2:"
		elog "its manifest hashes every file, resolve() re-checks them at every start).  The"
		elog "strata launcher links it into your own component store; start Strata through it."
		elog "Built for wave32 RDNA only: the targets are the intersection of AMDGPU_TARGETS"
		elog "with cmake/hip_backend.cmake's wave32 set (every gfx9xx CDNA part is a CMake error)."
		if use mmq; then
			elog "MMQ prompt kernels (llama.cpp's, with the Q4_K/Q5_K/Q5_1/Q6_K instances for"
			elog "Unsloth's UD K-quants) and the fused prompt experts are ON (USE=mmq), with the"
			elog "dense GGUF MMQ projections (STRATA_DENSE_MMQ)."
		else
			elog "MMQ is OFF (USE=-mmq): Strata's own FP16 prompt kernels only."
		fi
		elog "hipBLASLt tuning: this package carries the table of your ROCm tree in the"
		elog "component's tuning/ folder, but upstream only switches tables on by itself"
		elog "for gfx1151 (strata_control/hipblaslt_tuning.py: AUTO_ARCHS).  On gfx1100 it is"
		elog "opt-in: STRATA_HIPBLASLT_TABLE=on (or \"hipblaslt_table\": true in the run"
		elog "config).  Try it - the measured prompt gain on the family is large."
	fi

	if ! use rocm; then
		ewarn "No engine was built: USE=rocm is off, so this is the portable CPU tool set"
		ewarn "only (strata-plan / strata-gguf / strata-dequant) and no engine component."
	fi

	if use python; then
		elog "Strata writes only per-user folders: \$XDG_DATA_HOME/strata (components/, install/"
		elog "with the run configs and logs), \$XDG_CONFIG_HOME/strata, \$XDG_STATE_HOME/strata."
		elog "STRATA_COMPONENTS_DIR is deliberately NOT exported: pointing it at the read-only"
		elog "/${libdir}/components sent every write into /usr (#533).  Never run Strata as root."
		optfeature "structured output for serve/ API calls (response_format: json_schema)" \
			dev-python/jsonschema dev-python/referencing
		optfeature "the launch/monitor screens (without them strata launch prints plain lines)" \
			dev-python/rich dev-python/textual
	fi
}

src_test() {
	# Many of the ctest cases need a GPU or a model pack and skip themselves with a
	# message (SKIP_RETURN_CODE 77) - harmless noise.  check_amdgpu (rocm eclass)
	# dies with a friendly message when the portage user cannot read/write /dev/kfd
	# and the render nodes.
	use rocm && check_amdgpu
	cmake_src_test
}
