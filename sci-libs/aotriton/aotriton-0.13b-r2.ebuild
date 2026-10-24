# Copyright 1999-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2 or later

EAPI=8

MY_PN="aotriton"
MY_P="${MY_PN}-${PV}"

# Install-only: provides the prebuilt gfx110x AOT-Triton flash-attention
# runtime (C++ API + .so + kernel images, no python bindings). PyTorch
# 2.14.1 pins the 0.13b v3::flash C API (aotriton::v3::flash::VarlenType);
# the 0.14.x generation renames it to VarlenMode and cannot be used.
# Upstream labels the ROCm-10 generation shim (reporting HIP 7.15/7.16)
# "-rocm7.15"; keep it as "-rocm10.0" distfile for clarity.
# Verified offline against /var/cache/distfiles 2026-10-24:
#  - .so ldd: all NEEDED resolve (libamdhip64.so.7 = hip 7.16 runtime)
#  - flash.h: identical to the 0.13b tag API (VarlenType, cu_seqlens_q/k)
SRC_URI="
	https://github.com/ROCm/aotriton/releases/download/${PV}/${MY_P}-manylinux_2_28_x86_64-rocm7.15-shared.tar.gz
		-> ${MY_P}-manylinux_2_28_x86_64-rocm10.0-shared.tar.gz
	https://github.com/ROCm/aotriton/releases/download/${PV}/${MY_P}-images-amd-gfx110x.tar.gz
"

DESCRIPTION="Ahead of Time (AOT) Triton: prebuilt gfx110x flash-attention runtime (C++ only)"
HOMEPAGE="https://github.com/ROCm/aotriton"
LICENSE="MIT"
SLOT="0/${PV%b}"
KEYWORDS="-* ~amd64"

# Prebuilt artifact: do not strip; it conflicts with the other providers.
RESTRICT="strip"
RDEPEND="
	!!sci-libs/aotriton-bin
	sys-libs/glibc
	sys-devel/gcc
	app-arch/xz-utils
	>=dev-util/hip-6.4:=
	<dev-util/hip-11:=
"
DEPEND="${RDEPEND}"

S="${WORKDIR}/${MY_PN}"

src_unpack() {
	# Both assets unpack into the same aotriton/ tree:
	#   shim  -> aotriton/{include,lib/{libaotriton_v2.so*,cmake}}
	#   images-> aotriton/lib/aotriton.images/amd-gfx110x
	unpack "${MY_P}-manylinux_2_28_x86_64-rocm10.0-shared.tar.gz"
	unpack "${MY_P}-images-amd-gfx110x.tar.gz"
}

# no src_prepare(): the phase default (user-patch hooks) is used as-is.
src_compile() { :; }

src_install() {
	doheader -r "${S}"/include/*
	insinto /usr/$(get_libdir)
	doins "${S}"/lib/libaotriton_v2.so.0.13.0
	dodir /usr/$(get_libdir)
	ln -sf libaotriton_v2.so.0.13.0 "${D}"/usr/$(get_libdir)/libaotriton_v2.so
	doins -r "${S}"/lib/cmake
	doins -r "${S}"/lib/aotriton.images
}
