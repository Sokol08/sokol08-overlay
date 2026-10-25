# Copyright 2023-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

# Crates come from ::gentoo's prebundled archives for both graphs (core and
# python), the same ones its 0.23.3-r1 uses. Every vendored crate matches
# crates.io byte for byte, and the dependent crate licenses below match the
# vendored set. verified 2026-10-16
# -r90: ours is single-impl with ~arm64, so ::gentoo's 0.23.3-r1 must not
# outrank it.
# -r91: fork in sokol08-overlay. The stock build's final wheel carries
# KMP-OpenMP runtime calls (parallel code from the vendored crates) while
# losing its own DT_NEEDED libomp.so on this machine (clang + -fopenmp
# build flags, -Wl,--as-needed on the link line), so `import tokenizers`
# died on undefined symbol __kmpc_*. r91 keeps libomp in DT_NEEDED
# (link-args + patchelf backstop), declares the OpenMP runtime provider,
# and dies instead of installing a wheel that cannot be imported.
# verified 2026-10-25

EAPI=8

DISTUTILS_USE_PEP517=maturin
PYTHON_COMPAT=( python3_{12..14} )
DISTUTILS_EXT=1
DISTUTILS_SINGLE_IMPL=1
RUST_MIN_VER="1.88.0"

CRATES="
"

inherit cargo distutils-r1

DESCRIPTION="Implementation of today's most used tokenizers"
HOMEPAGE="https://github.com/huggingface/tokenizers"
SRC_URI="
	https://github.com/huggingface/${PN}/archive/refs/tags/v${PV}.tar.gz
	-> ${P}.gh.tar.gz
	${CARGO_CRATE_URIS}
	https://dev.gentoo.org/~tupone/distfiles/${P}-crates.tar.xz
	https://dev.gentoo.org/~tupone/distfiles/${PN}-python-${PV}-crates.tar.xz
"

LICENSE="Apache-2.0"
# Dependent crate licenses
LICENSE+="
	Apache-2.0 Apache-2.0-with-LLVM-exceptions BSD BSD-2 CDLA-Permissive-2.0 ISC MIT MPL-2.0 Unicode-3.0 ZLIB
"
SLOT="0"
KEYWORDS="~amd64"

# the wheel references KMP-OpenMP runtime symbols; its DT_NEEDED must
# resolve against the machine's libomp (KMP ABI) runtime provider
RDEPEND="
	dev-libs/oniguruma
	llvm-runtimes/openmp
"
# dev-util/patchelf is the backstop for the src_install() audit: pin
# DT_NEEDED libomp in with patchelf if the final link drops it under --as-needed
BDEPEND="
	test? ( sci-ml/datasets[${PYTHON_SINGLE_USEDEP}] )
	$(python_gen_cond_dep '
		dev-python/setuptools-rust[${PYTHON_USEDEP}]
	')
	dev-util/patchelf
"

EPYTEST_PLUGINS=( )
distutils_enable_tests pytest

QA_FLAGS_IGNORED=".*/site-packages/tokenizers/.*so"

src_unpack() {
	cargo_src_unpack
}

pkg_setup() {
	python-single-r1_pkg_setup
	rust_pkg_setup
}

src_prepare() {
	default
	cd bindings/python || die
	eapply "${FILESDIR}"/${PN}-0.21.2-test.patch
	distutils-r1_src_prepare
}

src_configure() {
	cd tokenizers || die
	cargo_src_configure
	cd ../bindings/python || die
	distutils-r1_src_configure
}

src_compile() {
	# keep libomp.so in the final .so's DT_NEEDED no matter what the
	# as-needed link line does: link-args are appended last, so -lomp is
	# scanned after all objects and its NEEDED entry cannot be pruned
	export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,--no-as-needed -C link-arg=-lomp"
	export RUSTONIG_SYSTEM_LIBONIG=1
	cd tokenizers || die
	cargo_src_compile
	cd ../bindings/python || die
	distutils-r1_src_compile
}

src_test() {
	cd bindings/python || die
	local -x EPYTEST_IGNORE=( benches/test_tiktoken.py )
	local -x EPYTEST_DESELECT=(
		tests/bindings/test_encoding.py::TestEncoding::test_char_to_token
		tests/bindings/test_encoding.py::TestEncoding::test_char_to_word
		tests/bindings/test_encoding.py::TestEncoding::test_invalid_truncate_direction
		tests/bindings/test_encoding.py::TestEncoding::test_n_sequences
		tests/bindings/test_encoding.py::TestEncoding::test_sequence_ids
		tests/bindings/test_encoding.py::TestEncoding::test_token_to_chars
		tests/bindings/test_encoding.py::TestEncoding::test_token_to_sequence
		tests/bindings/test_encoding.py::TestEncoding::test_token_to_word
		tests/bindings/test_encoding.py::TestEncoding::test_truncation
		tests/bindings/test_encoding.py::TestEncoding::test_word_to_chars
		tests/bindings/test_encoding.py::TestEncoding::test_word_to_tokens
		tests/bindings/test_models.py::TestWordLevel::test_instantiate
		tests/bindings/test_models.py::TestWordPiece::test_instantiate
		tests/bindings/test_processors.py::TestByteLevelProcessing::test_processing
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_async_methods_existence
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_basic_encoding
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_concurrency
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_decode
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_encode
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_error_handling
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_large_batch
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_numpy_inputs
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_various_input_formats
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_performance_comparison
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_with_special_tokens
		tests/bindings/test_tokenizer.py::TestAsyncTokenizer::test_with_truncation_padding
		tests/bindings/test_tokenizer.py::TestTokenizer::test_encode_add_special_tokens
		tests/bindings/test_tokenizer.py::TestTokenizer::test_encode_formats
		tests/bindings/test_tokenizer.py::TestTokenizer::test_encode_special_tokens
		tests/bindings/test_tokenizer.py::TestTokenizer::test_decode_skip_special_tokens
		tests/bindings/test_tokenizer.py::TestTokenizer::test_decode_stream_fallback
		tests/bindings/test_tokenizer.py::TestTokenizer::test_from_pretrained
		tests/bindings/test_tokenizer.py::TestTokenizer::test_from_pretrained_revision
		tests/bindings/test_tokenizer.py::TestTokenizer::test_splitting
		tests/bindings/test_trainers.py::TestUnigram::test_continuing_prefix_trainer_mismatch
		tests/bindings/test_trainers.py::TestUnigram::test_train
		tests/bindings/test_trainers.py::TestUnigram::test_train_parallelism_with_custom_pretokenizer
		tests/documentation/test_pipeline.py::TestPipeline::test_bert_example
		tests/documentation/test_pipeline.py::TestPipeline::test_pipeline
		tests/documentation/test_quicktour.py::TestQuicktour::test_quicktour
		tests/documentation/test_tutorial_train_from_iterators.py::TestTrainFromIterators::test_datasets
		tests/documentation/test_tutorial_train_from_iterators.py::TestTrainFromIterators::test_gzip
		tests/implementations/test_byte_level_bpe.py::TestByteLevelBPE::test_add_prefix_space
		tests/implementations/test_byte_level_bpe.py::TestByteLevelBPE::test_basic_encode
		tests/implementations/test_byte_level_bpe.py::TestByteLevelBPE::test_lowerspace
		tests/implementations/test_byte_level_bpe.py::TestByteLevelBPE::test_multiprocessing_with_parallelism
		tests/implementations/test_bert_wordpiece.py::TestBertWordPieceTokenizer::test_basic_encode
		tests/implementations/test_bert_wordpiece.py::TestBertWordPieceTokenizer::test_multiprocessing_with_parallelism
		tests/implementations/test_char_bpe.py::TestCharBPETokenizer::test_basic_encode
		tests/implementations/test_char_bpe.py::TestCharBPETokenizer::test_decoding
		tests/implementations/test_char_bpe.py::TestCharBPETokenizer::test_lowercase
		tests/implementations/test_char_bpe.py::TestCharBPETokenizer::test_multiprocessing_with_parallelism
		tests/test_serialization.py::TestSerialization::test_full_serialization_albert
		tests/test_serialization.py::TestSerialization::test_str_big
	)
	distutils-r1_src_test
}

src_install() {
	cd tokenizers || die
	cd ../bindings/python || die
	distutils-r1_src_install

	# audit the staged wheel: its extension must carry DT_NEEDED libomp.so,
	# else its KMP __kmpc_* calls are unresolvable at import time. If the
	# link still dropped it (e.g. the toolchain overrode RUSTFLAGS), pin
	# the NEEDED in with patchelf and re-check; then die — a wheel that
	# cannot be imported is worse than a failed build.
	local so
	so=$(find "${D}" -type f -name 'tokenizers*.so' -print -quit)
	[ -n "${so}" ] || die "no tokenizers extension .so staged"
	if ! readelf -d "${so}" | grep -q 'libomp\.so'; then
		einfo "src_install: DT_NEEDED libomp missing; pinning with patchelf"
		patchelf --add-needed libomp.so "${so}" || die "patchelf --add-needed failed"
	fi
	readelf -d "${so}" | grep -q 'libomp\.so' || die "staged tokenizers .so still lacks DT_NEEDED libomp; refusing to install an extension that cannot be imported"
}
