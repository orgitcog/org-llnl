import pytest
from bincfg import get_normalizer, Architectures, CFG, X86BaseNormalizer, X86InnerEyeNormalizer, X86SafeNormalizer, \
    X86DeepBinDiffNormalizer, X86DeepSemanticNormalizer, X86CompressedStatsNormalizer, X86HPCDataNormalizer,\
    JavaBaseNormalizer
from .manual_cfgs import get_all_manual_cfg_functions


# All of the normalizers to test per architecture
ARCH_NORMS = {
    Architectures.X86: {
        'base_norm': (X86BaseNormalizer, {}),
        'innereye': (X86InnerEyeNormalizer, {}),
        'safe': (X86SafeNormalizer, {}),
        'deepbindiff': (X86DeepBinDiffNormalizer, {}),
        'deepsemantic': (X86DeepSemanticNormalizer, {}),
        'compressed_stats': (X86CompressedStatsNormalizer, {}),
        'hpcdata': (X86HPCDataNormalizer, {}),
    },
    Architectures.JAVA: {
        'java_base': (JavaBaseNormalizer, {}),
    },
}


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_normalized(cfg_func):
    """Tests normalization of cfg's"""
    res = cfg_func(build_level='cfg')

    for norm_class, norm_kwargs in ARCH_NORMS[res['cfg'].architecture].values():
        cfg = res['cfg'].copy()
        assert not hasattr(cfg, 'tokens')

        normalizer = norm_class(**norm_kwargs)

        for inplace in [False, True]:
            using_tokens = {}
            norm_cfg = cfg.normalize(normalizer=normalizer, using_tokens=using_tokens, inplace=inplace)

            for block in norm_cfg.blocks:
                old_block = res['cfg'].get_block(block)
                assert block.asm_lines == normalizer.normalize(*old_block.asm_lines, cfg=res['cfg'], block=old_block)
            
            if inplace:
                assert cfg is norm_cfg
            else:
                assert cfg is not norm_cfg
                assert cfg != norm_cfg
                assert hash(cfg) != hash(norm_cfg)

            # Make sure the graph structure is all the same
            assert len(cfg.blocks) == len(norm_cfg.blocks)
            assert len(cfg.functions) == len(norm_cfg.functions)
            for f1, f2 in zip(cfg.functions, norm_cfg.functions):
                assert len(f1.blocks) == len(f2.blocks)

                for b1, b2 in zip(list(sorted(f1.blocks, key=lambda b: b.address)), list(sorted(f2.blocks, key=lambda b: b.address))):
                    assert b1.edges_in == b2.edges_in
                    assert b1.edges_out == b2.edges_out


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_normalized_from_inputs(cfg_func):
    """Tests the normalizers from inputs don't mess things up"""
    res = cfg_func(build_level='cfg')

    for norm_class, norm_kwargs in ARCH_NORMS[res['cfg'].architecture].values():
        normalizer = norm_class(**norm_kwargs)
        for input in res['inputs']:
            for tdict in [None, {}]:
                for make_str in [str, lambda x: x]:
                    new_cfg = CFG(input, normalizer=make_str(normalizer), using_tokens=tdict, metadata={'architecture': res['cfg'].metadata['architecture']})
                    assert res['cfg'].normalize(make_str(normalizer), inplace=False, using_tokens=tdict).update_metadata({'file_type': new_cfg.metadata['file_type']}) == new_cfg

