import os
import pytest
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from scripts.check_ordinary8_cpu import ordinary_reference_config


def test_eight_layer_control_sets_the_upstream_geometry_explicitly(monkeypatch):
    root=os.environ.get('PARCAE_REFERENCE_ROOT')
    if not root:pytest.skip('Pinned independent Parcae checkout required')
    monkeypatch.syspath_prepend(root)
    c=ParcaeCandidateConfig(dim=640,heads=5,ffn_dim=2560,vocab_size=32000,core=4)
    config=ordinary_reference_config(c)
    assert config.n_layer==8
    assert config.n_embd==640 and config.intermediate_size==2560
    assert config.num_attention_heads==config.num_key_value_heads==5
    assert config.tie_embeddings and config.block_size==2048
