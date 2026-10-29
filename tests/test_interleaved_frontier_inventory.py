import pytest
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import matrix_work
from scripts.run_interleaved_frontier_qualification import SHAPES


@pytest.mark.parametrize('shape,parameters,work',[
    ('wider',51743808,36521902080),('deeper',51654848,42902487040)])
def test_exact_inventoried_conventional_alternatives(shape,parameters,work):
    c=ModelConfig(**SHAPES[shape],vocab_size=32000,max_seq=512)
    assert c.parameter_count()+c.vocab_size*c.dim==parameters<=51747328
    assert c.dim//c.heads==64 and c.heads==c.kv_heads
    assert matrix_work(c,512,0,mode='ordinary')['forward_matmul_flops']==work
