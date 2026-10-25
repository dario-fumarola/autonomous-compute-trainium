"""Software fixtures only;these are never emitted as research examples."""
from fractions import Fraction
import pytest
from compute_trainium.natural_chain_views import conclusion_view,normalized_answer,source_fragments,deduplicate

class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        class Encoded:pass
        out=Encoded();out.ids=[ord(c)+3 for c in text];out.offsets=[(i,i+1) for i in range(len(text))];return out


def record():return dict(id='a',arithmetic_ast_group='tree',original_source_fragments=['(8+7)*3','15*3','45'])


def test_hidden_work_never_enters_inputs_and_mask_is_only_ordered_original_answer():
    tokenizer=CharacterTokenizer();r=record()
    for count in [0,1]:
        view=conclusion_view(r,tokenizer,count,context=128)
        assert ('15*3' in view.prompt)==bool(count)
        assert view.target=='45' and '45' not in view.prompt
        assert [x for x,m in zip(view.targets,view.target_mask) if m]==list(view.answer_ids)
        assert view.inputs[view.answer_start]==view.prefix_ids[-1]
        assert sum(view.target_mask)==2
        assert sum(view.termination_mask)==1
        assert [x for x,m in zip(view.targets,view.termination_mask) if m]==[2]
        assert not any(a and b for a,b in zip(view.target_mask,view.termination_mask))
    with pytest.raises(ValueError):conclusion_view(r,tokenizer,2)
    with pytest.raises(ValueError):conclusion_view(r,tokenizer,0,context=8)


def test_checker_rejects_repaired_or_copied_answers_and_verifier_is_scalar_only():
    r=record();r['original_source_fragments'][-1]='46'
    with pytest.raises(ValueError):source_fragments(r)
    r=record();r['original_source_fragments'][0]='(45+0)*1'
    with pytest.raises(ValueError):source_fragments(r)
    assert normalized_answer(' -6/9. ')==Fraction(-2,3)
    assert normalized_answer('60.')==60
    for value in ['2+3','\\frac{2}{3}','answer is 5','5\nwork','1/0']:
        with pytest.raises(ValueError):normalized_answer(value)


def test_problem_level_deduplication_is_independent_of_source_host_and_order():
    a=record();b={**a,'id':'b'};c={**a,'id':'c','arithmetic_ast_group':'different'}
    assert [x['id'] for x in deduplicate([c,b,a])]==['a','c']
