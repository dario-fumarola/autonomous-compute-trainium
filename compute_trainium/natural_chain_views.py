"""Provenance-preserving views of existing worked arithmetic;no answer creation."""
from dataclasses import dataclass
import hashlib
import json

from .source_arithmetic import check


@dataclass(frozen=True)
class ConclusionView:
    prompt:str
    target:str
    inputs:tuple
    targets:tuple
    target_mask:tuple
    termination_mask:tuple
    answer_ids:tuple
    prefix_ids:tuple
    visible_steps:int
    answer_start:int


def source_fragments(record):
    fragments=record['original_source_fragments']
    if len(fragments)<3 or not all(isinstance(s,str) and s for s in fragments):
        raise ValueError('Original premise,work and conclusion required')
    values=[check(s) for s in fragments]
    if any(x.value!=values[0].value for x in values) or values[0].operations<2:
        raise ValueError('Original arithmetic equality or minimum work fails')
    if not any(0<x.operations<values[0].operations for x in values[1:-1]):
        raise ValueError('Source intermediate simplification required')
    if values[-1].value in frozenset().union(*(x.literals for x in values[:-1])):
        raise ValueError('Conclusion already copied into a retained fragment')
    return fragments


def conclusion_view(record,tokenizer,visible_steps,*,context=128,bos=1,eos=2):
    """One original conclusion,masked at the normal decoder output.

    Formatting strings delimit the task. Every mathematical input/work/target
    is copied exactly from verified source fragments. Hidden spans are absent;
    padding follows the answer and cannot influence any scored causal prefix.
    """
    parts=source_fragments(record)
    if type(visible_steps) is not int or not 0<=visible_steps<=len(parts)-2:
        raise ValueError('Visible work must be an original prefix of the bridge')
    prompt='Evaluate the expression.\nExpression: '+parts[0]+'\n'
    if visible_steps:prompt+='Work: '+' = '.join(parts[1:1+visible_steps])+'\n'
    prompt+='Answer:\n';target=parts[-1]
    encoded=tokenizer.encode(prompt+target,add_special_tokens=False)
    prefix=tokenizer.encode(prompt,add_special_tokens=False).ids
    if encoded.ids[:len(prefix)]!=prefix or len(encoded.ids)<=len(prefix):
        raise ValueError('Task boundary changes tokenization;no silent realignment')
    # Offsets must not cause a scored token to contain hidden prompt text.
    if any(a<len(prompt) for a,b in encoded.offsets[len(prefix):]):
        raise ValueError('Answer token crosses the task boundary')
    answer=encoded.ids[len(prefix):];sequence=[bos,*encoded.ids,eos]
    if len(sequence)>context+1:raise ValueError('Original task exceeds fixed context;no truncation')
    missing=context+1-len(sequence);sequence += [eos]*missing
    mask=tuple(len(prefix)<=i<len(encoded.ids) for i in range(context))
    if sum(mask)!=len(answer):raise AssertionError('Every original conclusion token scored once')
    termination=tuple(i==len(encoded.ids) for i in range(context))
    if sum(termination)!=1:raise AssertionError('One explicit task termination target required')
    return ConclusionView(prompt,target,tuple(sequence[:-1]),tuple(sequence[1:]),mask,termination,
        tuple(answer),tuple([bos,*prefix]),visible_steps,len(prefix))


def normalized_answer(text):
    """Evaluation only:bounded scalar syntax,never unrestricted expressions."""
    import re
    value=text.strip().replace('−','-')
    value=re.sub(r'\.$','',value)
    if not re.fullmatch(r'[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*\d+)?',value):
        raise ValueError('Generated answer is not a scalar literal')
    return check(value).value


def semantic_key(record):
    return record['arithmetic_ast_group']


def deduplicate(records):
    # All versions of one literal-valued arithmetic tree form one task,even
    # when they occur on different hosts with different TeX/decimal formatting.
    kept={}
    for row in sorted(records,key=lambda r:r['id']):kept.setdefault(semantic_key(row),row)
    return list(kept.values())
