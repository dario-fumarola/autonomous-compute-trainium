"""Count full authored calculations visible within a packed causal window."""
from collections import Counter


def intact_exposure(spans,end_position,context=512):
    if end_position<1 or context<1:raise ValueError('Positive exposure geometry required')
    ops=Counter();answers=0;distinct=set()
    for span in spans:
        start=span['start'];pl,pr=span['premise_tokens'];tl,tr=span['target_tokens']
        if not (start>=0 and 0<=pl<pr<=tl<tr):raise ValueError('Ordered original premise/target spans required')
        first=start+1+tl;last=start+1+tr
        window=((first-1)//context)*context
        if last<=end_position and start+1+pl>=window and last<=window+context+1:
            ops[str(span['operations'])]+=1;answers+=tr-tl;distinct.add(span['expression_key'])
    return dict(intact_calculation_counts=dict(ops),intact_answer_tokens=answers,distinct_expressions=len(distinct))
