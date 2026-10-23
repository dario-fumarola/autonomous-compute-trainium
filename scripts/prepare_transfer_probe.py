"""Freeze natural exam examples for a small frozen-feature transfer screen."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from compute_trainium.training import file_sha256


def sha(s): return hashlib.sha256(s.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--arc', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    selected, source_hashes = [], {}
    seen_questions, seen_articles = set(), set()
    for split in ['train', 'validation']:
        path = args.arc/(split+'-00000-of-00001.parquet')
        source_hashes[str(path)] = file_sha256(path)
        for row in pq.read_table(path).to_pylist():
            choices = row['choices']['text']; labels = row['choices']['label']
            if len(choices) != 4: continue
            identity = sha(' '.join(row['question'].split()).lower())
            if identity in seen_questions: continue
            seen_questions.add(identity)
            prompt = 'Question: '+row['question']+'\n'+'\n'.join(f'{"ABCD"[i]}. {v}' for i,v in enumerate(choices))+'\nAnswer:'
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids)>512: continue
            selected.append({'task': 'arc_challenge', 'split': split, 'id': sha('arc:'+row['id']),
                'group': identity, 'ids': ids, 'answer': labels.index(row['answerKey'])})
    for split, limit in [('train',256),('validation',128)]:
        path = args.sources/('race_high_'+split+'.jsonl')
        source_hashes[str(path)] = file_sha256(path)
        pool = []
        for raw in path.read_text().splitlines():
            row = json.loads(raw); article_id = sha(' '.join(row['article'].split()).lower())
            if article_id in seen_articles: continue
            choices = []
            for problem in row['problems']:
                if len(problem['options'])!=4: continue
                prompt = 'Passage: '+row['article']+'\nQuestion: '+problem['question']+'\n'+'\n'.join(f'{"ABCD"[i]}. {v}' for i,v in enumerate(problem['options']))+'\nAnswer:'
                ids = tok.encode(prompt, add_special_tokens=False)
                if len(ids)>512: continue
                identity = sha(article_id+':'+problem['question'])
                choices.append({'task':'race_high', 'split':split, 'id':identity,
                                'group':article_id, 'ids':ids, 'answer':'ABCD'.index(problem['answer'])})
            if choices: pool.append(min(choices,key=lambda d:sha('question-selection-v1:'+d['id'])))
        chosen = sorted(pool,key=lambda d:sha('passage-selection-v1:'+d['group']))[:limit]
        if len(chosen)!=limit: raise ValueError('Not enough eligible natural passages')
        selected.extend(chosen);seen_articles.update(d['group'] for d in chosen)
    groups = {split: {d['group'] for d in selected if d['split']==split} for split in ['train','validation']}
    if groups['train'] & groups['validation']: raise AssertionError('Source-group overlap')
    ids = [tok.encode(v,add_special_tokens=False) for v in 'ABCD']
    if any(len(v)!=1 for v in ids): raise AssertionError('Answer labels must be single tokenizer tokens')
    data = args.output/'examples.json'
    data.write_text(json.dumps(selected)+'\n')
    manifest = {'status':'FROZEN_NATURAL_TRANSFER_PROBE', 'examples_sha256':file_sha256(data),
        'source_sha256': source_hashes, 'race_revision':'e30efe648089df42c548e93faa9c5f1816e2c44f',
        'arc_revision':'210d026faf9955653af8916fad021475a3f00453', 'answer_token_ids':[v[0] for v in ids],
        'counts':{task:{split:sum(d['task']==task and d['split']==split for d in selected) for split in ['train','validation']} for task in ['arc_challenge','race_high']},
        'source_group_overlap':0, 'max_prompt_tokens':512, 'truncation':False,
        'selection':'All eligible 4-option ARC train/validation; 256/128 SHA-selected RACE passages with one SHA-selected eligible question each; no official TEST.',
        'human_exam_data':True, 'generated_research_examples':False}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2))


if __name__=='__main__': main()
