from pathlib import Path

from scripts.qualify_bucketed_native import parser


def test_common_launcher_arguments_are_accepted_without_changing_data_plan():
    args=parser().parse_args(['--corpus','/work/data-v1/corpus.json',
        '--root','/work/instance-nvme','--output','/work/run',
        '--plan','/work/plan.json','--plan-sha','fixed','--comm-port','12345',
        '--mode','warm'])
    assert args.corpus==Path('/work/data-v1/corpus.json')
    assert args.plan==Path('/work/plan.json')
    assert args.root==Path('/work/instance-nvme')
    assert args.mode=='warm'
