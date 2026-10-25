"""Finite curve topology and guards against changing history across cooldowns."""


def stop_on_failed_future(future,failed):
    # Set the paired stop before an executor's context manager waits for its
    # sibling. Native/CPU owners observe this event and clean up only themselves.
    if future.cancelled() or future.exception() is not None:failed.set()


def curve_nodes(cases):
    required={'one-1b','fixed-1b','one-work-1b','one-2b','fixed-2b','one-work-2b'}
    if set(cases)!=required:raise ValueError('All six predeclared curve endpoints are required')
    nodes={}
    for family in ['one','fixed']:
        key=family+'-2b'
        nodes[key+'-prefix']=dict(case=key,end=960,parent=None)
        nodes[key+'-stable']=dict(case=key,end=1920,parent=key+'-prefix')
        nodes[key+'-finish']=dict(case=key,end=cases[key]['total_updates'],parent=key+'-stable')
        child=family+'-1b'
        nodes[child+'-finish']=dict(case=child,end=cases[child]['total_updates'],parent=key+'-prefix')
    for label,parent in [('1b','one-2b-prefix'),('2b','one-2b-stable')]:
        key='one-work-'+label
        nodes[key+'-finish']=dict(case=key,end=cases[key]['total_updates'],parent=parent)
    validate_curve(cases,nodes)
    return nodes


def validate_curve(cases,nodes):
    terminals={key:0 for key in cases}
    identity=['arm','seed','initial_state','peak_lr','warmup_targets','depth_table_sha256','work_table_sha256']
    for name,node in nodes.items():
        case=cases[node['case']];end=node['end']
        if type(end) is not int or end not in case['segment_ends'] or not 0<end<=case['total_updates']:
            raise ValueError('Every node must be an explicit finite scientific segment')
        if end==case['total_updates']:terminals[node['case']]+=1
        if node['parent'] is None:continue
        if node['parent'] not in nodes:raise ValueError('Missing exact stable parent')
        parent=nodes[node['parent']];source=cases[parent['case']]
        if not parent['end']<end or parent['end']>min(source['cooldown_start_update'],case['cooldown_start_update']):
            raise ValueError('Branches must leave an untouched stable prefix before either cooldown')
        if any(source[key]!=case[key] for key in identity):raise ValueError('Shared prefix cannot change weights,source/depth assignment or optimizer recipe')
        if parent['case']!=node['case'] and case.get('parent')!={'case_id':parent['case'],'update':parent['end']}:
            raise ValueError('Cross-endpoint branch must name its exact source case/update')
    if any(count!=1 for count in terminals.values()):raise ValueError('Every declared endpoint must finish exactly once')
    return sum(node['end']-(nodes[node['parent']]['end'] if node['parent'] else 0) for node in nodes.values())
