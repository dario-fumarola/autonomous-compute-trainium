"""Accept the isolated launcher's legacy corpus flag without opening that corpus."""
def without_launcher_corpus(arguments):
    result=list(arguments)
    if '--corpus' not in result:return result
    if result.count('--corpus')!=1:raise ValueError('Duplicate launcher corpus flag')
    index=result.index('--corpus')
    if index+1>=len(result) or result[index+1]!='/work/data-v1/corpus.json':
        raise ValueError('Unexpected legacy launcher corpus override')
    return result[:index]+result[index+2:]
