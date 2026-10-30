"""Analytic cached one-token decode work, not a hardware latency estimate."""


def decode_inventory(config, source_context, *, latent, batch=1, bytes_per_element=4):
    if (type(source_context) is not int or not 1<=source_context<=config.max_seq
        or type(batch) is not int or batch<1 or bytes_per_element not in (2,4,8)):
        raise ValueError('Supported positive source context/batch and explicit cache precision required')
    d=config.dim;kv=config.kv_heads*(d//config.heads);layers=config.layers
    # Two dependent body calls append source then latent. Their inclusive key
    # lengths are2T-1 and2T; ordinary appends just its source entry at lengthT.
    key_lengths=[2*source_context-1,2*source_context] if latent else [source_context]
    body_projections=batch*layers*len(key_lengths)*(4*d*d+4*d*kv+6*d*config.ffn_dim)
    attention=batch*layers*4*d*sum(key_lengths)
    head=2*batch*d*config.vocab_size
    slots_per_source=2 if latent else 1
    kv_per_source=batch*slots_per_source*layers*2*kv*bytes_per_element
    return dict(source_context=source_context,batch=batch,body_calls_per_prediction=len(key_lengths),
        inclusive_attention_key_lengths=key_lengths,body_projection_matmul_flops=body_projections,
        attention_matmul_flops=attention,vocabulary_projection_matmul_flops=head,
        forward_matmul_flops=body_projections+attention+head,
        kv_bytes_per_source_token=kv_per_source,kv_bytes_at_context=kv_per_source*source_context,
        bytes_per_element=bytes_per_element,hardware_time_measured=False,
        scope='Exact cached sequential decode; one vocabulary projection per natural prediction. '
              'FMA=2; excludes elementwise operations, cache allocation/copies and kernel/communication overhead. '
              'No cache sharing, approximate prefill or unsupported context is assumed.')
