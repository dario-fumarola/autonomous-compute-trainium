"""Read-only instrumentation of the independent pre-norm reference."""
import torch
import torch.nn.functional as F


def traced_unroll(model,tokens,repetitions,extra=None):
    n=tokens.shape[1];e=model.embedding(tokens)
    cos,sin=model.cos[:,:,:n].to(e.dtype),model.sin[:,:,:n].to(e.dtype)
    for i in range(model.prelude):e=model.blocks[i](e,cos,sin)
    states=[];h=torch.zeros_like(e)
    for r in range(repetitions):
        h=h+e
        for j in range(model.core):
            block=model.blocks[model.prelude+j] if r==0 or extra is None else extra[(r-1)*model.core+j]
            h=block(h,cos,sin)
        states.append(h)
    for i in range(model.prelude+model.core,model.config.layers):h=model.blocks[i](h,cos,sin)
    return model.norm(h),e,states


def state_statistics(value):
    x=value.detach().float().flatten(0,1)
    rms=x.square().mean().sqrt()
    centered=(x-x.mean(0,keepdim=True)).square().mean().sqrt()
    token_rms=x.square().mean(-1).sqrt()
    return {'rms':float(rms),'centered_token_rms':float(centered),
            'centered_over_total_rms':float(centered/rms.clamp_min(1e-30)),
            'token_rms_mean':float(token_rms.mean()),'token_rms_std':float(token_rms.std())}


def step_statistics(first,second):
    a,b=first.detach().float(),second.detach().float()
    return {'relative_rms_change':float((b-a).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-30)),
            'mean_token_cosine':float(F.cosine_similarity(a,b,dim=-1).mean())}
