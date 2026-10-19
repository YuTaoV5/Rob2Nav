"""Read-only reconstruction of terrain attention for one observed state."""
import torch


def describe(actor,observation):
    history=observation[:,:225];current=history[:,-45:]
    proprio=torch.cat((actor.base.student_moe_encoder(history),current),-1)
    query_input=proprio+actor.goal_projection(observation[:,2304:2308]) if actor.goal_conditioned else proprio
    module=actor.terrain;maps=observation[:,225:2304].reshape(-1,3,21,33)
    masked=torch.stack((maps[:,0]*maps[:,1],maps[:,1],maps[:,2]*maps[:,1]),1)
    features=module.cnn(masked).flatten(2).transpose(1,2)
    tokens=torch.cat((features,module.xy.unsqueeze(0).expand(features.shape[0],-1,-1)),-1)
    tokens=torch.cat((tokens,module.null.expand(features.shape[0],-1,-1)),1)
    keys=module.key(tokens).reshape(-1,694,4,16).transpose(1,2)
    query=module.query(query_input).reshape(-1,4,1,16)
    valid=maps[:,1].flatten(1)>.5;known=torch.cat((valid,torch.ones_like(valid[:,:1])),1)
    logits=(query*keys).sum(-1)/4.
    if module.spatial_prior:logits=logits+module.spatial_bias[None]
    weights=logits.masked_fill(~known[:,None,:],-1e4).softmax(-1)
    values=module.value(tokens).reshape(-1,694,4,16).transpose(1,2)
    reconstructed=(weights.unsqueeze(-1)*values).sum(2).reshape(-1,64)
    torch.testing.assert_close(reconstructed,module(observation[:,225:2304],query_input),rtol=1e-5,atol=1e-6)
    real=weights[:,:,:693];mass=real.sum(-1).clamp(min=1e-9)
    xy=(real.unsqueeze(-1)*module.xy).sum(-2)/mass.unsqueeze(-1)
    near=(module.xy[:,0]<.5)&(module.xy[:,0]>-.35)&(module.xy[:,1].abs()<.3)
    supported=(maps[:,0].flatten(1)<.5)&valid
    return {'head_xy_centroid':xy[0].cpu().tolist(),
        'head_null_mass':weights[0,:,-1].cpu().tolist(),
        'head_near_mass':real[:,:,near].sum(-1)[0].cpu().tolist(),
        'head_support_mass':(real*supported[:,None,:]).sum(-1)[0].cpu().tolist(),
        'head_entropy':(-(weights*weights.clamp(min=1e-12).log()).sum(-1))[0].cpu().tolist(),
        'note':'Generic heads are not assigned to legs; support threshold is diagnostic, not an exact foothold label.'}
