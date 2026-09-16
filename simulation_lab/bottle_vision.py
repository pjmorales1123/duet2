"""Explicit color-based bottle localization for the fixed overhead camera.

This small experiment uses RGB only. The calibrated region excludes the drawer
and arms. It is not a general detector and rejects missing/ambiguous amber blobs.
"""
import numpy as np
from .policy_observation import ObservationRejected

def bottle_features(images):
    rgb=np.asarray(images[0],dtype=float)
    if rgb.shape!=(240,320,3):raise ObservationRejected('Bottle vision requires the calibrated 320x240 overhead camera.')
    r,g,b=rgb.transpose(2,0,1)
    mask=(r>g*1.28)&(g>b*1.18)&(r>45)
    roi=np.zeros((240,320),bool);roi[125:210,95:210]=True;mask&=roi
    candidates=[];remaining=set(zip(*np.where(mask)))
    while remaining:
        todo=[remaining.pop()];pixels=[]
        while todo:
            y,x=todo.pop();pixels.append((y,x))
            for next_pixel in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                if next_pixel in remaining:remaining.remove(next_pixel);todo.append(next_pixel)
        ys,xs=np.array(pixels).T
        if xs.min()<=95 or xs.max()>=209 or ys.min()<=125 or ys.max()>=209:continue
        if 40<=len(xs)<=260:
            candidates.append((xs,ys))
    if len(candidates)!=1:raise ObservationRejected('Amber bottle is missing or ambiguous in the calibrated camera region: '+str([(len(x),round(x.mean(),1),round(y.mean(),1)) for x,y in candidates]))
    xs,ys=candidates[0]
    if xs.max()-xs.min()>20 or ys.max()-ys.min()>20:
        raise ObservationRejected('This visual controller supports upright bottles only.')
    features=np.zeros(32,dtype=np.float32)
    features[:2]=[xs.mean()/320,ys.mean()/240]
    return features
