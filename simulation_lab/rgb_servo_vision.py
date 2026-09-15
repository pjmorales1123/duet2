"""Experimental RGB-only bottle observations; no simulator-state inputs."""
import numpy as np
from .dinner_vision import components
from .policy_observation import ObservationRejected


def amber_candidates(image):
    rgb=np.asarray(image)
    if rgb.shape!=(240,320,3) or not np.isfinite(rgb).all():
        raise ObservationRejected('Expected finite 320x240 RGB camera input.')
    r,g,b=rgb.astype(np.float32).transpose(2,0,1)
    mask=(r>g*1.28)&(g>b*1.18)&(r>45)
    # Conservative appearance candidates only. No body IDs or segmentation.
    rows=[]
    for points in components(mask):
        ys,xs=points.T; width=int(xs.max()-xs.min()+1);height=int(ys.max()-ys.min()+1)
        if 20<=len(points)<=1400 and 4<=width<=70 and 4<=height<=70 and len(points)/(width*height)>.2:
            rows.append({'centroid_px':[float(xs.mean()),float(ys.mean())],'pixels':len(points),
                         'bbox_px':[int(xs.min()),int(ys.min()),width,height]})
    return rows


def observe_bottle_views(images):
    views={}
    for camera,image in images.items():
        candidates=amber_candidates(image)
        views[camera]={'status':'unique' if len(candidates)==1 else 'missing' if not candidates else 'ambiguous',
                       'candidates':candidates}
    return {'source':'RGB appearance only; experimental, not a 3D pose estimate', 'views':views,
            'unique_views':sum(v['status']=='unique' for v in views.values())}
