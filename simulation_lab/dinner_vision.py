"""RGB-only localization in calibrated dinner workspace views.

Color/shape rules are explicit classical perception, not a learned detector.
They do not read simulator bodies, depth, segmentation IDs or object poses.
"""
import numpy as np
from .policy_observation import ObservationRejected


def components(mask):
    remaining = set(zip(*np.where(mask)))
    found = []
    while remaining:
        todo = [remaining.pop()]; pixels = []
        while todo:
            y, x = todo.pop(); pixels.append((y, x))
            for neighbor in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor); todo.append(neighbor)
        if len(pixels) >= 12: found.append(np.array(pixels))
    return found


def dinner_features(images, skill):
    index = 1 if skill in ('fork', 'spoon', 'drawer') else 0
    rgb = np.asarray(images[index], dtype=float)
    if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all():
        raise ObservationRejected('Dinner vision needs finite calibrated 320x240 RGB views.')
    r, g, b = rgb.transpose(2, 0, 1)
    roi = np.zeros((240, 320), dtype=bool)
    if skill == 'plate':
        roi[105:178, 105:185] = True
        mask = (b > r*1.25) & (g > r*1.2) & (b > g*.98) & (r > 100)
        candidates=[p for p in components(mask & roi) if 80 <= len(p) <= 1100]
        if len(candidates)!=1:raise ObservationRejected('Blue plate is missing, obscured or ambiguous in the overhead view.')
        points=candidates[0]
    elif skill == 'mug':
        roi[95:192, 190:280] = True
        mask = (g > r*1.5) & (g > b*1.015) & (b > r*1.3) & (g > 45)
        candidates = [p for p in components(mask & roi) if 18 <= len(p) <= 450]
        if len(candidates) != 1: raise ObservationRejected('Teal mug is missing or ambiguous in the overhead view.')
        points = candidates[0]
    elif skill == 'drawer':
        roi[76:121,115:190]=True
        mask=(r>100)&(b>r*.98)&(g>r*1.005)
        candidates=[p for p in components(mask & roi) if 80<=len(p)<=300 and p[:,0].mean()>97]
        if len(candidates)!=1:raise ObservationRejected('Closed drawer handle is missing or ambiguous in the workspace view.')
        points=candidates[0]
    elif skill in ('fork', 'spoon'):
        roi[65:138, 100:185] = True
        mask = (r > 100) & (b > r*.98) & (g > r*1.005)
        candidates = [p for p in components(mask & roi) if 70 <= len(p) <= 450]
        # The upper and lower utensil lanes are calibrated spatial supports.
        # Arbitrary exchanges of the two utensils are not claimed supported.
        candidates = [p for p in candidates if (p[:, 0].mean() < 104) == (skill == 'fork')]
        if len(candidates) != 1: raise ObservationRejected('Requested utensil is missing or ambiguous in its visible drawer lane.')
        points = candidates[0]
    else:
        raise ObservationRejected('No RGB geometry encoder for '+skill)
    ys, xs = points.T
    result = np.zeros(32, dtype=np.float32)
    result[:2] = [xs.mean()/320, ys.mean()/240]
    if skill in ('fork', 'spoon'):
        centered = np.column_stack((xs-xs.mean(), ys-ys.mean()))
        covariance = centered.T@centered/len(xs)
        angle = .5*np.arctan2(2*covariance[0, 1], covariance[0, 0]-covariance[1, 1])
        result[2:4] = [np.cos(2*angle), np.sin(2*angle)]
    return result


def scene_observation(images):
    from .bottle_vision import bottle_features
    result={'bottle_path_clear':None,'drawer_path_clear':None,'drawer_open':None,
            'source':'calibrated RGB color/geometry rules; no simulator object positions'}
    try:
        feature=bottle_features(images)
        result['bottle_path_clear']=bool(feature[0]*320>180)
        result['bottle_centroid_px']=(feature[:2]*[320,240]).tolist()
        x,y=result['bottle_centroid_px']
        result['bottle_region']='left_reach' if 130<=x<=147 and 166<=y<=185 else 'standard'
    except ObservationRejected:pass
    try:
        feature=dinner_features(images,'plate')
        result['drawer_path_clear']=bool(feature[0]*320>137)
        result['plate_centroid_px']=(feature[:2]*[320,240]).tolist()
    except ObservationRejected:pass
    rgb=np.asarray(images[1],dtype=float)
    if rgb.shape!=(240,320,3) or not np.isfinite(rgb).all():
        raise ObservationRejected('Scene checks require finite calibrated RGB views.')
    if rgb.mean()<10:return result
    r,g,b=rgb.transpose(2,0,1)
    mask=(g>r*1.15)&(g>b*.95)&(g>25)&(g<140)
    area=int(mask[80:150,100:210].sum())
    result['drawer_lining_pixels']=area
    if area>1500:result['drawer_open']=True
    elif area<750:result['drawer_open']=False
    return result
