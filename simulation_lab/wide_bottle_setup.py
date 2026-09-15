"""Reset-only overlap validation; no learned action or observation generation."""
import mujoco


def validate_start(model, data):
    mujoco.mj_forward(model, data)
    body = model.body('bottle').id
    penetration = max((-float(c.dist) for c in data.contact
        if model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]
        and body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])), default=0.)
    if penetration > .001:
        raise ValueError('This seed starts the bottle overlapping another object. Choose another seed; no movement was started.')
    return {'initial_bottle_penetration_m': penetration, 'initial_bottle_overlap_valid': True}
