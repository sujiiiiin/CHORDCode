REFERENCE_LOSS_WIDTH = 832
REFERENCE_LOSS_HEIGHT = 480


def compute_resolution_loss_scales(wan_guide, image_width, image_height, frame_num):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive for loss scaling.")

    _, _, ref_lat_w, ref_lat_h, _ = wan_guide._compute_latent_shape(
        size=(REFERENCE_LOSS_WIDTH, REFERENCE_LOSS_HEIGHT),
        frame_num=frame_num,
    )
    cur_lat_w = getattr(wan_guide, "lat_w")
    cur_lat_h = getattr(wan_guide, "lat_h")
    if cur_lat_w <= 0 or cur_lat_h <= 0:
        raise ValueError("Wan latent width and height must be positive for loss scaling.")

    sds_scale = (ref_lat_w * ref_lat_h) / float(cur_lat_w * cur_lat_h)
    temporal_scale = (REFERENCE_LOSS_WIDTH * REFERENCE_LOSS_HEIGHT) / float(
        image_width * image_height
    )
    return sds_scale, temporal_scale


def set_resolution_loss_scales(opt, wan_guide, image_width, image_height):
    sds_scale, temporal_scale = compute_resolution_loss_scales(
        wan_guide,
        image_width,
        image_height,
        opt.frame_num,
    )
    opt.sds_resolution_scale = sds_scale
    opt.temporal_resolution_scale = temporal_scale
    return sds_scale, temporal_scale


def get_sds_resolution_scale(opt):
    return getattr(opt, "sds_resolution_scale", 1.0)


def get_temporal_resolution_scale(opt):
    return getattr(opt, "temporal_resolution_scale", 1.0)
