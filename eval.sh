python eval.py \
  trained/ca_with_dog/ca_with_dog_izar_v100 \
  trained/ca_with_dog \
  trained/ca_with_dog/ca_with_dog_izar_v100/offline_eval \
  --size '416*240' \
  --cam_radius 1.8 \
  --ref_cam_radius 1.8 \
  --ref_azim 60 \
  --elev_l -10 \
  --elev_r 50