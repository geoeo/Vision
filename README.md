# Papers Implemented
Fast http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.3991&rep=rep1&type=pdf

Orb http://www.gwylab.com/download/ORB_2012.pdf

Dense VO https://ieeexplore.ieee.org/document/6631104

Imu Preintgration http://rpg.ifi.uzh.ch/docs/TRO16_forster.pdf

Sparse Bundle Adjustment (WIP) http://users.ics.forth.gr/~lourakis/sba/sba-toms.pdf

## Orb
![billa_cereal_far](doc/cereal_far.png)![billa_cereal_tilted](doc/cereal_tilted.png)

## Fast Corner
![fast](doc/lenna_fast.png)

## Dense VO (LM + SoftOneLoss) - Tum Dataset Freiburg2 Desk

![dense](doc/freiburg2_desk_0_max_its_800_w_true_l_SoftOneLoss_+1e-16_lm_max_norm_eps_+1e-10_delta_eps_+1e-10_t_+1e-6_t_+1e-3_t_+1e-3_t_+1e0_s_0.01_o_4_b_true_br_1_neg_d_false.png)

## Visual-Interial Odometry - Realsense D455

### With Bias Estimation
![vi_bias](doc/d455_all_simple_trans_imu_max_its_100_w_true_HuberLossForPos_+1e-16_l_SoftOneLoss_+1e-16_lm_max_norm_eps_+1e-10_delta_eps_+1e-10_t_+1e-6_t_+1e-3_t_+1e-3_convert_to_cam_coords_true_bias_30e4.png)

### Without Bias Estimation
![vi](doc/d455_all_simple_trans_imu_max_its_800_w_true_HuberLossForPos_+1e-16_l_SoftOneLoss_+1e-16_lm_max_norm_eps_+1e-10_delta_eps_+1e-10_t_+1e-6_t_+1e-3_t_+1e-3_convert_to_cam_coords_true.png)

