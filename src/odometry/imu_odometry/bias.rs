extern crate nalgebra as na;

use na::{Vector,Vector3,Vector6,Matrix3,Matrix,SMatrix,U3,storage::Storage, Const};

use crate::odometry::imu_odometry::{imu_delta::ImuDelta, ImuResidual};
use crate::numerics::lie::{right_jacobian,skew_symmetric, right_inverse_jacobian, exp_r};
use crate::Float;

pub struct BiasDelta {
    pub bias_a_delta: Vector3<Float>,
    pub bias_g_delta: Vector3<Float>
}

impl BiasDelta {

    pub fn empty() -> BiasDelta {
        BiasDelta {
            bias_a_delta: Vector3::<Float>::zeros(),
            bias_g_delta: Vector3::<Float>::zeros()
        }

    }

    pub fn add_pertb<R>(&self, new_pertb: &Vector<Float,Const<6>,R>) -> BiasDelta where R: Storage<Float,Const<6>,Const<1>> {
        BiasDelta {
            bias_a_delta: self.bias_a_delta + new_pertb.fixed_rows::<3>(0),
            bias_g_delta: self.bias_g_delta + new_pertb.fixed_rows::<3>(3)
        }

    }

    pub fn add_delta(&self, new_delta: &BiasDelta) -> BiasDelta {
        BiasDelta {
            bias_a_delta: self.bias_a_delta + new_delta.bias_a_delta,
            bias_g_delta: self.bias_g_delta + new_delta.bias_g_delta
        }

    }

    pub fn norm(&self) -> Float {
        self.bias_a_delta.norm() + self.bias_g_delta.norm()
    }

}


pub struct BiasPreintegrated {
    pub rotation_jacobian_bias_g: Matrix3<Float>,
    pub velocity_jacobian_bias_a: Matrix3<Float>,
    pub velocity_jacobian_bias_g: Matrix3<Float>,
    pub position_jacobian_bias_a: Matrix3<Float>,
    pub position_jacobian_bias_g: Matrix3<Float>,
    pub integrated_bias_a: Vector3<Float>,
    pub integrated_bias_g: Vector3<Float>,
    pub bias_a_std: Vector3<Float>,
    pub bias_g_std: Vector3<Float>,
}

impl BiasPreintegrated {

    //TODO:rename bias spectral -> discrete
    pub fn new(bias_accelerometer: &Vector3<Float>,bias_gyro: &Vector3<Float>, bias_spectral_noise_density_acc: &Vector3<Float>, bias_spectral_noise_density_gyro: &Vector3<Float>, acceleration_data: &[Vector3<Float>],gyro_delta_times: &Vec<Float>, 
        delta_lie_i_k: &Vec<Vector3<Float>>, delta_rotations_i_k: &Vec<Matrix3::<Float>>) -> BiasPreintegrated {

        let acc_rotations_i_k = delta_rotations_i_k.iter().scan(Matrix3::identity(), |acc, dr| {
            *acc = *acc*dr;
            Some(*acc)
        } ).collect::<Vec<Matrix3<Float>>>();

        let acc_delta_times_i_k = gyro_delta_times.iter().scan(0.0, |acc,dt| {
            *acc = *acc+dt;
            Some(*acc)
        }).collect::<Vec<Float>>();

        let acc_rotations_i_k_delta_times = acc_rotations_i_k.iter().zip(acc_delta_times_i_k.iter()).map(|(&dr,&dt)| dr*dt).collect::<Vec<Matrix3<Float>>>();
        let acceleration_skew_symmetric_matrices = acceleration_data.iter().map(|x| skew_symmetric(&(x - bias_accelerometer))).collect::<Vec<Matrix3<Float>>>();

        
        let acc_delta_times_k_plus_1_j_rev = gyro_delta_times[1..].iter().rev().scan(0.0, |acc,dt| {
            *acc = *acc+dt;
            Some(*acc)
        }).collect::<Vec<Float>>();
        let acc_rotation_k_plus_1_j_rev = delta_rotations_i_k[1..].iter().rev().scan(Matrix3::identity(), |acc, dr| {
            *acc = dr*(*acc);
            Some(*acc)
        }).collect::<Vec<Matrix3<Float>>>();
        let right_jacobians_k_plus_1_j_rev = delta_lie_i_k[1..].iter().rev().map(|x| right_jacobian(x)).collect::<Vec<Matrix3<Float>>>();

        let rotation_jacobians = acc_rotation_k_plus_1_j_rev.iter()
            .zip(acc_delta_times_k_plus_1_j_rev.iter())
            .zip(right_jacobians_k_plus_1_j_rev.iter())
            .scan(Matrix3::<Float>::zeros(), |acc,((&dr,&dt),&j)| {
                *acc = *acc + dr.transpose()*j*dt;
                Some(*acc)
            })
            .map(|x| -x)
            .collect::<Vec<Matrix3<Float>>>();

        let rotation_jacobian_i_j = rotation_jacobians.last().unwrap_or_else(|| panic!("preintegration rotation jacobian empty!"));

        let velocity_jacobians_for_bias_a = acc_rotations_i_k_delta_times.iter()
        .scan(Matrix3::<Float>::zeros(),|acc,delta_r_dt| {
            *acc = *acc + delta_r_dt;
            Some(*acc)
        })
        .map(|x| -x)
        .collect::<Vec<Matrix3<Float>>>();
        let velocity_jacobian_for_bias_a = velocity_jacobians_for_bias_a.last().unwrap_or_else(|| panic!("preintegration velocity bias a jacobian empty!"));


        let velocity_jacobians_for_bias_g = acc_rotations_i_k_delta_times.iter()
        .zip(acceleration_skew_symmetric_matrices.iter())
        .zip(rotation_jacobians.iter())
        .scan(Matrix3::<Float>::zeros(),|acc,((delta_r_dt,acceleration_skew),j)| {
            *acc = *acc + delta_r_dt*acceleration_skew*j;
            Some(*acc)
        })
        .map(|x| -x)
        .collect::<Vec<Matrix3<Float>>>();

        let velocity_jacobian_for_bias_g = velocity_jacobians_for_bias_g.last().unwrap_or_else(|| panic!("preintegration velocity bias g jacobian empty!"));

        let pos_jacobian_for_bias_a = velocity_jacobians_for_bias_a.iter()
        .zip(acc_delta_times_i_k.iter())
        .zip(acc_rotations_i_k_delta_times.iter())
        .fold(Matrix3::<Float>::zeros(),|acc, ((dv,&dt),delta_r_dt)| {
            acc + dv*dt -0.5*delta_r_dt*dt
        });

        let pos_jacobian_for_bias_g = velocity_jacobians_for_bias_g.iter()
        .zip(acc_delta_times_i_k.iter())
        .zip(acc_rotations_i_k_delta_times.iter())
        .zip(acceleration_skew_symmetric_matrices.iter())
        .zip(rotation_jacobians.iter())
        .fold(Matrix3::<Float>::zeros(),|acc,((((dv,&dt),delta_r_dt),acceleration_ss),dr)| {
            acc + dv*dt-0.5*delta_r_dt*acceleration_ss*dr*dt
        });

        let dt = *acc_delta_times_i_k.last().unwrap();
        let dt_sqrt = dt.sqrt();
        let bias_a_std = bias_spectral_noise_density_acc*(dt_sqrt);
        let bias_g_std = bias_spectral_noise_density_gyro*(dt_sqrt);

        //TODO: check if dt or sqrt(dt) is better
        // let integrated_bias_a = bias_accelerometer + bias_a_std;
        // let integrated_bias_g = bias_gyro + bias_g_std;

        let integrated_bias_a = bias_accelerometer + bias_spectral_noise_density_acc*dt;
        let integrated_bias_g = bias_gyro + bias_spectral_noise_density_gyro*dt;

        BiasPreintegrated {
            rotation_jacobian_bias_g: *rotation_jacobian_i_j,
            velocity_jacobian_bias_a: *velocity_jacobian_for_bias_a,
            velocity_jacobian_bias_g: *velocity_jacobian_for_bias_g,
            position_jacobian_bias_a: pos_jacobian_for_bias_a,
            position_jacobian_bias_g: pos_jacobian_for_bias_g,
            integrated_bias_a,
            integrated_bias_g,
            bias_a_std,
            bias_g_std
        }
    }
}

pub fn compute_residual(bias_est: &Vector3<Float>, bias_preintegrated: &Vector3<Float>) -> Vector3<Float> {
    bias_preintegrated - bias_est
}

pub fn compute_cost_for_weighted(residual_bias_a_weighted: &Vector3<Float>, residual_bias_g_weighted: &Vector3<Float>) -> Float {
    (residual_bias_a_weighted.transpose()*residual_bias_a_weighted + residual_bias_g_weighted.transpose()*residual_bias_g_weighted)[0]

}

pub fn weight_residual(res_target: &mut Vector3<Float>, weights: &Vector3<Float>) -> () {
    res_target.component_mul_assign(weights);
}

pub fn genrate_residual_jacobian(bias_delta: &BiasDelta, preintegrated_bias: &BiasPreintegrated, residuals: &ImuResidual) -> SMatrix<Float,9,6> {
    let mut jacobian = SMatrix::<Float,9,6>::zeros();

    let residual_position_jacobian_bias_a = -preintegrated_bias.position_jacobian_bias_a;
    let residual_position_jacobian_bias_g = -preintegrated_bias.position_jacobian_bias_g;
    let residual_velocity_jacobian_bias_a = -preintegrated_bias.velocity_jacobian_bias_a;
    let residual_velocity_jacobian_bias_g = -preintegrated_bias.velocity_jacobian_bias_g;


    let j_r_b = right_jacobian(&(preintegrated_bias.rotation_jacobian_bias_g*bias_delta.bias_g_delta));
    let residual_rotation = residuals.fixed_rows::<3>(3);
    let residual_rotation_jacbobian_bias_g = -right_inverse_jacobian(&residual_rotation)*exp_r(&residual_rotation).transpose()*j_r_b*preintegrated_bias.rotation_jacobian_bias_g;

    jacobian.fixed_slice_mut::<3,3>(0,0).copy_from(&residual_position_jacobian_bias_a);
    jacobian.fixed_slice_mut::<3,3>(0,3).copy_from(&residual_position_jacobian_bias_g);
    jacobian.fixed_slice_mut::<3,3>(3,3).copy_from(&residual_rotation_jacbobian_bias_g);
    jacobian.fixed_slice_mut::<3,3>(6,0).copy_from(&residual_velocity_jacobian_bias_a);
    jacobian.fixed_slice_mut::<3,3>(6,3).copy_from(&residual_velocity_jacobian_bias_g);

    jacobian

}