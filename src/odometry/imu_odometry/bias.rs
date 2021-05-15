extern crate nalgebra as na;

use na::{Vector3,Vector6,Matrix3, SMatrix};

use crate::numerics::lie::{right_jacobian,skew_symmetric};
use crate::Float;

pub struct BiasDelta {
    pub bias_a_delta: Vector3<Float>,
    pub bias_g_delta: Vector3<Float>
}


pub struct BiasPreintegrated {
    pub rotation_jacobian_bias_g: Matrix3<Float>,
    pub velocity_jacobian_bias_a: Matrix3<Float>,
    pub velocity_jacobian_bias_g: Matrix3<Float>,
    pub position_jacobian_bias_a: Matrix3<Float>,
    pub position_jacobian_bias_g: Matrix3<Float>
}

impl BiasPreintegrated {

    //TODO:test
    pub fn new(bias_accelerometer: &Vector3<Float>,acceleration_data: &[Vector3<Float>],gyro_delta_times: &Vec<Float>, 
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
        //let acc_rotations_i_k_delta_times_sqrd = acc_rotations_i_k_delta_times.iter().zip(acc_delta_times_i_k.iter()).map(|(&dr,&dt)| dr*dt).collect::<Vec<Matrix3<Float>>>();
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

        BiasPreintegrated {
            rotation_jacobian_bias_g: *rotation_jacobian_i_j,
            velocity_jacobian_bias_a: *velocity_jacobian_for_bias_a,
            velocity_jacobian_bias_g: *velocity_jacobian_for_bias_g,
            position_jacobian_bias_a: pos_jacobian_for_bias_a,
            position_jacobian_bias_g: pos_jacobian_for_bias_g

        }
    }
}

pub fn generate_residuals() -> Vector6<Float> {

    panic!("TODO")

}

pub fn generate_jacobian<const R: usize>() -> SMatrix<Float, R, 6> {

    panic!("TODO")

}