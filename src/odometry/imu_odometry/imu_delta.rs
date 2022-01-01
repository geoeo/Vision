extern crate nalgebra as na;

use na::{Vector3,Matrix3,Rotation3, Isometry3,Const, Vector, storage::Storage};
use crate::odometry::imu_odometry::{ImuPertrubation, bias::{BiasPreintegrated, BiasDelta}};
use crate::numerics::lie::{exp_so3,ln_SO3, vector_from_skew_symmetric};
use crate::Float;


//TODO: make make preintegrated term and delta more explicit/separate
pub struct ImuDelta {
    pub delta_position: Vector3<Float>,
    pub delta_velocity: Vector3<Float>,
    pub delta_rotation_i_k: Matrix3<Float>,
    pub delta_rotation_k: Matrix3<Float>
}

impl ImuDelta {

    pub fn empty() -> ImuDelta {
        ImuDelta {
            delta_position: Vector3::<Float>::zeros(),
            delta_velocity: Vector3::<Float>::zeros(),
            delta_rotation_i_k: Matrix3::<Float>::identity(),
            delta_rotation_k: Matrix3::<Float>::identity()
        }
    }

    pub fn add_pertb<R>(&self, new_pertb: &Vector<Float,Const<9>,R>) -> ImuDelta where R: Storage<Float,Const<9>,Const<1>> {
        //TODO: check this, we are interpreting delta trans as a differential quantity
        //let delta_pose = exp(&new_pertb.fixed_rows::<3>(0),&new_pertb.fixed_rows::<3>(3));
        ImuDelta {
            delta_position: self.delta_position + new_pertb.fixed_rows::<3>(0),
            //delta_position: self.delta_position + delta_pose.fixed_slice::<3,1>(0,3),
            delta_velocity: self.delta_velocity + new_pertb.fixed_rows::<3>(6),
            delta_rotation_i_k: self.delta_rotation(),
            delta_rotation_k: exp_so3(&new_pertb.fixed_rows::<3>(3))
            //delta_rotation_k: delta_pose.fixed_slice::<3,3>(0,0).clone_owned()
        }
    }

    pub fn delta_rotation(&self) -> Matrix3<Float> {
        self.delta_rotation_i_k*self.delta_rotation_k
    }

    pub fn rotation_lie(&self) -> Vector3<Float> {
        vector_from_skew_symmetric(&ln_SO3(&self.delta_rotation()))
    }

    pub fn get_pose(&self) -> Isometry3<Float> {
        let rotation = Rotation3::<Float>::from_matrix(&self.delta_rotation());
        Isometry3::<Float>::new(self.delta_position,rotation.scaled_axis())
    }

    pub fn state_vector(&self) -> ImuPertrubation {
        let lie_r = self.rotation_lie();
        let mut state_vector = ImuPertrubation::zeros();
        state_vector.fixed_rows_mut::<3>(0).copy_from(&self.delta_position);
        state_vector.fixed_rows_mut::<3>(3).copy_from(&lie_r);
        state_vector.fixed_rows_mut::<3>(6).copy_from(&self.delta_velocity);
        state_vector

    }

    pub fn norm(&self) -> Float {
        let lie_r = self.rotation_lie();
        let mut state_vector = ImuPertrubation::zeros();
        state_vector.fixed_rows_mut::<3>(0).copy_from(&self.delta_position);
        state_vector.fixed_rows_mut::<3>(3).copy_from(&lie_r);
        state_vector.fixed_rows_mut::<3>(6).copy_from(&self.delta_velocity);
        state_vector.norm()
    }

    pub fn update_with_bias(&mut self, bias_preintegrated: &BiasPreintegrated, bias_delta: &BiasDelta) -> () {
        self.delta_rotation_k = self.delta_rotation_k*exp_so3(&(bias_preintegrated.rotation_jacobian_bias_g*bias_delta.bias_g_delta));
        self.delta_velocity = self.delta_velocity + bias_preintegrated.velocity_jacobian_bias_a*bias_delta.bias_a_delta 
        + bias_preintegrated.velocity_jacobian_bias_g*bias_delta.bias_g_delta;
        self.delta_position = self.delta_position + bias_preintegrated.position_jacobian_bias_a*bias_delta.bias_a_delta 
        + bias_preintegrated.position_jacobian_bias_g*bias_delta.bias_g_delta;
    }
}