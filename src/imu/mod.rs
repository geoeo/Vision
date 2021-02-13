
use crate::Float;

#[derive(Debug,Copy,Clone)]
pub struct Imu {
    pub linear_acc: [Float;3],

    pub rotational_vel: [Float;3]
}

impl Imu {

    //TODO: scaling if value in Gs
    pub fn new(linear_acc: &Vec<Float>, rot_vel: &Vec<Float>) -> Imu {
        assert_eq!(linear_acc.len(),3);
        assert_eq!(rot_vel.len(),3);

        Imu{linear_acc: [linear_acc[0],linear_acc[1],linear_acc[2]], rotational_vel: [rot_vel[0],rot_vel[1],rot_vel[2]]}
    }

}