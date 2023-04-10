extern crate nalgebra as na;

use serde::{Serialize, Deserialize};
use std::cmp::{Ordering, Eq,PartialOrd, PartialEq, Ord};
use na::Vector2;


#[derive(Debug,Clone,Copy, Serialize, Deserialize)]
pub struct Point<T> where T: PartialOrd + PartialEq + Clone {
    pub x: T,
    pub y: T
}

impl<T> PartialEq for Point<T> where T: PartialOrd + Clone{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<T> Eq for Point<T> where T: PartialEq + PartialOrd + Clone {

}

impl<T> Ord for Point<T> where T: PartialOrd + PartialEq + Clone {
    fn cmp(&self, other: &Self) -> Ordering {
        let x_cmp = self.x.partial_cmp(&other.x);
        let y_cmp = self.y.partial_cmp(&other.y);
        
        match (x_cmp,y_cmp) {
            (Some(Ordering::Less),_) => Ordering::Less,
            (Some(Ordering::Greater),_) => Ordering::Greater,
            (Some(Ordering::Equal),Some(Ordering::Less)) => Ordering::Less,
            (Some(Ordering::Equal),Some(Ordering::Greater)) => Ordering::Greater,
            (Some(Ordering::Equal),Some(Ordering::Equal)) => Ordering::Equal,
            (None,Some(_)) => Ordering::Less,
            (Some(_),_) => Ordering::Greater,
            (None,None) => Ordering::Equal
        }
    }
}

impl<T> PartialOrd for Point<T>  where T: PartialOrd + PartialEq + Clone {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Point<T> where T: PartialOrd + PartialEq + Clone {
    pub fn new(x: T, y:T) -> Point<T> {
        Point{x,y}
    }

    pub fn to_vector(&self) -> Vector2<T> {
        Vector2::<T>::new(self.x.clone(),self.y.clone())
    }
}
