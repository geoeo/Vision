use std::cmp::Ordering;

#[derive(Copy,Debug,Clone,Eq)]
pub struct Offset {
    pub x: isize,
    pub y: isize
}

impl PartialEq for Offset {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y== other.y
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> Ordering {
        let x_cmp = self.x.cmp(&other.x);
        let y_cmp = self.y.cmp(&other.y);
        
        match (x_cmp,y_cmp) {
            (Ordering::Less,_) => Ordering::Less,
            (Ordering::Greater,_) => Ordering::Greater,
            (Ordering::Equal,Ordering::Less) => Ordering::Less,
            (Ordering::Equal,Ordering::Greater) => Ordering::Greater,
            (Ordering::Equal,Ordering::Equal) => Ordering::Equal
        }
    }
}

impl PartialOrd for Offset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


