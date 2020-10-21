use crate::features::geometry::{circle::Circle,offset::Offset};

#[derive(Debug,Clone)]
struct FastDescriptor {
    pub x_center: usize,
    pub y_center: usize,
    pub starting_offsets: [Offset;4],
    pub continuous_offsets: Vec<Offset>

}

//TODO: check this
impl FastDescriptor {
    pub fn new(circle: &Circle) -> FastDescriptor {
        let starting_offsets = [circle.offsets[0],circle.offsets[1],circle.offsets[2],circle.offsets[3]];
        let mut positive_y_offset = Vec::<Offset>::with_capacity(circle.offsets.len()/2);
        let mut negative_y_offset = Vec::<Offset>::with_capacity(circle.offsets.len()/2);
        
        for offset in &circle.offsets {
            match offset {
                val if val.y > 0 || val.y == 0 && val.x > 0 => positive_y_offset.push(*val),
                val => negative_y_offset.push(*val)
            };

            positive_y_offset.sort_unstable_by(|a,b| b.cmp(a));
            negative_y_offset.sort_unstable_by(|a,b| a.cmp(b));

        }

    FastDescriptor {x_center:circle.x_center, y_center: circle.y_center,starting_offsets,continuous_offsets: [positive_y_offset,negative_y_offset].concat()}
    }
}