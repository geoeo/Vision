
#[derive(Debug,Clone)]
pub struct BitVector{
    data: Vec<u64>,
    entries: usize,
    bits: usize
}

impl BitVector {

    pub fn new(n: usize) -> BitVector{
        assert!(n == 256 || n == 512 || n == 128);
        let size = n/8;

        BitVector{data:  vec![0; size], entries: 0, bits: n }
    }

    pub fn bytes_per_element() -> u64 {
        8 // size of u64
    }

    pub fn bits_per_element() -> u64 {
        8*BitVector::bytes_per_element()
    }

    pub fn add_value(&mut self, new_val: u64) -> () {
        assert!(new_val == 0 || new_val == 1);
        assert!(self.entries <= self.bits-1);

        let last_bit_flag = (BitVector::bits_per_element()-1).pow(2);
        let mut last_bits = Vec::<u64>::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            let last_bit = self.data[i] & last_bit_flag;
            last_bits.push(last_bit);
        }

        for i in 0..self.data.len() {
            let v = self.data[i];
            self.data[i] = v << 1;
        }

        for i in 1..self.data.len() {
            let moved_bit = match last_bits[i-1] {
                0 => 0,
                _ => 1
            };

            self.data[i] |= moved_bit;
        }

        self.data[0] |= new_val;
        self.entries = self.entries+1;
    }

    //TODO: performance offender
    pub fn hamming_distance(&self, other: &BitVector) -> u64 {

        let mut xor_vec = Vec::<u64>::with_capacity(self.data.len());
        let mut distance = 0;

        for i in 0..self.data.len() {
            xor_vec.push(self.data[i] ^ other.data[i]);
        }

        for i in 0..self.data.len() {
            let mut xor_v = xor_vec[i];
            for _ in 0..BitVector::bits_per_element() {
                let bit_v = xor_v & 1;
                if bit_v == 1 {
                    distance = distance + 1;
                }
                xor_v = xor_v >> 1;
            }
        }

        distance
    }

}