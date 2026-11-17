//! A Rust program for shem: the standard library, collections, formatting and an exit status.

use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut letters: BTreeMap<char, usize> = BTreeMap::new();
    for c in "hello from rust on shem".chars().filter(|c| c.is_alphabetic()) {
        *letters.entry(c).or_default() += 1;
    }
    let squares: u64 = (0u64..1000).map(|n| n * n).sum();
    println!("args = {args:?}");
    println!("sum of squares below 1000 = {squares}");
    println!("most common letter = {:?}", letters.iter().max_by_key(|&(c, n)| (n, std::cmp::Reverse(c))));
    std::process::exit(args.len() as i32);
}
