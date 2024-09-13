use csv::Reader;
use std::fs::File;

struct Strip {}

pub fn get_strips() -> Vec<Strip> {
    let file = File::open("/home/paul/qwantz-strips.csv").unwrap();
    let mut reader = Reader::from_reader(file);

    for result in reader.records().take(5) {
        let record = result.unwrap();

        for field in &record {
            for line in field.split("[SPEAKER]") {
                print!("{}\n", line);
            }
        }
        println!("---------");
    }
}

pub fn qwantz() -> () {}
