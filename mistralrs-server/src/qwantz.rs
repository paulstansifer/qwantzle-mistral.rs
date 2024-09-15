use csv::Reader;
use mistralrs_core::Pipeline;
use std::{fs::File, sync::Arc};

struct Strip {
    leadup: String,
    punchline: String,
}

fn get_strips(path: String) -> Vec<Strip> {
    let file = File::open(path).unwrap();
    let mut reader = Reader::from_reader(file);

    let mut res = vec![];
    for result in reader.records() {
        let record = result.unwrap();

        let (prefix_lines, last_line) = record.get(0).unwrap().rsplit_once("[LINE] ").unwrap();
        // Skip one-word punchlines:
        if let Some((first_punchword, mystery)) = last_line.split_once(" ") {
            res.push(Strip {
                leadup: prefix_lines.to_owned() + "[LINE] " + first_punchword,
                punchline: " ".to_owned() + mystery,
            });
        }
    }
    return res;
}

pub fn qwantz(_pipeline: Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>, path: String) -> () {
    let strips = get_strips(path);

    for strip in strips.iter().take(3) {
        println!("{} ==>> {}", strip.leadup, strip.punchline);
    }
}

//cargo run -- --qwantz data/strips.csv plain --model-id Maykeye/TinyLLama-v0
