use csv::Reader;
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{pipeline::ForwardInputsResult, Pipeline};
use std::{fs::File, os::unix::process, sync::Arc};

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

fn step(pipeline: &mut dyn Pipeline, text: String) {
    // Create several dummy objects for the sequences. No custom logits processors.
    let (dummy_sender, _) = tokio::sync::mpsc::channel(10000);
    let dummy_sampler = mistralrs_core::sampler::Sampler::new(
        None,
        0,
        pipeline.tokenizer().clone(),
        None,
        None,
        None,
        -1,
        0.0,
        0.0,
        vec![],
    )
    .expect("sampler");

    let dummy_group = Arc::new(tokio::sync::Mutex::new(
        mistralrs_core::sequence::SequenceGroup::new(1, false, false, 0),
    ));

    let mut seqs = vec![];

    // let tokens = processor
    //     .process(
    //         &*pipeline,
    //         vec![IndexMap::from([
    //             ("role".to_string(), Either::Left("user".to_string())),
    //             ("content".to_string(), Either::Left(text)),
    //         ])],
    //         true,
    //         Vec::new(),
    //     )
    //     .expect("### tokens");

    // Here we are specifically not applying a chat template, because this is a completion task

    let tokens = pipeline
        .tokenizer()
        .encode(text, /*add_special_tokens*/ true)
        .expect("### tok")
        .get_ids()
        .to_vec();

    seqs.push(mistralrs_core::pipeline::amoe::new_dummy_seq(
        tokens,
        dummy_sender.clone(),
        dummy_sampler.clone(),
        dummy_group.clone(),
        None,
        (*pipeline.get_metadata().tok_trie).clone(),
    ));

    let mut input_seqs = seqs.iter_mut().collect::<Vec<_>>();

    let inputs_iter = pipeline.get_processor().inputs_processor().process_inputs(
        pipeline.tokenizer(),
        &mut input_seqs,
        /*is_prompt*/ true,
        pipeline.get_metadata().is_xlora,
        &pipeline.device(),
        pipeline.get_metadata().has_no_kv_cache,
        None,
        pipeline.get_input_processor_config(),
        None,
        pipeline.get_metadata().prompt_batchsize,
    );

    let mut logits = vec![None; seqs.len()];

    for (i, inputs) in inputs_iter.enumerate() {
        let mistralrs_core::pipeline::inputs_processor::InputProcessorOutput {
            inputs,
            seq_indices,
        } = inputs
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
            .expect("### Input error!");

        let raw_logits = pipeline
            .forward_inputs(inputs)
            .expect("### Forward failed!");

        for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
            logits[seq_idx] = Some(raw_logits.index_bs(logit_idx).expect("### Logits problem!"));
        }

        println!("Logits! {} ", logits.len());

        let crate::qwantz::ForwardInputsResult::CausalGeneration { logits: l } =
            logits[0].clone().unwrap();

        print!("result: ");

        // Based on code in sampler.rs:
        let mut probs: Vec<Vec<half::f16>> = l.to_vec2().expect("### l->v");
        let mut argsort_indices = (0..probs[0].len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_unstable_by(|&i, &j| {
            probs[0][j].partial_cmp(&probs[0][i]).expect("No ordering.")
        });

        for index in argsort_indices.iter().take(10) {
            let tok: usize = argsort_indices[*index];
            let tok_u32: u32 = tok as u32;
            print!(
                "{:?} => {}  ",
                probs[0][*index],
                pipeline
                    .tokenizer()
                    .decode(&[tok_u32], false)
                    .expect("###t")
            )
        }
        println!();
        println!();
    }
}

pub fn qwantz(pipeline: Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>, path: String) -> () {
    let strips = get_strips(path);

    step(&mut *pipeline.try_lock().unwrap(), "".to_owned());

    step(&mut *pipeline.try_lock().unwrap(), "I".to_owned());

    step(&mut *pipeline.try_lock().unwrap(), "I had".to_owned());

    step(&mut *pipeline.try_lock().unwrap(), "I had a".to_owned());

    step(
        &mut *pipeline.try_lock().unwrap(),
        "I had a little".to_owned(),
    );

    for strip in strips.iter().take(3) {
        println!("{} ==>> {}", strip.leadup, strip.punchline);
        step(&mut *pipeline.try_lock().unwrap(), strip.leadup.clone());
    }
}

//cargo run -- --qwantz data/strips.csv plain --model-id Maykeye/TinyLLama-v0

/* metrics to think about

top-p (sum probabilites (sorted) until passing the cutoff)
min-p (take all probabilites above the cutoff)

*/
