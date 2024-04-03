use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
    sync::mpsc::Sender,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::Tensor;
use candle_sampling::logits_processor::{LogitsProcessor, Logprobs};

use crate::{
    deref_mut_refcell, deref_refcell,
    response::{ChatCompletionChunkResponse, Choice, ChunkChoice, Response, SYSTEM_FINGERPRINT},
    ChatCompletionResponse, ChatCompletionUsage,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
    ModelLength(usize),
}

impl ToString for StopReason {
    fn to_string(&self) -> String {
        match self {
            StopReason::Eos => "stop".to_string(),
            StopReason::Length(_) | StopReason::ModelLength(_) => "length".to_string(),
            StopReason::StopTok(_) => "stop".to_string(),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum SequenceState {
    Done(StopReason),
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
}

pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    logits_processor: LogitsProcessor,
    stop_tokens: Vec<u32>,
    return_logprobs: bool,
    responder: Sender<Response>,
    response_index: usize,

    // Cache
    scaling_cache: Option<Tensor>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    xlora_cache: Option<Vec<Option<(Tensor, Tensor)>>>,

    // Mutables
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    group: Rc<RefCell<SequenceGroup>>,
    pub total_sampling_time: u128,
    state: Cell<SequenceState>,
}

impl Sequence {
    #[allow(clippy::too_many_arguments)]
    pub fn new_waiting(
        tokens: Vec<u32>,
        id: usize,
        timestamp: u128,
        layers: usize,
        responder: Sender<Response>,
        logits_processor: LogitsProcessor,
        stop_tokens: Vec<u32>,
        max_len: Option<usize>,
        return_logprobs: bool,
        is_xlora: bool,
        group: Rc<RefCell<SequenceGroup>>,
        response_index: usize,
    ) -> Self {
        let prompt_len = tokens.len();
        Self {
            tokens,
            logprobs: Vec::new(),
            prompt_len,
            id,
            timestamp,
            state: Cell::new(SequenceState::Waiting),
            cache: vec![None; layers],
            xlora_cache: if is_xlora {
                Some(vec![None; layers])
            } else {
                None
            },
            responder,
            logits_processor,
            stop_tokens,
            max_len,
            return_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            group,
            scaling_cache: None,
            total_sampling_time: 0,
            response_index,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub fn is_running(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
            || self.state.get() == SequenceState::RunningPrompt
    }

    pub fn is_completion(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
    }

    pub fn is_prompt(&self) -> bool {
        self.state.get() == SequenceState::RunningPrompt
    }

    pub fn is_waiting(&self) -> bool {
        self.state.get() == SequenceState::Waiting
    }

    pub fn get_toks(&self) -> &[u32] {
        &self.tokens
    }

    pub fn cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        &mut self.cache
    }

    pub fn xlora_cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        self.xlora_cache.as_mut().unwrap()
    }

    pub fn scaling_cache(&mut self) -> &mut Option<Tensor> {
        &mut self.scaling_cache
    }

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    pub fn logits_processor(&mut self) -> &mut LogitsProcessor {
        &mut self.logits_processor
    }

    pub fn add_token(&mut self, tok: Logprobs) {
        self.tokens.push(tok.token);
        self.logprobs.push(tok);
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub fn set_state(&self, state: SequenceState) {
        if matches!(state, SequenceState::Error) {
            deref_mut_refcell!(self.group).n_choices -= 1;
        }
        self.state.set(state);
    }

    pub fn is_done(&self, tok: u32, eos_tok: u32, max_model_len: usize) -> Option<StopReason> {
        if tok == eos_tok {
            Some(StopReason::Eos)
        } else if self.stop_tokens.contains(&tok) {
            Some(StopReason::StopTok(tok))
        } else if self.max_len.is_some()
            && self.tokens.len().saturating_sub(self.prompt_len) == self.max_len.unwrap()
        {
            // add_token was already called
            Some(StopReason::Length(self.max_len.unwrap()))
        } else if self.tokens.len().saturating_sub(self.prompt_len) == max_model_len {
            Some(StopReason::ModelLength(max_model_len))
        } else {
            None
        }
    }

    pub fn logprobs(&self) -> &[Logprobs] {
        &self.logprobs
    }

    pub fn return_logprobs(&self) -> bool {
        self.return_logprobs
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }

    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }

    pub fn prompt_timestamp(&self) -> Option<u128> {
        self.prompt_timestamp
    }

    pub fn add_choice_to_group(&self, choice: Choice) {
        deref_mut_refcell!(self.group).choices.push(choice);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();

        deref_mut_refcell!(self.group).total_completion_time +=
            now - self.prompt_timestamp.unwrap();
        deref_mut_refcell!(self.group).total_prompt_time +=
            self.prompt_timestamp.unwrap() - self.timestamp;
        deref_mut_refcell!(self.group).total_time += now - self.timestamp;

        deref_mut_refcell!(self.group).total_prompt_toks += self.prompt_len;
        deref_mut_refcell!(self.group).total_toks += self.len();

        deref_mut_refcell!(self.group).total_sampling_time += self.total_sampling_time;
    }

    pub fn get_response_index(&self) -> usize {
        self.response_index
    }

    pub fn get_group(&self) -> Ref<'_, SequenceGroup> {
        deref_refcell!(self.group)
    }

    pub fn get_mut_group(&self) -> RefMut<'_, SequenceGroup> {
        deref_mut_refcell!(self.group)
    }

    pub fn add_streaming_chunk_choice_to_group(&self, chunk: ChunkChoice) {
        deref_mut_refcell!(self.group).streaming_chunks.push(chunk);
    }
}

pub struct SequenceGroup {
    n_choices: usize, // The target number of choices to return. Can be decreased if an error is thrown.
    pub total_prompt_toks: usize,
    pub total_toks: usize,
    pub total_prompt_time: u128,
    pub total_time: u128,
    pub total_completion_time: u128,
    pub total_sampling_time: u128,
    choices: Vec<Choice>,
    pub streaming_chunks: Vec<ChunkChoice>,
    pub is_streaming: bool,
}

impl SequenceGroup {
    pub fn new(n_choices: usize, is_streaming: bool) -> Self {
        Self {
            choices: Vec::new(),
            n_choices,
            total_prompt_toks: 0,
            total_toks: 0,
            total_prompt_time: 0,
            total_time: 0,
            total_completion_time: 0,
            total_sampling_time: 0,
            streaming_chunks: Vec::new(),
            is_streaming,
        }
    }

    pub fn get_choices(&self) -> &[Choice] {
        &self.choices
    }

    pub fn get_usage(&self) -> ChatCompletionUsage {
        #[allow(clippy::cast_precision_loss)]
        ChatCompletionUsage {
            completion_tokens: self.total_toks - self.total_prompt_toks,
            prompt_tokens: self.total_prompt_toks,
            total_tokens: self.total_toks,
            avg_tok_per_sec: (self.total_toks as f32 / self.total_time as f32) * 1000.,
            avg_prompt_tok_per_sec: (self.total_prompt_toks as f32 / self.total_prompt_time as f32)
                * 1000.,
            avg_compl_tok_per_sec: ((self.total_toks - self.total_prompt_toks) as f32
                / self.total_completion_time as f32)
                * 1000.,
            avg_sample_tok_per_sec: (self.total_toks as f32 / self.total_sampling_time as f32)
                * 1000.,
        }
    }

    pub fn maybe_send_done_response(
        &self,
        response: ChatCompletionResponse,
        sender: Sender<Response>,
    ) {
        if self.choices.len() == self.n_choices {
            // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
            sender.send(Response::Done(response)).unwrap();
        }
    }

    pub fn maybe_send_streaming_response(&mut self, seq: &Sequence, model: String) {
        if self.streaming_chunks.len() == self.n_choices && self.is_streaming {
            seq.responder()
                .send(Response::Chunk(ChatCompletionChunkResponse {
                    id: seq.id.to_string(),
                    choices: self.streaming_chunks.clone(),
                    created: seq.timestamp,
                    model: model.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "chat.completion.chunk".to_string(),
                }))
                .unwrap();
            self.streaming_chunks.clear();
        }
    }
}
