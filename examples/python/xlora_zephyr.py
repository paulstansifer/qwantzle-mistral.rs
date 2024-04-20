from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.XLoraMistralGGUF(
        tok_model_id="HuggingFaceH4/zephyr-7b-beta",
        quantized_model_id="TheBloke/zephyr-7B-beta-GGUF",
        quantized_filename="zephyr-7b-beta.Q4_0.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
        xlora_model_id="lamm-mit/x-lora",
        order="orderings/xlora-paper-ordering.json",
        tgt_non_granular_index=None,
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[{"role": "user", "content": "What is graphene?"}],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.5,
    )
)
print(res)
