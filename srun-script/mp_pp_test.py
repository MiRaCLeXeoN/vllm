from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/scr/dataset/yuke/zepeng/models/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0",
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    device="cuda",
    # speculative_config={
    #     "model": "facebook/opt-125m",
    #     "method": "eagle",
    #     "num_speculative_tokens": 5,
    # },
    distributed_executor_backend="mp"    # NOTE(zt): mp-pp will replace mp in the __post_init__ ParallelConfig
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
