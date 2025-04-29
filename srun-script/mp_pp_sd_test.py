from vllm import LLM, SamplingParams

prompts_file_path = "prompts.txt"
with open(prompts_file_path, "r") as f:
    prompts = f.readlines()

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    device="cuda",
    # speculative_config={
    #     "model": "facebook/opt-125m",
    #     "method": "ngram",
    #     "num_speculative_tokens": 5,
    #     "prompt_lookup_max": 10,
    #     "prompt_lookup_min": 1,
    # },
    speculative_config={
        # "model": "ibm-ai-platform/llama3-70b-accelerator",
        # "model": "facebook/opt-125m",
        "model" : "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "method": "eagle",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 5,
    },
    distributed_executor_backend="mp"    # NOTE(zt): mp-pp will replace mp in the __post_init__ ParallelConfig
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
