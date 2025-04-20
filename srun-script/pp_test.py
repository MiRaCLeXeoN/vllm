from vllm import LLM, SamplingParams

def main():
    prompts = ["The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    try:
        llm = LLM(
            model="facebook/opt-30b",       # 推荐 30B 级模型，Fit 4×32GB V100
            tensor_parallel_size=1,         # 不做张量并行
            pipeline_parallel_size=4,       # 4 GPU 做 4 段流水线
            # distributed_executor_backend="mp"  # mp 是默认；可显式指定
            device="cuda"
        )
    except Exception as e:
        print(f"LLM init error: {e}")
        return

    try:
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            print("Prompt:", output.prompt)
            print("Generated text:", output.outputs[0].text)
    except Exception as e:
        print(f"Generation error: {e}")

if __name__ == '__main__':
    main()
